//! COELANOX packager for creating containers
//! 
//! This module provides the main packaging functionality for COELANOX, including
//! model translation, optimization, and container creation.

use coelanox_core::{
    CoelanoxError, CoelanoxResult,
    format::{BackendMetadata, BackendSelectionMode, CoelanoxContainer, CoelanoxManifest, CompressionType, HardwareTarget, OptimizationLevel},
    ir::{GraphNode, IntermediateTensor, OpType as IrOpType, UniversalIR, WeightTensor},
    memory_hal::MemoryBackendRegistry,
    optimization::MaxShapeReader,
    utils::SystemEnvironment,
};
use clf::{ClfReader as ClfFileReader, MissingOpIdPolicy, op_type_to_clf_id as clf_op_type_to_clf_id, OpType as ClfOpType};
use coelanox_orchestrator::{BackendManager, BackendKind};
use coelanox_executor::build_plan_from_ir;
use coelanox_core::ir::TensorId;
use std::collections::{HashMap, HashSet, VecDeque};
use crate::memory::NvmeAllocator;
// CodeGenerator trait removed - generators folder deleted
use crate::optimizers::HardwareOptimizer;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Map coelanox-core IR OpType to CLF crate OpType for canonical op_id lookup.
fn ir_op_type_to_clf_op_type(op_type: &IrOpType) -> ClfOpType {
    use IrOpType as Ir;
    use ClfOpType as Clf;
    match op_type {
        Ir::Add => Clf::Add,
        Ir::Subtract => Clf::Subtract,
        Ir::Multiply => Clf::Multiply,
        Ir::Divide => Clf::Divide,
        Ir::Relu => Clf::Relu,
        Ir::Sigmoid => Clf::Sigmoid,
        Ir::Tanh => Clf::Tanh,
        Ir::Softmax => Clf::Softmax,
        Ir::LogSoftmax => Clf::LogSoftmax,
        Ir::Gelu => Clf::Gelu,
        Ir::Swish => Clf::Swish,
        Ir::Sqrt => Clf::Sqrt,
        Ir::Pow => Clf::Pow,
        Ir::Cos => Clf::Cos,
        Ir::Sin => Clf::Sin,
        Ir::Exp => Clf::Exp,
        Ir::Log => Clf::Log,
        Ir::Convolution => Clf::Convolution,
        Ir::MaxPool => Clf::MaxPool,
        Ir::AvgPool => Clf::AvgPool,
        Ir::GlobalMaxPool => Clf::GlobalMaxPool,
        Ir::GlobalAvgPool => Clf::GlobalAvgPool,
        Ir::BatchNorm => Clf::BatchNorm,
        Ir::LayerNorm => Clf::LayerNorm,
        Ir::Dropout => Clf::Dropout,
        Ir::Reshape => Clf::Reshape,
        Ir::Transpose => Clf::Transpose,
        Ir::Permute => Clf::Permute,
        Ir::Concatenate => Clf::Concatenate,
        Ir::Split => Clf::Split,
        Ir::Slice => Clf::Slice,
        Ir::Gather => Clf::Gather,
        Ir::Scatter => Clf::Scatter,
        Ir::MatMul => Clf::MatMul,
        Ir::Gemm => Clf::Gemm,
        Ir::ReduceSum => Clf::ReduceSum,
        Ir::ReduceMean => Clf::ReduceMean,
        Ir::ReduceMax => Clf::ReduceMax,
        Ir::ReduceMin => Clf::ReduceMin,
        Ir::ReduceProd => Clf::ReduceProd,
        Ir::Broadcast | Ir::Expand => Clf::Unknown,
        Ir::Equal => Clf::Equal,
        Ir::NotEqual => Clf::NotEqual,
        Ir::Greater => Clf::Greater,
        Ir::GreaterEqual => Clf::GreaterEqual,
        Ir::Less => Clf::Less,
        Ir::LessEqual => Clf::LessEqual,
        Ir::And => Clf::And,
        Ir::Or => Clf::Or,
        Ir::Not => Clf::Not,
        Ir::Custom(s) => Clf::Custom(s.bytes().fold(0u32, |a, b| a.wrapping_add(b as u32)) % 0xFE + 256),
    }
}

/// Serializable version of OptimizedIR (without Arc)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedOptimizedIR {
    target: HardwareTarget,
    optimized_nodes: Vec<GraphNode>,
    intermediate_tensors: Vec<IntermediateTensor>,
    optimization_metadata: HashMap<String, String>,
    original_model_name: String,
    weight_tensors: Vec<WeightTensor>,
}

pub mod plugins;
pub mod translators;
pub mod optimizers;
pub mod optimization;
pub mod backend_trait;
pub mod utils;
pub mod memory;

/// COELANOX packager for creating containers
pub struct CoelanoxPackager {
    /// Plugin manager for translators
    plugin_manager: plugins::PluginManager,
    /// Environment information
    environment: SystemEnvironment,
    /// Backend manager for discovering vendor BackendTranslators
    backend_manager: BackendManager,
    /// Memory HAL registry for storage backends (NVMe, SD, OS). When a backend is registered
    /// for storage, container backing is created via HAL; otherwise NvmeAllocator is used.
    memory_registry: MemoryBackendRegistry,
}

impl CoelanoxPackager {
    /// Create a new COELANOX packager
    pub fn new() -> CoelanoxResult<Self> {
        let mut plugin_manager = plugins::PluginManager::new();
        
        // Load default plugins (none for now - community will add them)
        plugin_manager.load_default_plugins()?;
        
        // Detect environment
        let environment = SystemEnvironment::detect()?;
        
        // Initialize backend manager and discover vendor backends
        let mut backend_manager = BackendManager::new();
        let search_paths = BackendManager::default_search_paths();
        let discovery_result = backend_manager.discover_backends(&search_paths)?;
        // Log discovery results (failures are tracked but don't stop initialization)
        if discovery_result.failure_count() > 0 {
            eprintln!("⚠️  Backend discovery: {} succeeded, {} failed", 
                discovery_result.success_count(), 
                discovery_result.failure_count());
        }
        
        Ok(Self {
            plugin_manager,
            environment,
            backend_manager,
            memory_registry: MemoryBackendRegistry::new(),
        })
    }

    /// Access the memory HAL registry to register storage backends (e.g. NVMe, SD, OS).
    /// When `default_for_storage()` returns a backend, container backing is created via HAL.
    pub fn memory_registry(&mut self) -> &mut MemoryBackendRegistry {
        &mut self.memory_registry
    }
    
    /// Compress weights using available compression algorithms
    #[cfg(feature = "packaging")]
    fn compress_weights(&self, weights: &[f32]) -> CoelanoxResult<(Vec<u8>, CompressionType)> {
        println!("🗜️  Starting dynamic compression for {} weights", weights.len());
        
        // For very large weight sets, use streaming compression
        if weights.len() > 10_000_000 { // 10M weights threshold
            return self.compress_weights_streaming(weights);
        }
        
        // For smaller weight sets, use standard compression
        self.compress_weights_standard(weights)
    }
    
    #[cfg(feature = "packaging")]
    fn compress_weights_streaming(&self, weights: &[f32]) -> CoelanoxResult<(Vec<u8>, CompressionType)> {
        println!("🌊 Using streaming compression for large weight set ({}M weights)", weights.len() / 1_000_000);
        
        // Convert weights to bytes in chunks to avoid memory explosion
        let mut compressed_data = Vec::new();
        let chunk_size = 1_000_000; // 1M weights per chunk
        
        for chunk_start in (0..weights.len()).step_by(chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + chunk_size, weights.len());
            let weight_chunk = &weights[chunk_start..chunk_end];
            
            // Convert chunk to bytes
            let mut chunk_bytes = Vec::with_capacity(weight_chunk.len() * 4);
            for &weight in weight_chunk {
                chunk_bytes.extend_from_slice(&weight.to_le_bytes());
            }
            
            // Compress chunk
            #[cfg(feature = "zstd")]
            match zstd::encode_all(&*chunk_bytes, 1) { // Lower compression level for speed
                Ok(compressed_chunk) => {
                    compressed_data.extend_from_slice(&compressed_chunk);
                }
                Err(e) => {
                    println!("⚠️  Zstd compression failed for chunk, using LZ4: {}", e);
                    #[cfg(feature = "lz4")]
                    match lz4::block::compress(&chunk_bytes, Some(lz4::block::CompressionMode::FAST(1)), false) {
                        Ok(compressed_chunk) => {
                            compressed_data.extend_from_slice(&compressed_chunk);
                        }
                        Err(e) => {
                            println!("⚠️  LZ4 compression failed for chunk, using raw: {}", e);
                            compressed_data.extend_from_slice(&chunk_bytes);
                        }
                    }
                }
            }
            
            if chunk_start % 5_000_000 == 0 { // Progress every 5M weights
                println!("📊 Compressed {}M weights", chunk_start / 1_000_000);
            }
        }
        
        println!("✅ Streaming compression complete: {} bytes from {}M weights", compressed_data.len(), weights.len() / 1_000_000);
        Ok((compressed_data, CompressionType::Zstd))
    }
    
    #[cfg(feature = "packaging")]
    fn compress_weights_standard(&self, weights: &[f32]) -> CoelanoxResult<(Vec<u8>, CompressionType)> {
        println!("📦 Using standard compression for {} weights", weights.len());
        
        // Debug: Check if packaging feature is enabled
        #[cfg(feature = "packaging")]
        println!("✅ Packaging feature is ENABLED - compression available");
        #[cfg(not(feature = "packaging"))]
        println!("❌ Packaging feature is DISABLED - no compression available");
        
        // Convert weights to bytes
        let mut weight_bytes = Vec::with_capacity(weights.len() * 4);
        for &weight in weights {
            weight_bytes.extend_from_slice(&weight.to_le_bytes());
        }
        
        // Debug: Analyze weight distribution
        let mut zero_count = 0;
        let mut small_count = 0;
        let mut large_count = 0;
        for &weight in weights.iter().take(1000) { // Sample first 1000 weights
            if weight == 0.0 { zero_count += 1; }
            else if weight.abs() < 0.1 { small_count += 1; }
            else if weight.abs() > 10.0 { large_count += 1; }
        }
        println!("🔍 Weight analysis (first 1000): zeros={}, small(<0.1)={}, large(>10)={}", 
                zero_count, small_count, large_count);
        
        // Try different compression algorithms and choose the best
        let mut best_compressed = weight_bytes.clone();
        let mut best_type = CompressionType::None;
        let mut best_size = weight_bytes.len();
        
        // Try Zstd (use higher compression level)
        #[cfg(feature = "packaging")]
        {
            println!("🔧 Attempting Zstd compression...");
            if let Ok(zstd_compressed) = zstd::encode_all(&*weight_bytes, 3) { // Level 3 for better compression
                println!("🔧 Zstd compressed: {} bytes (from {} bytes)", zstd_compressed.len(), weight_bytes.len());
                if zstd_compressed.len() < best_size {
                    best_compressed = zstd_compressed;
                    best_type = CompressionType::Zstd;
                    best_size = best_compressed.len();
                    println!("✅ Zstd is best so far: {} bytes", best_size);
                }
            } else {
                println!("❌ Zstd compression failed");
            }
        }
        #[cfg(not(feature = "packaging"))]
        {
            println!("⚠️  Packaging feature not enabled - no compression available");
        }
        
        // Try LZ4
        #[cfg(feature = "packaging")]
        {
            println!("🔧 Attempting LZ4 compression...");
            if let Ok(lz4_compressed) = lz4::block::compress(&weight_bytes, Some(lz4::block::CompressionMode::FAST(1)), false) {
                println!("🔧 LZ4 compressed: {} bytes (from {} bytes)", lz4_compressed.len(), weight_bytes.len());
                if lz4_compressed.len() < best_size {
                    best_compressed = lz4_compressed;
                    best_type = CompressionType::Lz4;
                    best_size = best_compressed.len();
                    println!("✅ LZ4 is best so far: {} bytes", best_size);
                }
            } else {
                println!("❌ LZ4 compression failed");
            }
        }
        
        // Skip deflate compression for now (has reporting bug)
        // TODO: Fix deflate compression reporting
        
        let compression_ratio = weight_bytes.len() as f64 / best_size as f64;
        println!("✅ Standard compression complete: {} bytes ({}x compression)", best_size, compression_ratio);
        
        Ok((best_compressed, best_type))
    }
    
    /// Compress weights (fallback when compression features are not available)
    #[cfg(not(feature = "packaging"))]
    fn compress_weights(&self, weights: &[f32]) -> CoelanoxResult<(Vec<u8>, CompressionType)> {
        // Convert f32 weights to bytes (no compression)
        let mut weight_bytes = Vec::new();
        for &weight in weights {
            weight_bytes.extend_from_slice(&weight.to_le_bytes());
        }
        Ok((weight_bytes, CompressionType::None))
    }
    
    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8], format: &coelanox_core::format::ModelFormat) -> CoelanoxResult<Vec<f32>> {
        match format {
            coelanox_core::format::ModelFormat::Onnx => {
                // Get ONNX plugin
                let onnx_plugin = self.plugin_manager.find_plugin(coelanox_core::format::ModelFormat::Onnx)
                    .ok_or_else(|| CoelanoxError::format("ONNX plugin not found"))?;
                
                // Extract weights using ONNX plugin (which has its own fallback mechanism)
                onnx_plugin.extract_weights(model_data)
            }
            coelanox_core::format::ModelFormat::Custom(name) if name == "resnet_tiny_mnist" => {
                let plugin = self.plugin_manager.find_plugin(format.clone())
                    .ok_or_else(|| CoelanoxError::format("ResNet-tiny-MNIST plugin not found"))?;
                plugin.extract_weights(model_data)
            }
            _ => {
                Err(CoelanoxError::format(&format!("Weight extraction not supported for format: {:?}", format)))
            }
        }
    }
    
    /// Package a model into a COELANOX container
    /// 
    /// # Arguments
    /// * `model_path` - Path to input model file
    /// * `target` - Hardware target for code generation
    /// * `target_str` - Original target string from CLI (for backend metadata naming)
    /// * `model_type` - Type of model (cv, nlp, etc.)
    /// * `custom_input_shapes` - Custom input shapes (optional)
    /// * `custom_output_shapes` - Custom output shapes (optional)
    /// * `use_demo_translator` - Use demo/custom translator
    /// * `fat` - Include all backends (fat container)
    /// * `auto` - Auto-detect backends
    /// * `backend_add` - Additional backends to add
    pub fn package(&self, model_path: &Path, target: HardwareTarget, target_str: &str, model_type: &str, custom_input_shapes: Option<&str>, custom_output_shapes: Option<&str>, use_demo_translator: bool, fat: bool, auto: bool, backend_add: Option<&str>) -> CoelanoxResult<CoelanoxContainer> {
        // If directory given, use ResNet bundle inside it (no PyTorch needed at package time)
        let actual_path: std::path::PathBuf = if model_path.is_dir() {
            let bundle = model_path.join("resnet_tiny_mnist.bundle");
            if bundle.exists() {
                bundle
            } else {
                return Err(CoelanoxError::format(
                    "Directory given but resnet_tiny_mnist.bundle not found. Create it first (e.g. python export_resnet_bundle.py <dir>) or pass the .bundle file path."
                ));
            }
        } else {
            model_path.to_path_buf()
        };
        let model_path = actual_path.as_path();

        eprintln!("📦 Starting package for: {}", model_path.display());
        eprintln!("📦 Model path: {:?}", model_path);
        
        // Detect model format (read just first few bytes for format detection)
        println!("🔍 Detecting model format...");
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut file, &mut magic)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        let format = self.detect_format(&magic)?;
        println!("✅ Detected format: {:?}", format);
        
        // Re-read full file (we need it later for weight extraction)
        println!("📖 Reading full model file...");
        let model_data = std::fs::read(model_path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        println!("✅ File read successfully ({}MB)", model_data.len() / (1024 * 1024));
        
        // Get appropriate translator
        println!("🔍 Looking for translator for format: {:?}", format);
        let translator = self.plugin_manager.find_plugin_with_preference(format.clone(), Some(use_demo_translator))
            .ok_or_else(|| CoelanoxError::UnsupportedModelFormat(format!("No translator for format: {:?}", format)))?;
        println!("✅ Found translator: {} ({})", translator.name(), if use_demo_translator { "demo/custom translator" } else { "standard translator" });
        
        // Translate model to Universal IR using file path (supports sharded models)
        println!("🔄 Starting translation from path: {}", model_path.display());
        let mut ir = translator.translate_from_path(model_path)?;
        println!("✅ Translation complete");

        // Sort IR into topological (execution) order before optimization
        ir.sort_nodes_for_execution()
            .map_err(|e| CoelanoxError::Runtime(format!("IR sort for execution failed: {}", e)))?;
        
        // Determine optimization level based on environment
        let optimization_level = self.determine_optimization_level(&target);
        
        // Optimize IR for target hardware
        let optimized_ir = self.optimize_ir(ir.clone(), &target, optimization_level)?;
        
        // Generate machine code (and kernel offsets for CLF execution plan)
        let (machine_code, kernel_byte_offsets) = self.generate_machine_code(&optimized_ir, &target, &ir.metadata.input_shapes)?;
        
        // Extract weights from model
        let weights = self.extract_weights(&model_data, &format)?;
        
        // Compress weights
        let (compressed_weights, compression_type) = self.compress_weights(&weights)?;
        
        // For serialization, we need to create a copy without Arc
        // Create weight tensors WITHOUT data (weights are already in compressed_weights)
        // This saves significant space - weights are stored once, compressed
        let weight_tensors_metadata: Vec<WeightTensor> = optimized_ir.shared_weights.as_ref().iter()
            .map(|wt| {
                // Create a new WeightTensor with same metadata but empty data
                // Runtime will reconstruct weights from compressed_weights
                WeightTensor::new(wt.name.clone(), wt.shape.clone(), wt.data_type.clone())
            })
            .collect();
        
        // Create a serializable version with weight metadata only (no data duplication)
        let serializable_ir = SerializedOptimizedIR {
            target: optimized_ir.target.clone(),
            optimized_nodes: optimized_ir.optimized_nodes.clone(),
            intermediate_tensors: optimized_ir.intermediate_tensors.clone(),
            optimization_metadata: optimized_ir.optimization_metadata.clone(),
            original_model_name: optimized_ir.original_model_name.clone(),
            weight_tensors: weight_tensors_metadata,
        };
        
        // Serialize IR for runtime use
        let ir_data = bincode::serialize(&serializable_ir)
            .map_err(|e| CoelanoxError::Runtime(format!("Failed to serialize IR: {}", e)))?;
        
        // Determine AOT backend selection
        let backend_metadata = self.determine_backend_selection(
            &target,
            target_str,
            backend_add,
            fat,
            auto,
        )?;
        
        // Read max shapes from IR for container backing (stack arena / pre-allocated region)
        let max_shapes = MaxShapeReader::read_max_shapes(&ir)?;
        
        // Create backing via Memory HAL when a storage backend is registered; else NvmeAllocator
        let nvme_backing_data = if let Some((_, backend)) = self.memory_registry.default_for_storage() {
            backend.create_backing(&max_shapes)?
        } else {
            NvmeAllocator::create_nvme_backing(&max_shapes)?
        };
        
        // Create manifest with backend metadata first (to calculate its size)
        let mut manifest = self.create_manifest(&ir, machine_code.len(), backend_metadata)?;
        
        // Calculate NVMe backing offset (after IR data)
        // Structure: Header + Manifest + Code + Weights (with 4-byte prefix) + IR (with 4-byte prefix) + NVMe (with 8-byte prefix)
        let manifest_size = serde_json::to_vec(&manifest).unwrap_or_default().len();
        let code_offset = coelanox_core::format::COELANOX_HEADER_SIZE as u64 + manifest_size as u64;
        let weights_offset = code_offset + machine_code.len() as u64;
        let ir_offset = weights_offset + 4 + compressed_weights.len() as u64;
        let nvme_backing_offset = ir_offset + 4 + ir_data.len() as u64;
        
        // Update manifest with NVMe fields
        manifest.nvme_backing_offset = Some(nvme_backing_offset);
        manifest.nvme_backing_size = Some(nvme_backing_data.len());
        
        // Build execution plan for CLF multi-blob (mini-executor)
        let execution_plan = if let Some(ref kernel_offsets_by_node) = kernel_byte_offsets {
            let node_order = Self::execution_node_order(&optimized_ir)?;
            // op_ids and kernel offsets in execution order (step j = node_order[j])
            let op_ids: Vec<u32> = node_order
                .iter()
                .map(|&i| clf_op_type_to_clf_id(ir_op_type_to_clf_op_type(&optimized_ir.optimized_nodes[i].op_type)))
                .collect();
            let offsets_in_exec_order: Vec<u32> = node_order
                .iter()
                .map(|&i| kernel_offsets_by_node[i])
                .collect();
            let weight_offsets: Vec<usize> = {
                let mut w = vec![0];
                for wt in optimized_ir.shared_weights.iter() {
                    w.push(w.last().copied().unwrap_or(0) + wt.elements());
                }
                w
            };
            let input_elements: usize = ir.metadata.input_shapes.iter().map(|s| s.total_elements()).sum();
            let output_elements: usize = ir.metadata.output_shapes.iter().map(|s| s.total_elements()).sum();
            let (plan, _ws) = build_plan_from_ir(
                &optimized_ir.optimized_nodes,
                &optimized_ir.intermediate_tensors,
                &weight_offsets,
                input_elements,
                output_elements,
                &node_order,
                &op_ids,
                &offsets_in_exec_order,
            )?;
            Some(plan)
        } else {
            None
        };
        
        // Create container with weights, IR data, NVMe backing, and optional execution plan
        let container = CoelanoxContainer::new_with_hal_data_and_plan(
            manifest,
            machine_code,
            compressed_weights,
            compression_type,
            ir_data,
            Some(nvme_backing_data),
            None,
            None,
            execution_plan,
        );
        
        Ok(container)
    }
    
    /// Package with automatic target detection
    pub fn package_auto(&self, model_path: &Path) -> CoelanoxResult<CoelanoxContainer> {
        let recommended_target = self.environment.recommended_target();
        self.package(model_path, recommended_target, "custom", "auto", None, None, false, false, false, None)
    }
    
    /// Detect model format from data
    fn detect_format(&self, model_data: &[u8]) -> CoelanoxResult<coelanox_core::format::ModelFormat> {
        if model_data.len() >= 4 {
            // ResNet-tiny-MNIST bundle (RTM1 magic)
            if model_data.len() >= 4 && model_data.starts_with(b"RTM1") {
                return Ok(coelanox_core::format::ModelFormat::Custom("resnet_tiny_mnist".to_string()));
            }
            // ONNX files start with protobuf magic (08 is the protobuf field tag)
            // The second byte can vary, so we just check for 08 as the first byte
            if model_data[0] == 0x08 {
                return Ok(coelanox_core::format::ModelFormat::Onnx);
            }
            
            // Check for PyTorch magic (ZIP format) - REMOVED
            // if model_data[0] == 0x50 && model_data[1] == 0x4B && model_data[2] == 0x03 && model_data[3] == 0x04 {
            //     return Ok(coelanox_core::format::ModelFormat::PyTorch);
            // }
            
            // Check for TensorFlow SavedModel (directory structure) - REMOVED
            // if model_data.len() >= 8 {
            //     let header = &model_data[0..8];
            //     if header == b"SavedModel" {
            //         return Ok(coelanox_core::format::ModelFormat::TensorFlow);
            //     }
            // }
        }
        
        // TensorFlow detection removed - focusing on ONNX only
        // Check for TensorFlow protobuf
        // if model_data.len() >= 4 && model_data[0] == 0x08 && model_data[1] == 0x00 && model_data[2] == 0x00 && model_data[3] == 0x00 {
        //     // Look for TensorFlow-specific markers
        //     if model_data.windows(4).any(|window| window == b"node") {
        //         return Ok(coelanox_core::format::ModelFormat::TensorFlow);
        //     }
        // }
        
        // Fallback to ONNX if we can't determine
        Ok(coelanox_core::format::ModelFormat::Onnx)
    }
    
    /// Determine optimization level based on environment and target
    fn determine_optimization_level(&self, target: &HardwareTarget) -> OptimizationLevel {
        match target {
            HardwareTarget::Cpu => {
                if self.environment.cpu_cores >= 8 && self.environment.simd_capabilities.iter().any(|c| matches!(c, coelanox_core::utils::SimdCapability::AVX2)) {
                    OptimizationLevel::Maximum
                } else if self.environment.cpu_cores >= 4 {
                    OptimizationLevel::Advanced
                } else {
                    OptimizationLevel::Basic
                }
            }
            HardwareTarget::Gpu => {
                if self.environment.gpu_info.is_some() {
                    OptimizationLevel::Maximum
                } else {
                    OptimizationLevel::Basic
                }
            }
            HardwareTarget::Edge => {
                if self.environment.simd_capabilities.iter().any(|c| matches!(c, coelanox_core::utils::SimdCapability::NEON)) {
                    OptimizationLevel::Advanced
                } else {
                    OptimizationLevel::Basic
                }
            }
            HardwareTarget::Custom(_) => OptimizationLevel::Basic,
            HardwareTarget::NeuralEngine => OptimizationLevel::Maximum,
            HardwareTarget::NPU => OptimizationLevel::Maximum,
        }
    }
    
    /// Optimize IR for target hardware
    fn optimize_ir(&self, ir: UniversalIR, target: &HardwareTarget, _level: OptimizationLevel) -> CoelanoxResult<coelanox_core::ir::OptimizedIR> {
        match target {
            HardwareTarget::Cpu => {
                let optimizer = optimizers::CpuOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
            HardwareTarget::Gpu => {
                let optimizer = optimizers::GpuOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
            HardwareTarget::Edge => {
                let optimizer = optimizers::EdgeOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
            HardwareTarget::Custom(_) => {
                // For custom targets, use CPU optimizer as fallback
                let optimizer = optimizers::CpuOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
            HardwareTarget::NeuralEngine => {
                // For Neural Engine, use CPU optimizer as fallback for now
                let optimizer = optimizers::CpuOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
            HardwareTarget::NPU => {
                // For NPU, use CPU optimizer as fallback for now
                let optimizer = optimizers::CpuOptimizer::new();
                HardwareOptimizer::optimize(&optimizer, &ir, target.clone())
            }
        }
    }
    
    /// Generate machine code for target hardware.
    /// For CLF backends, also returns kernel byte offsets (for execution plan); otherwise returns (code, None).
    fn generate_machine_code(
        &self,
        optimized_ir: &coelanox_core::ir::OptimizedIR,
        target: &HardwareTarget,
        input_shapes: &[coelanox_core::ir::Shape],
    ) -> CoelanoxResult<(Vec<u8>, Option<Vec<u32>>)> {
        if let Some(backend_info) = self.backend_manager.find_backend_for_target(target) {
            if backend_info.kind == BackendKind::Clf {
                let mut reader = ClfFileReader::open(backend_info.library_path.as_path())
                    .map_err(|e| CoelanoxError::format(&format!("CLF load failed: {}", e)))?;
                let op_ids: Vec<u32> = optimized_ir
                    .optimized_nodes
                    .iter()
                    .map(|node| clf_op_type_to_clf_id(ir_op_type_to_clf_op_type(&node.op_type)))
                    .collect();
                let mut code = Vec::new();
                let mut kernel_byte_offsets = Vec::with_capacity(op_ids.len());
                for op_id in &op_ids {
                    if let Ok(Some(blob)) = reader.get_blob(*op_id) {
                        kernel_byte_offsets.push(code.len() as u32);
                        code.extend_from_slice(&blob);
                    }
                }
                // Only return kernel offsets when we have a blob for every node (required for execution plan)
                let offsets = if kernel_byte_offsets.len() == op_ids.len() {
                    Some(kernel_byte_offsets)
                } else {
                    None
                };
                return Ok((code, offsets));
            }
        }
        let _ = (optimized_ir, input_shapes);
        Ok((Vec::new(), None))
    }

    /// Topological execution order for optimized nodes (same logic as scalar fallback).
    fn execution_node_order(optimized_ir: &coelanox_core::ir::OptimizedIR) -> CoelanoxResult<Vec<usize>> {
        let nodes = &optimized_ir.optimized_nodes;
        let produced: HashSet<TensorId> = nodes
            .iter()
            .flat_map(|n| n.output_tensors.iter().copied())
            .collect();
        let mut producer: HashMap<TensorId, usize> = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            for &tid in &node.output_tensors {
                if tid.is_intermediate() {
                    producer.insert(tid, i);
                }
            }
        }
        let mut in_degree: Vec<usize> = vec![0; nodes.len()];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            for &tid in &node.input_tensors {
                if tid.is_intermediate() && produced.contains(&tid) {
                    if let Some(&p) = producer.get(&tid) {
                        in_degree[i] += 1;
                        dependents[p].push(i);
                    }
                }
            }
        }
        let mut queue: VecDeque<usize> = (0..nodes.len()).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(nodes.len());
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for &j in &dependents[i] {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    queue.push_back(j);
                }
            }
        }
        if order.len() != nodes.len() {
            return Err(CoelanoxError::Runtime(format!(
                "Graph cycle or missing producers: sorted {} of {} nodes",
                order.len(),
                nodes.len()
            )));
        }
        Ok(order)
    }
    
    /// Create manifest from IR
    fn create_manifest(&self, ir: &UniversalIR, machine_code_size: usize, backend_metadata: Option<BackendMetadata>) -> CoelanoxResult<CoelanoxManifest> {
        let mut manifest = CoelanoxManifest::new(
            ir.metadata.name.clone(),
            coelanox_core::format::ModelFormat::Onnx, // TODO: Get from IR
            ir.metadata.input_shapes.iter().map(|s| s.dimensions.clone()).collect(),
            ir.metadata.output_shapes.iter().map(|s| s.dimensions.clone()).collect(),
            machine_code_size,
        );
        
        // Set backend metadata
        manifest.backend_metadata = backend_metadata;
        
        Ok(manifest)
    }
    
    /// Determine AOT backend selection based on CLI flags
    /// 
    /// Returns BackendMetadata describing which backends were selected and how they map to device IDs.
    /// 
    /// # Arguments
    /// * `target` - Hardware target for code generation
    /// * `target_str` - Original target string from CLI (for backend naming)
    /// * `backend_add` - Additional backends to add (comma-separated)
    /// * `fat` - Include all available backends
    /// * `auto` - Auto-detect and select backends
    fn determine_backend_selection(
        &self,
        target: &HardwareTarget,
        target_str: &str,
        backend_add: Option<&str>,
        fat: bool,
        auto: bool,
    ) -> CoelanoxResult<Option<BackendMetadata>> {
        // Fat mode: validate all available backends
        if fat {
            let all_backends = self.backend_manager.list_backends()
                .iter()
                .map(|b| b.name.clone())
                .collect::<Vec<_>>();
            
            if all_backends.is_empty() {
                return Err(CoelanoxError::BackendNotFound(
                    "No backends available for fat container".to_string()
                ));
            }
            
            return Ok(Some(BackendMetadata::new_fat(all_backends)));
        }
        
        // Auto mode: validate detected backends
        if auto {
            let recommended = self.environment.recommended_target();
            let detected_name = format!("{:?}", recommended).to_lowercase();
            
            // Validate detected backend exists
            if self.backend_manager.get_backend(&detected_name).is_none() {
                // Fallback to CPU if detected backend not available
                if self.backend_manager.get_backend("cpu").is_none() {
                    return Err(CoelanoxError::BackendNotFound(
                        format!("Auto-detected backend '{}' not available, and CPU fallback not found", detected_name)
                    ));
                }
                let detected_backends = vec!["cpu".to_string()];
                return Ok(Some(BackendMetadata::new_auto(detected_backends)));
            }
            
            let detected_backends = vec![detected_name];
            return Ok(Some(BackendMetadata::new_auto(detected_backends)));
        }
        
        // Slim mode: resolve primary backend name. For generic targets (gpu, edge), if no backend
        // is named exactly "gpu"/"edge", use any discovered backend that supports that target
        // (e.g. CLF file resnet_tiny_mnist.spv.clfc is registered as "resnet_tiny_mnist.spv" and supports Gpu).
        let primary = target_str.to_lowercase();
        // When target is "cpu", use discovered backend name if it's a CLF (e.g. resnet_tiny_mnist_simd)
        // so the runtime skips single-entry-point native execution and falls back to scalar (SIMD blobs are multi-kernel).
        let primary_resolved: String = if primary == "cpu" {
            self.backend_manager.find_backend_for_target(target)
                .map(|b| b.name.clone())
                .unwrap_or_else(|| primary.clone())
        } else if self.backend_manager.get_backend(&primary).is_some() {
            primary.clone()
        } else if let Some(backend_info) = self.backend_manager.find_backend_for_target(target) {
            backend_info.name.clone()
        } else {
            let available: Vec<String> = self.backend_manager.list_backends()
                .iter()
                .map(|b| b.name.clone())
                .collect();
            return Err(CoelanoxError::BackendNotFound(
                format!("Primary backend '{}' not found and no backend supports target {:?}. Available: {}", 
                    primary,
                    target,
                    if available.is_empty() {
                        "none".to_string()
                    } else {
                        available.join(", ")
                    }
                )
            ));
        };
        
        let additional = if let Some(add_str) = backend_add {
            let additional_list: Vec<String> = add_str.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            
            // Validate all additional backends exist
            if let Err(e) = self.backend_manager.validate_backends(&additional_list) {
                return Err(CoelanoxError::InvalidBackendName(
                    format!("Invalid backend in --backend-add: {}", e)
                ));
            }
            
            additional_list
        } else {
            Vec::new()
        };
        
        // Emit backend metadata so runtime knows whether container code is CPU or not
        // (e.g. GPU-packed container has primary_resolved = "resnet_tiny_mnist.spv"; runtime skips native execution when backend is cpu)
        Ok(Some(BackendMetadata::new_slim(primary_resolved, additional)))
    }
    
    /// Get environment information
    pub fn get_environment(&self) -> &SystemEnvironment {
        &self.environment
    }
    
    /// Get supported formats
    pub fn get_supported_formats(&self) -> Vec<coelanox_core::format::ModelFormat> {
        vec![coelanox_core::format::ModelFormat::Onnx]
    }
}