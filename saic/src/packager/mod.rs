//! SAIC packager for creating containers
//! 
//! This module provides the main packaging functionality for SAIC, including
//! model translation, optimization, and container creation.

use crate::{SaicError, SaicResult, format::{HardwareTarget, OptimizationLevel, SaicContainer, SaicManifest, CompressionType}};
use crate::utils::SystemEnvironment;
use crate::packager::generators::CodeGenerator;
use std::path::Path;
use prost::Message;

pub mod ir;
pub mod plugins;
pub mod optimizers;
pub mod generators;

/// SAIC packager for creating containers
pub struct SaicPackager {
    /// Plugin manager for translators
    plugin_manager: plugins::PluginManager,
    /// Environment information
    environment: SystemEnvironment,
}

impl SaicPackager {
    /// Create a new SAIC packager
    pub fn new() -> SaicResult<Self> {
        let mut plugin_manager = plugins::PluginManager::new();
        
        // Register default plugins
        plugin_manager.register_plugin(Box::new(plugins::OnnxPlugin::new()));
        
        // Detect environment
        let environment = SystemEnvironment::detect()?;
        
        Ok(Self {
            plugin_manager,
            environment,
        })
    }
    
    /// Compress weights using available compression algorithms
    #[cfg(feature = "packaging")]
    fn compress_weights(&self, weights: &[f32]) -> SaicResult<(Vec<u8>, CompressionType)> {
        println!("🗜️  Starting dynamic compression for {} weights", weights.len());
        
        // For very large weight sets, use streaming compression
        if weights.len() > 10_000_000 { // 10M weights threshold
            return self.compress_weights_streaming(weights);
        }
        
        // For smaller weight sets, use standard compression
        self.compress_weights_standard(weights)
    }
    
    #[cfg(feature = "packaging")]
    fn compress_weights_streaming(&self, weights: &[f32]) -> SaicResult<(Vec<u8>, CompressionType)> {
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
    fn compress_weights_standard(&self, weights: &[f32]) -> SaicResult<(Vec<u8>, CompressionType)> {
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
    fn compress_weights(&self, weights: &[f32]) -> SaicResult<(Vec<u8>, CompressionType)> {
        // Convert f32 weights to bytes (no compression)
        let mut weight_bytes = Vec::new();
        for &weight in weights {
            weight_bytes.extend_from_slice(&weight.to_le_bytes());
        }
        Ok((weight_bytes, CompressionType::None))
    }
    
    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8], format: &crate::format::ModelFormat) -> SaicResult<Vec<f32>> {
        match format {
            crate::format::ModelFormat::Onnx => {
                // Get ONNX plugin
                let onnx_plugin = self.plugin_manager.find_plugin(crate::format::ModelFormat::Onnx)
                    .ok_or_else(|| SaicError::format("ONNX plugin not found"))?;
                
                // Extract weights using ONNX plugin (which has its own fallback mechanism)
                onnx_plugin.extract_weights(model_data)
            }
            _ => {
                Err(SaicError::format(&format!("Weight extraction not supported for format: {:?}", format)))
            }
        }
    }
    
    /// Package a model into a SAIC container
    pub fn package(&self, model_path: &Path, target: HardwareTarget, model_type: &str, custom_input_shapes: Option<&str>, custom_output_shapes: Option<&str>) -> SaicResult<SaicContainer> {
        // Detect model format
        let model_data = std::fs::read(model_path)
            .map_err(|e| SaicError::FileSystem(e))?;
        
        let format = self.detect_format(&model_data)?;
        
        // Get appropriate translator
        let translator = self.plugin_manager.find_plugin(format.clone())
            .ok_or_else(|| SaicError::UnsupportedModelFormat(format!("No translator for format: {:?}", format)))?;
        
        // Translate model to Universal IR with model type context
        let universal_ir = translator.translate_with_context(&model_data, model_type, custom_input_shapes, custom_output_shapes)?;
        
        // Determine optimization level based on environment
        let optimization_level = self.determine_optimization_level(&target);
        
        // Optimize IR for target hardware
        let optimized_ir = self.optimize_ir(universal_ir.clone(), &target, optimization_level)?;
        
        // Generate machine code
        let machine_code = self.generate_machine_code(&optimized_ir, &target)?;
        
        // Extract weights from model
        let weights = self.extract_weights(&model_data, &format)?;
        
        // Compress weights
        let (compressed_weights, compression_type) = self.compress_weights(&weights)?;
        
        // Create manifest
        let manifest = self.create_manifest(&universal_ir, machine_code.len())?;
        
        // Create container with weights
        let container = SaicContainer::new(manifest, machine_code, compressed_weights, compression_type);
        
        Ok(container)
    }
    
    /// Package with automatic target detection
    pub fn package_auto(&self, model_path: &Path) -> SaicResult<SaicContainer> {
        let recommended_target = self.environment.recommended_target();
        self.package(model_path, recommended_target, "custom", None, None)
    }
    
    /// Detect model format from data
    fn detect_format(&self, model_data: &[u8]) -> SaicResult<crate::format::ModelFormat> {
        if model_data.len() >= 4 {
            // ONNX files start with protobuf magic (08 is the protobuf field tag)
            // The second byte can vary, so we just check for 08 as the first byte
            if model_data[0] == 0x08 {
                return Ok(crate::format::ModelFormat::Onnx);
            }
            
            // Check for PyTorch magic (ZIP format) - REMOVED
            // if model_data[0] == 0x50 && model_data[1] == 0x4B && model_data[2] == 0x03 && model_data[3] == 0x04 {
            //     return Ok(crate::format::ModelFormat::PyTorch);
            // }
            
            // Check for TensorFlow SavedModel (directory structure) - REMOVED
            // if model_data.len() >= 8 {
            //     let header = &model_data[0..8];
            //     if header == b"SavedModel" {
            //         return Ok(crate::format::ModelFormat::TensorFlow);
            //     }
            // }
        }
        
        // TensorFlow detection removed - focusing on ONNX only
        // Check for TensorFlow protobuf
        // if model_data.len() >= 4 && model_data[0] == 0x08 && model_data[1] == 0x00 && model_data[2] == 0x00 && model_data[3] == 0x00 {
        //     // Look for TensorFlow-specific markers
        //     if model_data.windows(4).any(|window| window == b"node") {
        //         return Ok(crate::format::ModelFormat::TensorFlow);
        //     }
        // }
        
        // Fallback to ONNX if we can't determine
        Ok(crate::format::ModelFormat::Onnx)
    }
    
    /// Determine optimization level based on environment and target
    fn determine_optimization_level(&self, target: &HardwareTarget) -> OptimizationLevel {
        match target {
            HardwareTarget::Cpu => {
                if self.environment.cpu_cores >= 8 && self.environment.simd_capabilities.iter().any(|c| matches!(c, crate::utils::SimdCapability::AVX2)) {
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
                if self.environment.simd_capabilities.iter().any(|c| matches!(c, crate::utils::SimdCapability::NEON)) {
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
    fn optimize_ir(&self, ir: ir::UniversalIR, target: &HardwareTarget, level: OptimizationLevel) -> SaicResult<ir::OptimizedIR> {
        let mut optimized_ir = ir::OptimizedIR::new(ir.clone(), target.clone());
        
        match target {
            HardwareTarget::Cpu => {
                let optimizer = optimizers::CpuOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
            HardwareTarget::Gpu => {
                let optimizer = optimizers::GpuOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
            HardwareTarget::Edge => {
                let optimizer = optimizers::EdgeOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
            HardwareTarget::Custom(_) => {
                // For custom targets, use CPU optimizer as fallback
                let optimizer = optimizers::CpuOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
            HardwareTarget::NeuralEngine => {
                // For Neural Engine, use CPU optimizer as fallback for now
                let optimizer = optimizers::CpuOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
            HardwareTarget::NPU => {
                // For NPU, use CPU optimizer as fallback for now
                let optimizer = optimizers::CpuOptimizer::new();
                let optimized_operations = optimizer.optimize(&ir.operations, level)?;
                for op in optimized_operations {
                    optimized_ir.add_optimized_operation(op);
                }
            }
        }
        
        Ok(optimized_ir)
    }
    
    /// Generate machine code for target hardware
    fn generate_machine_code(&self, optimized_ir: &ir::OptimizedIR, target: &HardwareTarget) -> SaicResult<Vec<u8>> {
        match target {
            HardwareTarget::Cpu => {
                let generator = generators::CpuCodeGenerator::new();
                generator.generate(optimized_ir)
            }
            HardwareTarget::Gpu => {
                let generator = generators::GpuCodeGenerator::new();
                generator.generate(optimized_ir)
            }
            HardwareTarget::Edge => {
                let generator = generators::EdgeCodeGenerator::new();
                generator.generate(optimized_ir)
            }
            HardwareTarget::Custom(_) => {
                // For custom targets, use CPU generator as fallback
                let generator = generators::CpuCodeGenerator::new();
                generator.generate(optimized_ir)
            }
            HardwareTarget::NeuralEngine => {
                // For Neural Engine, use CPU generator as fallback for now
                let generator = generators::CpuCodeGenerator::new();
                generator.generate(optimized_ir)
            }
            HardwareTarget::NPU => {
                // For NPU, use CPU generator as fallback for now
                let generator = generators::CpuCodeGenerator::new();
                generator.generate(optimized_ir)
            }
        }
    }
    
    /// Create manifest from IR
    fn create_manifest(&self, ir: &ir::UniversalIR, machine_code_size: usize) -> SaicResult<SaicManifest> {
        let manifest = SaicManifest::new(
            ir.metadata.name.clone(),
            crate::format::ModelFormat::Onnx, // TODO: Get from IR
            ir.metadata.input_shapes.iter().map(|s| s.dimensions.clone()).collect(),
            ir.metadata.output_shapes.iter().map(|s| s.dimensions.clone()).collect(),
            machine_code_size,
        );
        
        Ok(manifest)
    }
    
    /// Get environment information
    pub fn get_environment(&self) -> &SystemEnvironment {
        &self.environment
    }
    
    /// Get supported formats
    pub fn get_supported_formats(&self) -> Vec<crate::format::ModelFormat> {
        vec![crate::format::ModelFormat::Onnx]
    }
}
