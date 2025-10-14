//! SAIC packager for creating containers
//! 
//! This module provides the main packaging functionality for SAIC, including
//! model translation, optimization, and container creation.

use crate::{SaicError, SaicResult, format::{HardwareTarget, OptimizationLevel, SaicContainer, SaicManifest}};
use crate::utils::SystemEnvironment;
use crate::packager::generators::CodeGenerator;
use std::path::Path;

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
    
    /// Package a model into a SAIC container
    pub fn package(&self, model_path: &Path, target: HardwareTarget) -> SaicResult<SaicContainer> {
        // Detect model format
        let model_data = std::fs::read(model_path)
            .map_err(|e| SaicError::FileSystem(e))?;
        
        let format = self.detect_format(&model_data)?;
        
        // Get appropriate translator
        let translator = self.plugin_manager.find_plugin(format.clone())
            .ok_or_else(|| SaicError::UnsupportedModelFormat(format!("No translator for format: {:?}", format)))?;
        
        // Translate model to Universal IR
        let universal_ir = translator.translate(&model_data)?;
        
        // Determine optimization level based on environment
        let optimization_level = self.determine_optimization_level(&target);
        
        // Optimize IR for target hardware
        let optimized_ir = self.optimize_ir(universal_ir.clone(), &target, optimization_level)?;
        
        // Generate machine code
        let machine_code = self.generate_machine_code(&optimized_ir, &target)?;
        
        // Create manifest
        let manifest = self.create_manifest(&universal_ir, machine_code.len())?;
        
        // Create container
        let container = SaicContainer::new(manifest, machine_code);
        
        Ok(container)
    }
    
    /// Package with automatic target detection
    pub fn package_auto(&self, model_path: &Path) -> SaicResult<SaicContainer> {
        let recommended_target = self.environment.recommended_target();
        self.package(model_path, recommended_target)
    }
    
    /// Detect model format from data
    fn detect_format(&self, model_data: &[u8]) -> SaicResult<crate::format::ModelFormat> {
        if model_data.len() >= 4 {
            // ONNX files start with protobuf magic (08 08 is common)
            if (model_data[0] == 0x08 && model_data[1] == 0x08) || 
               (model_data[0] == 0x08 && model_data[1] == 0x00 && model_data[2] == 0x00 && model_data[3] == 0x00) {
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
