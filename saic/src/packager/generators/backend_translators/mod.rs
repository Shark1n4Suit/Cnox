//! Backend Translators for SAIC
//! 
//! This module provides backend translators for different hardware types.
//! Each hardware vendor implements their own translator to convert SAIC AI operations
//! to hardware-specific code.

pub mod saic_operations;
pub mod backend_translator;
pub mod cpu_backend_translator;
pub mod gpu_backend_translator;
pub mod npu_backend_translator;
pub mod tpu_backend_translator;

// Re-export main types
pub use saic_operations::{
    SaicAiOperation, SaicAiOperationParams, SaicOptimizationHint, SaicHardwareType,
    SaicAiOperationMetadata, SaicPerformanceProfile, SaicMemoryRequirements
};
pub use backend_translator::{
    BackendTranslator, SaicPerformanceEstimate, BackendTranslatorCapabilities,
    SaicHardwareFeature, SaicDataType, BackendTranslatorRegistry
};
pub use cpu_backend_translator::{CpuBackendTranslator, SimdCapability, CpuArchitecture};
pub use gpu_backend_translator::{GpuBackendTranslator, GpuType};
pub use npu_backend_translator::{NpuBackendTranslator, NpuType};
pub use tpu_backend_translator::{TpuBackendTranslator, TpuType};

use crate::format::HardwareTarget;
use crate::{SaicResult, SaicError};

/// Backend translator factory
/// 
/// Creates backend translators for different hardware types
pub struct BackendTranslatorFactory;

impl BackendTranslatorFactory {
    /// Create a backend translator for the specified hardware target
    pub fn create_translator(target: HardwareTarget) -> SaicResult<Box<dyn BackendTranslator>> {
        match target {
            HardwareTarget::Cpu => {
                Ok(Box::new(CpuBackendTranslator::new()))
            }
            HardwareTarget::Gpu => {
                Ok(Box::new(GpuBackendTranslator::new()))
            }
            HardwareTarget::NPU => {
                Ok(Box::new(NpuBackendTranslator::new()))
            }
            HardwareTarget::Custom(ref name) if name == "TPU" => {
                Ok(Box::new(TpuBackendTranslator::new()))
            }
            _ => {
                Err(SaicError::UnsupportedHardwareTarget(target))
            }
        }
    }
    
    /// Create a CPU translator for specific architecture
    pub fn create_cpu_translator(architecture: CpuArchitecture, simd_capabilities: Vec<SimdCapability>) -> Box<dyn BackendTranslator> {
        Box::new(CpuBackendTranslator::new_with_architecture(architecture, simd_capabilities))
    }
    
    /// Create a GPU translator for specific GPU type
    pub fn create_gpu_translator(gpu_type: GpuType) -> Box<dyn BackendTranslator> {
        Box::new(GpuBackendTranslator::new_with_gpu_type(gpu_type))
    }
    
    /// Create an NPU translator for specific NPU type
    pub fn create_npu_translator(npu_type: NpuType) -> Box<dyn BackendTranslator> {
        Box::new(NpuBackendTranslator::new_with_npu_type(npu_type))
    }
    
    /// Create a TPU translator for specific TPU type
    pub fn create_tpu_translator(tpu_type: TpuType) -> Box<dyn BackendTranslator> {
        Box::new(TpuBackendTranslator::new_with_tpu_type(tpu_type))
    }
    
    /// Get all supported hardware targets
    pub fn supported_hardware_targets() -> Vec<HardwareTarget> {
        vec![
            HardwareTarget::Cpu,
            HardwareTarget::Gpu,
            HardwareTarget::NPU,
            HardwareTarget::Custom("TPU".to_string()),
        ]
    }
    
    /// Check if hardware target is supported
    pub fn is_hardware_supported(target: &HardwareTarget) -> bool {
        Self::supported_hardware_targets().contains(target)
    }
}

/// Backend translator manager
/// 
/// Manages multiple backend translators and provides unified interface
pub struct BackendTranslatorManager {
    registry: BackendTranslatorRegistry,
}

impl BackendTranslatorManager {
    /// Create a new backend translator manager
    pub fn new() -> Self {
        let mut registry = BackendTranslatorRegistry::new();
        
        // Register default translators
        registry.register_translator(
            HardwareTarget::Cpu,
            BackendTranslatorFactory::create_translator(HardwareTarget::Cpu).unwrap()
        );
        
        registry.register_translator(
            HardwareTarget::Gpu,
            BackendTranslatorFactory::create_translator(HardwareTarget::Gpu).unwrap()
        );
        
        registry.register_translator(
            HardwareTarget::NPU,
            BackendTranslatorFactory::create_translator(HardwareTarget::NPU).unwrap()
        );
        
        registry.register_translator(
            HardwareTarget::Custom("TPU".to_string()),
            BackendTranslatorFactory::create_translator(HardwareTarget::Custom("TPU".to_string())).unwrap()
        );
        
        Self { registry }
    }
    
    /// Get the backend translator registry
    pub fn registry(&self) -> &BackendTranslatorRegistry {
        &self.registry
    }
    
    /// Get mutable reference to the backend translator registry
    pub fn registry_mut(&mut self) -> &mut BackendTranslatorRegistry {
        &mut self.registry
    }
    
    /// Translate operation to hardware-specific code
    pub fn translate_operation(&self, operation: &SaicAiOperation, params: &SaicAiOperationParams, target: HardwareTarget) -> SaicResult<String> {
        self.registry.translate_operation(operation, params, target)
    }
    
    /// Get performance estimate for operation
    pub fn get_performance_estimate(&self, operation: &SaicAiOperation, params: &SaicAiOperationParams, target: HardwareTarget) -> SaicResult<SaicPerformanceEstimate> {
        self.registry.get_performance_estimate(operation, params, target)
    }
    
    /// List supported hardware targets
    pub fn list_supported_hardware(&self) -> Vec<HardwareTarget> {
        self.registry.list_supported_hardware()
    }
    
    /// Check if hardware target is supported
    pub fn is_hardware_supported(&self, target: &HardwareTarget) -> bool {
        self.registry.is_hardware_supported(target)
    }
    
    /// Check if operation is supported on hardware
    pub fn is_operation_supported(&self, operation: &SaicAiOperation, target: &HardwareTarget) -> bool {
        self.registry.is_operation_supported(operation, target)
    }
    
    /// Get translator capabilities
    pub fn get_capabilities(&self, target: &HardwareTarget) -> Option<BackendTranslatorCapabilities> {
        self.registry.get_capabilities(target)
    }
    
    /// Register a custom backend translator
    pub fn register_custom_translator(&mut self, target: HardwareTarget, translator: Box<dyn BackendTranslator>) {
        self.registry.register_translator(target, translator);
    }
    
    /// Unregister a backend translator
    pub fn unregister_translator(&mut self, target: &HardwareTarget) -> Option<Box<dyn BackendTranslator>> {
        self.registry.remove_translator(target)
    }
}

impl Default for BackendTranslatorManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_translator_factory() {
        // Test CPU translator creation
        let cpu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::Cpu);
        assert!(cpu_translator.is_ok());
        
        // Test GPU translator creation
        let gpu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::Gpu);
        assert!(gpu_translator.is_ok());
        
        // Test NPU translator creation
        let npu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::NPU);
        assert!(npu_translator.is_ok());
        
        // Test TPU translator creation
        let tpu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::Custom("TPU".to_string()));
        assert!(tpu_translator.is_ok());
        
        // Test unsupported hardware
        let unsupported = BackendTranslatorFactory::create_translator(HardwareTarget::Custom("UNSUPPORTED".to_string()));
        assert!(unsupported.is_err());
    }

    #[test]
    fn test_backend_translator_manager() {
        let manager = BackendTranslatorManager::new();
        
        // Should support multiple hardware targets
        let supported_hardware = manager.list_supported_hardware();
        assert!(supported_hardware.contains(&HardwareTarget::Cpu));
        assert!(supported_hardware.contains(&HardwareTarget::Gpu));
        assert!(supported_hardware.contains(&HardwareTarget::NPU));
        assert!(supported_hardware.contains(&HardwareTarget::Custom("TPU".to_string())));
    }

    #[test]
    fn test_cpu_translator_creation() {
        let translator = BackendTranslatorFactory::create_cpu_translator(
            CpuArchitecture::X86_64,
            vec![SimdCapability::AVX2, SimdCapability::SSE4]
        );
        
        assert_eq!(translator.name(), "cpu_backend_translator_X86_64");
        assert!(translator.supports_native_saic());
    }

    #[test]
    fn test_gpu_translator_creation() {
        let translator = BackendTranslatorFactory::create_gpu_translator(GpuType::Nvidia);
        
        assert_eq!(translator.name(), "gpu_backend_translator_Nvidia");
        assert!(!translator.supports_native_saic());
    }

    #[test]
    fn test_npu_translator_creation() {
        let translator = BackendTranslatorFactory::create_npu_translator(NpuType::Apple);
        
        assert_eq!(translator.name(), "npu_backend_translator_Apple");
        assert!(!translator.supports_native_saic());
    }

    #[test]
    fn test_tpu_translator_creation() {
        let translator = BackendTranslatorFactory::create_tpu_translator(TpuType::Google);
        
        assert_eq!(translator.name(), "tpu_backend_translator_Google");
        assert!(!translator.supports_native_saic());
    }
}
