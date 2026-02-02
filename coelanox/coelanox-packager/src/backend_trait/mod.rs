//! Backend Translators for COELANOX
//! 
//! This module provides the BackendTranslator trait interface.
//! All concrete backend translator implementations have been removed - vendors will implement their own.

pub mod coelanox_operations;
#[path = "trait.rs"]
pub mod backend_translator;

// Re-export main types
pub use coelanox_operations::{
    CoelanoxAiOperationParams, CoelanoxOptimizationHint, CoelanoxHardwareType,
    CoelanoxAiOperationMetadata, CoelanoxPerformanceProfile, CoelanoxMemoryRequirements,
    get_optimization_hints_for_op_type
};
pub use backend_translator::{
    BackendTranslator, CoelanoxPerformanceEstimate, BackendTranslatorCapabilities,
    CoelanoxHardwareFeature, CoelanoxDataType, BackendTranslatorRegistry
};

use coelanox_core::format::HardwareTarget;
use coelanox_core::{CoelanoxResult, CoelanoxError};

/// Backend translator factory
/// 
/// Creates backend translators for different hardware types
pub struct BackendTranslatorFactory;

impl BackendTranslatorFactory {
    /// Create a backend translator for the specified hardware target
    /// 
    /// All concrete implementations have been removed - vendors must provide their own
    /// backend translator implementations via dynamic library loading.
    pub fn create_translator(_target: HardwareTarget) -> CoelanoxResult<Box<dyn BackendTranslator>> {
        // No default implementations - vendors must provide their own
        Err(CoelanoxError::UnsupportedHardwareTarget(HardwareTarget::Cpu))
    }
    
    /// Get all supported hardware targets
    /// 
    /// Returns empty list - vendors must register their own translators
    pub fn supported_hardware_targets() -> Vec<HardwareTarget> {
        vec![]
    }
    
    /// Check if hardware target is supported
    pub fn is_hardware_supported(_target: &HardwareTarget) -> bool {
        false
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
    /// 
    /// No default translators are registered - vendors must provide their own
    /// implementations via dynamic library loading and register them.
    pub fn new() -> Self {
        let registry = BackendTranslatorRegistry::new();
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
    pub fn translate_operation(&self, operation: &coelanox_core::ir::OpType, params: &CoelanoxAiOperationParams, target: HardwareTarget) -> CoelanoxResult<Vec<u8>> {
        self.registry.translate_operation(operation, params, target)
    }
    
    /// Get performance estimate for operation
    pub fn get_performance_estimate(&self, operation: &coelanox_core::ir::OpType, params: &CoelanoxAiOperationParams, target: HardwareTarget) -> CoelanoxResult<CoelanoxPerformanceEstimate> {
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
    pub fn is_operation_supported(&self, operation: &coelanox_core::ir::OpType, target: &HardwareTarget) -> bool {
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
        // All concrete implementations removed - factory returns error
        let cpu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::Cpu);
        assert!(cpu_translator.is_err());
        
        let unsupported = BackendTranslatorFactory::create_translator(HardwareTarget::Custom("UNSUPPORTED".to_string()));
        assert!(unsupported.is_err());
    }

    #[test]
    fn test_backend_translator_manager() {
        let manager = BackendTranslatorManager::new();
        
        // No default translators registered
        let supported_hardware = manager.list_supported_hardware();
        assert!(supported_hardware.is_empty());
    }
}
