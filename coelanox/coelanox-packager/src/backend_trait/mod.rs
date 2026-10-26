//! Backend translator trait for hardware-specific code generation.
//!
//! Defines the interface vendors implement to translate IR to machine code.
//! Packager uses in-tree translators (ONNX→IR) and CLF for code; no in-tree
//! BackendTranslator implementations (GPU/NPU/etc. are vendor-provided).

#[path = "trait.rs"]
pub mod backend_translator;
pub mod coelanox_operations;

// Re-export main types
pub use backend_translator::{
    BackendTranslator, BackendTranslatorCapabilities, BackendTranslatorRegistry, CoelanoxDataType,
    CoelanoxHardwareFeature, CoelanoxPerformanceEstimate,
};
pub use coelanox_operations::{
    get_optimization_hints_for_op_type, CoelanoxAiOperationMetadata, CoelanoxAiOperationParams,
    CoelanoxHardwareType, CoelanoxMemoryRequirements, CoelanoxOptimizationHint,
    CoelanoxPerformanceProfile,
};

use coelanox_core::format::HardwareTarget;
use coelanox_core::{CoelanoxError, CoelanoxResult};

/// Backend translator factory
///
/// Creates backend translators for different hardware types
pub struct BackendTranslatorFactory;

impl BackendTranslatorFactory {
    /// Create a backend translator for the specified hardware target.
    /// Returns error; no in-tree implementations (vendors register via registry).
    pub fn create_translator(
        _target: HardwareTarget,
    ) -> CoelanoxResult<Box<dyn BackendTranslator>> {
        Err(CoelanoxError::UnsupportedHardwareTarget(
            HardwareTarget::Cpu,
        ))
    }

    /// Supported hardware targets (empty; vendors register).
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
    /// Create empty manager; vendors register translators via `register_custom_translator`.
    pub fn new() -> Self {
        let registry = BackendTranslatorRegistry::new();
        Self { registry }
    }

    /// Get the backend translator registry
    pub fn registry(&self) -> &BackendTranslatorRegistry {
        &self.registry
    }

    /// Mutable access to the registry
    pub fn registry_mut(&mut self) -> &mut BackendTranslatorRegistry {
        &mut self.registry
    }

    /// Translate operation to hardware-specific code
    pub fn translate_operation(
        &self,
        operation: &coelanox_core::ir::OpType,
        params: &CoelanoxAiOperationParams,
        target: HardwareTarget,
    ) -> CoelanoxResult<Vec<u8>> {
        self.registry.translate_operation(operation, params, target)
    }

    /// Get performance estimate for operation
    pub fn get_performance_estimate(
        &self,
        operation: &coelanox_core::ir::OpType,
        params: &CoelanoxAiOperationParams,
        target: HardwareTarget,
    ) -> CoelanoxResult<CoelanoxPerformanceEstimate> {
        self.registry
            .get_performance_estimate(operation, params, target)
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
    pub fn is_operation_supported(
        &self,
        operation: &coelanox_core::ir::OpType,
        target: &HardwareTarget,
    ) -> bool {
        self.registry.is_operation_supported(operation, target)
    }

    /// Get translator capabilities
    pub fn get_capabilities(
        &self,
        target: &HardwareTarget,
    ) -> Option<BackendTranslatorCapabilities> {
        self.registry.get_capabilities(target)
    }

    /// Register a custom backend translator
    pub fn register_custom_translator(
        &mut self,
        target: HardwareTarget,
        translator: Box<dyn BackendTranslator>,
    ) {
        self.registry.register_translator(target, translator);
    }

    /// Unregister a backend translator
    pub fn unregister_translator(
        &mut self,
        target: &HardwareTarget,
    ) -> Option<Box<dyn BackendTranslator>> {
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
        // Factory returns error; no in-tree BackendTranslator
        let cpu_translator = BackendTranslatorFactory::create_translator(HardwareTarget::Cpu);
        assert!(cpu_translator.is_err());

        let unsupported = BackendTranslatorFactory::create_translator(HardwareTarget::Custom(
            "UNSUPPORTED".to_string(),
        ));
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
