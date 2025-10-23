//! Template System for COELANOX
//! 
//! This module provides hardware-specific templates for code generation.
//! Each hardware type has its own template implementation.

pub mod cpu_templates;
pub mod gpu_templates;
pub mod npu_templates;
pub mod tpu_templates;

use crate::ir::{OpType, Operation};
use coelanox_core::format::HardwareTarget;
use coelanox_core::{CoelanoxResult, CoelanoxError};

// Re-export template types
pub use cpu_templates::{CpuTemplates, CpuTemplate, SimdCapability};
pub use gpu_templates::{GpuTemplates, GpuTemplate};
pub use npu_templates::{NpuTemplates, NpuTemplate};
pub use tpu_templates::{TpuTemplates, TpuTemplate};

/// Universal template trait for all hardware types
pub trait HardwareTemplate {
    /// Get the operation type this template handles
    fn operation(&self) -> OpType;
    
    /// Get template parameters
    fn parameters(&self) -> Vec<String>;
    
    /// Get supported hardware targets
    fn hardware_targets(&self) -> Vec<HardwareTarget>;
    
    /// Generate code for the operation
    fn generate_code(&self, operation: &Operation) -> CoelanoxResult<String>;
}

/// Template registry for managing all hardware templates
pub struct TemplateRegistry {
    cpu_templates: CpuTemplates,
    gpu_templates: GpuTemplates,
    npu_templates: NpuTemplates,
    tpu_templates: TpuTemplates,
}

impl TemplateRegistry {
    /// Create a new template registry
    pub fn new() -> Self {
        Self {
            cpu_templates: CpuTemplates::new(),
            gpu_templates: GpuTemplates::new(),
            npu_templates: NpuTemplates::new(),
            tpu_templates: TpuTemplates::new(),
        }
    }
    
    /// Get templates for specific hardware target
    pub fn get_templates(&self, target: &HardwareTarget) -> Option<&dyn HardwareTemplate> {
        match target {
            HardwareTarget::Cpu => Some(&self.cpu_templates),
            HardwareTarget::Gpu => Some(&self.gpu_templates),
            HardwareTarget::NPU => Some(&self.npu_templates),
            HardwareTarget::Custom(ref name) if name == "TPU" => Some(&self.tpu_templates),
            _ => None,
        }
    }
    
    /// Generate code for operation on specific hardware
    pub fn generate_code(&self, operation: &Operation, target: HardwareTarget) -> CoelanoxResult<String> {
        match target {
            HardwareTarget::Cpu => {
                let available_simd = vec![SimdCapability::AVX2, SimdCapability::SSE4];
                self.cpu_templates.generate_cpu_code(operation, &available_simd)
                    .map(|_| "CPU machine code generated".to_string())
            }
            HardwareTarget::Gpu => {
                self.gpu_templates.generate_gpu_code(operation)
            }
            HardwareTarget::NPU => {
                self.npu_templates.generate_npu_code(operation)
            }
            HardwareTarget::Custom(ref name) if name == "TPU" => {
                self.tpu_templates.generate_tpu_code(operation)
            }
            _ => Err(CoelanoxError::UnsupportedHardwareTarget(target)),
        }
    }
    
    /// List supported hardware targets
    pub fn supported_hardware(&self) -> Vec<HardwareTarget> {
        vec![
            HardwareTarget::Cpu,
            HardwareTarget::Gpu,
            HardwareTarget::NPU,
            HardwareTarget::Custom("TPU".to_string()),
        ]
    }
    
    /// List supported operations for hardware target
    pub fn supported_operations(&self, target: &HardwareTarget) -> Vec<OpType> {
        match target {
            HardwareTarget::Cpu => self.cpu_templates.supported_operations(),
            HardwareTarget::Gpu => self.gpu_templates.supported_operations(),
            HardwareTarget::NPU => self.npu_templates.supported_operations(),
            HardwareTarget::Custom(ref name) if name == "TPU" => self.tpu_templates.supported_operations(),
            _ => vec![],
        }
    }
    
    /// Check if hardware target is supported
    pub fn is_hardware_supported(&self, target: &HardwareTarget) -> bool {
        self.supported_hardware().contains(target)
    }
    
    /// Check if operation is supported on hardware target
    pub fn is_operation_supported(&self, operation: &OpType, target: &HardwareTarget) -> bool {
        self.supported_operations(target).contains(operation)
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Attribute;

    #[test]
    fn test_template_registry_creation() {
        let registry = TemplateRegistry::new();
        
        // Should support multiple hardware targets
        let supported_hardware = registry.supported_hardware();
        assert!(supported_hardware.contains(&HardwareTarget::Cpu));
        assert!(supported_hardware.contains(&HardwareTarget::Gpu));
        assert!(supported_hardware.contains(&HardwareTarget::NPU));
        assert!(supported_hardware.contains(&HardwareTarget::Custom("TPU".to_string())));
    }

    #[test]
    fn test_cpu_code_generation() {
        let registry = TemplateRegistry::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let result = registry.generate_code(&operation, HardwareTarget::Cpu);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_code_generation() {
        let registry = TemplateRegistry::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let result = registry.generate_code(&operation, HardwareTarget::Gpu);
        assert!(result.is_ok());
        
        let code = result.unwrap();
        assert!(code.contains("gpu_add"));
    }

    #[test]
    fn test_npu_code_generation() {
        let registry = TemplateRegistry::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let result = registry.generate_code(&operation, HardwareTarget::NPU);
        assert!(result.is_ok());
        
        let code = result.unwrap();
        assert!(code.contains("npu_add"));
    }

    #[test]
    fn test_tpu_code_generation() {
        let registry = TemplateRegistry::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let result = registry.generate_code(&operation, HardwareTarget::Custom("TPU".to_string()));
        assert!(result.is_ok());
        
        let code = result.unwrap();
        assert!(code.contains("tpu_add"));
    }

    #[test]
    fn test_unsupported_hardware() {
        let registry = TemplateRegistry::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let result = registry.generate_code(&operation, HardwareTarget::Custom("UNSUPPORTED".to_string()));
        assert!(result.is_err());
    }
}
