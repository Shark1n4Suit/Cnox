//! Assembly Template System for SAIC
//! 
//! This module implements template-based code generation that allows SAIC to generate
//! hardware-specific assembly code from universal templates.

use std::collections::HashMap;
use crate::packager::ir::{OpType, Operation};
use crate::format::HardwareTarget;
use crate::{SaicResult, SaicError};

/// Universal assembly template structure
#[derive(Debug, Clone)]
pub struct AssemblyTemplate {
    /// The operation type this template handles
    pub operation: OpType,
    /// Universal assembly template with parameter placeholders
    pub template: String,
    /// List of parameter names used in the template
    pub parameters: Vec<String>,
    /// Hardware targets this template supports
    pub hardware_targets: Vec<HardwareTarget>,
}

impl AssemblyTemplate {
    /// Create a new assembly template
    pub fn new(
        operation: OpType,
        template: String,
        parameters: Vec<String>,
        hardware_targets: Vec<HardwareTarget>,
    ) -> Self {
        Self {
            operation,
            template,
            parameters,
            hardware_targets,
        }
    }

    /// Instantiate the template with given parameters for a specific hardware target
    pub fn instantiate(
        &self,
        params: &HashMap<String, String>,
        target: HardwareTarget,
    ) -> SaicResult<String> {
        // Validate hardware target is supported
        if !self.hardware_targets.contains(&target) {
            return Err(SaicError::UnsupportedHardwareTarget(target));
        }

        // Validate all required parameters are provided
        for param in &self.parameters {
            if !params.contains_key(param) {
                return Err(SaicError::MissingTemplateParameter(param.clone()));
            }
        }

        // Start with the template
        let mut assembly = self.template.clone();

        // Substitute parameters
        for (param, value) in params {
            let placeholder = format!("{{{{{}}}}}", param);
            assembly = assembly.replace(&placeholder, value);
        }

        // Translate to hardware-specific assembly
        let hardware_assembly = self.translate_for_hardware(&assembly, target)?;

        Ok(hardware_assembly)
    }

    /// Translate universal assembly to hardware-specific assembly
    fn translate_for_hardware(
        &self,
        assembly: &str,
        target: HardwareTarget,
    ) -> SaicResult<String> {
        match target {
            HardwareTarget::Cpu => self.translate_to_cpu(assembly),
            HardwareTarget::Gpu => self.translate_to_gpu(assembly),
            HardwareTarget::NeuralEngine => self.translate_to_neural_engine(assembly),
            HardwareTarget::NPU => self.translate_to_npu(assembly),
            _ => Err(SaicError::UnsupportedHardwareTarget(target)),
        }
    }

    /// Translate to x86-64 CPU assembly
    fn translate_to_cpu(&self, assembly: &str) -> SaicResult<String> {
        let mut cpu_assembly = assembly.to_string();

        // Translate universal instructions to x86-64
        cpu_assembly = cpu_assembly
            .replace("load", "mov")
            .replace("add", "add")
            .replace("multiply", "mul")
            .replace("store", "mov")
            .replace("{{REG_A}}", "rax")
            .replace("{{REG_B}}", "rcx")
            .replace("{{RESULT_REG}}", "rdx")
            .replace("{{INPUT_REG}}", "rsi")
            .replace("{{KERNEL_REG}}", "rdi")
            .replace("{{OUTPUT_REG}}", "r8");

        Ok(cpu_assembly)
    }

    /// Translate to CUDA GPU assembly
    fn translate_to_gpu(&self, assembly: &str) -> SaicResult<String> {
        let mut gpu_assembly = assembly.to_string();

        // Translate universal instructions to CUDA
        gpu_assembly = gpu_assembly
            .replace("load", "ld.global")
            .replace("add", "add")
            .replace("multiply", "mul")
            .replace("store", "st.global")
            .replace("{{REG_A}}", "r0")
            .replace("{{REG_B}}", "r1")
            .replace("{{RESULT_REG}}", "r2")
            .replace("{{INPUT_REG}}", "r3")
            .replace("{{KERNEL_REG}}", "r4")
            .replace("{{OUTPUT_REG}}", "r5");

        Ok(gpu_assembly)
    }

    /// Translate to Apple Neural Engine assembly
    fn translate_to_neural_engine(&self, assembly: &str) -> SaicResult<String> {
        let mut ne_assembly = assembly.to_string();

        // Translate universal instructions to Neural Engine
        ne_assembly = ne_assembly
            .replace("load", "ne_load")
            .replace("add", "ne_add")
            .replace("multiply", "ne_mul")
            .replace("store", "ne_store")
            .replace("{{REG_A}}", "ne_reg0")
            .replace("{{REG_B}}", "ne_reg1")
            .replace("{{RESULT_REG}}", "ne_reg2")
            .replace("{{INPUT_REG}}", "ne_reg3")
            .replace("{{KERNEL_REG}}", "ne_reg4")
            .replace("{{OUTPUT_REG}}", "ne_reg5");

        Ok(ne_assembly)
    }

    /// Translate to NPU assembly
    fn translate_to_npu(&self, assembly: &str) -> SaicResult<String> {
        let mut npu_assembly = assembly.to_string();

        // Translate universal instructions to NPU
        npu_assembly = npu_assembly
            .replace("load", "npu_load")
            .replace("add", "npu_add")
            .replace("multiply", "npu_mul")
            .replace("store", "npu_store")
            .replace("{{REG_A}}", "npu_reg0")
            .replace("{{REG_B}}", "npu_reg1")
            .replace("{{RESULT_REG}}", "npu_reg2")
            .replace("{{INPUT_REG}}", "npu_reg3")
            .replace("{{KERNEL_REG}}", "npu_reg4")
            .replace("{{OUTPUT_REG}}", "npu_reg5");

        Ok(npu_assembly)
    }
}

/// Collection of universal assembly templates
#[derive(Debug)]
pub struct UniversalTemplates {
    pub add_template: AssemblyTemplate,
    pub mul_template: AssemblyTemplate,
    pub conv_template: AssemblyTemplate,
    pub matmul_template: AssemblyTemplate,
}

impl UniversalTemplates {
    /// Create a new collection of universal templates
    pub fn new() -> Self {
        Self {
            add_template: AssemblyTemplate::new(
                OpType::Add,
                r#"
                    // Universal Add operation
                    load {{INPUT_A_PTR}}, {{REG_A}}
                    load {{INPUT_B_PTR}}, {{REG_B}}
                    add {{REG_A}}, {{REG_B}}, {{RESULT_REG}}
                    store {{RESULT_REG}}, {{OUTPUT_PTR}}
                "#.to_string(),
                vec![
                    "INPUT_A_PTR".to_string(),
                    "INPUT_B_PTR".to_string(),
                    "OUTPUT_PTR".to_string(),
                    "REG_A".to_string(),
                    "REG_B".to_string(),
                    "RESULT_REG".to_string(),
                ],
                vec![HardwareTarget::Cpu, HardwareTarget::Gpu, HardwareTarget::NeuralEngine, HardwareTarget::NPU],
            ),
            mul_template: AssemblyTemplate::new(
                OpType::Multiply,
                r#"
                    // Universal Multiply operation
                    load {{INPUT_A_PTR}}, {{REG_A}}
                    load {{INPUT_B_PTR}}, {{REG_B}}
                    multiply {{REG_A}}, {{REG_B}}, {{RESULT_REG}}
                    store {{RESULT_REG}}, {{OUTPUT_PTR}}
                "#.to_string(),
                vec![
                    "INPUT_A_PTR".to_string(),
                    "INPUT_B_PTR".to_string(),
                    "OUTPUT_PTR".to_string(),
                    "REG_A".to_string(),
                    "REG_B".to_string(),
                    "RESULT_REG".to_string(),
                ],
                vec![HardwareTarget::Cpu, HardwareTarget::Gpu, HardwareTarget::NeuralEngine, HardwareTarget::NPU],
            ),
            conv_template: AssemblyTemplate::new(
                OpType::Convolution,
                r#"
                    // Universal Convolution operation
                    load {{INPUT_PTR}}, {{INPUT_REG}}
                    load {{KERNEL_PTR}}, {{KERNEL_REG}}
                    convolution {{INPUT_REG}}, {{KERNEL_REG}}, {{OUTPUT_REG}}, {{KERNEL_SIZE}}, {{STRIDE}}, {{PADDING}}
                    store {{OUTPUT_REG}}, {{OUTPUT_PTR}}
                "#.to_string(),
                vec![
                    "INPUT_PTR".to_string(),
                    "KERNEL_PTR".to_string(),
                    "OUTPUT_PTR".to_string(),
                    "INPUT_REG".to_string(),
                    "KERNEL_REG".to_string(),
                    "OUTPUT_REG".to_string(),
                    "KERNEL_SIZE".to_string(),
                    "STRIDE".to_string(),
                    "PADDING".to_string(),
                ],
                vec![HardwareTarget::Cpu, HardwareTarget::Gpu, HardwareTarget::NeuralEngine, HardwareTarget::NPU],
            ),
            matmul_template: AssemblyTemplate::new(
                OpType::MatMul,
                r#"
                    // Universal Matrix Multiplication operation
                    load {{MATRIX_A_PTR}}, {{MATRIX_A_REG}}
                    load {{MATRIX_B_PTR}}, {{MATRIX_B_REG}}
                    matmul {{MATRIX_A_REG}}, {{MATRIX_B_REG}}, {{RESULT_REG}}, {{M}}, {{N}}, {{K}}
                    store {{RESULT_REG}}, {{OUTPUT_PTR}}
                "#.to_string(),
                vec![
                    "MATRIX_A_PTR".to_string(),
                    "MATRIX_B_PTR".to_string(),
                    "OUTPUT_PTR".to_string(),
                    "MATRIX_A_REG".to_string(),
                    "MATRIX_B_REG".to_string(),
                    "RESULT_REG".to_string(),
                    "M".to_string(),
                    "N".to_string(),
                    "K".to_string(),
                ],
                vec![HardwareTarget::Cpu, HardwareTarget::Gpu, HardwareTarget::NeuralEngine, HardwareTarget::NPU],
            ),
        }
    }

    /// Get template for a specific operation type
    pub fn get_template(&self, operation: OpType) -> Option<&AssemblyTemplate> {
        match operation {
            OpType::Add => Some(&self.add_template),
            OpType::Multiply => Some(&self.mul_template),
            OpType::Convolution => Some(&self.conv_template),
            OpType::MatMul => Some(&self.matmul_template),
            _ => None,
        }
    }

    /// Generate assembly for an operation
    pub fn generate_assembly(
        &self,
        operation: &Operation,
        target: HardwareTarget,
    ) -> SaicResult<String> {
        let template = self.get_template(operation.op_type)
            .ok_or_else(|| SaicError::UnsupportedOperation(operation.op_type))?;

        // Extract parameters from operation attributes
        let mut params = HashMap::new();
        
        // Add common parameters
        params.insert("INPUT_A_PTR".to_string(), "[rbp+8]".to_string());
        params.insert("INPUT_B_PTR".to_string(), "[rbp+16]".to_string());
        params.insert("OUTPUT_PTR".to_string(), "[rbp+24]".to_string());

        // Add operation-specific parameters
        match operation.op_type {
            OpType::Convolution => {
                let kernel_size = operation.attributes.get("kernel_shape")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "3".to_string());
                params.insert("KERNEL_SIZE".to_string(), kernel_size);
                
                let stride = operation.attributes.get("strides")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "1".to_string());
                params.insert("STRIDE".to_string(), stride);
                
                let padding = operation.attributes.get("pads")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "0".to_string());
                params.insert("PADDING".to_string(), padding);
            }
            OpType::MatMul => {
                let m = operation.attributes.get("M")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "1".to_string());
                params.insert("M".to_string(), m);
                
                let n = operation.attributes.get("N")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "1".to_string());
                params.insert("N".to_string(), n);
                
                let k = operation.attributes.get("K")
                    .map(|attr| format!("{:?}", attr))
                    .unwrap_or_else(|| "1".to_string());
                params.insert("K".to_string(), k);
            }
            _ => {}
        }

        // Instantiate template
        template.instantiate(&params, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packager::ir::Attribute;

    #[test]
    fn test_add_template_instantiation() {
        let templates = UniversalTemplates::new();
        let mut operation = Operation::new(OpType::Add);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let assembly = templates.generate_assembly(&operation, HardwareTarget::CPU)
            .expect("Failed to generate assembly");

        // Should contain CPU-specific instructions
        assert!(assembly.contains("mov"));
        assert!(assembly.contains("add"));
        assert!(assembly.contains("rax"));
        assert!(assembly.contains("rcx"));
    }

    #[test]
    fn test_gpu_template_instantiation() {
        let templates = UniversalTemplates::new();
        let mut operation = Operation::new(OpType::Mul);
        operation.attributes.insert("test".to_string(), Attribute::Int(42));

        let assembly = templates.generate_assembly(&operation, HardwareTarget::GPU)
            .expect("Failed to generate assembly");

        // Should contain GPU-specific instructions
        assert!(assembly.contains("ld.global"));
        assert!(assembly.contains("mul"));
        assert!(assembly.contains("st.global"));
        assert!(assembly.contains("r0"));
        assert!(assembly.contains("r1"));
    }

    #[test]
    fn test_convolution_template_with_parameters() {
        let templates = UniversalTemplates::new();
        let mut operation = Operation::new(OpType::Convolution);
        operation.attributes.insert("kernel_shape".to_string(), Attribute::Ints(vec![3, 3]));
        operation.attributes.insert("strides".to_string(), Attribute::Ints(vec![1, 1]));
        operation.attributes.insert("pads".to_string(), Attribute::Ints(vec![0, 0, 0, 0]));

        let assembly = templates.generate_assembly(&operation, HardwareTarget::CPU)
            .expect("Failed to generate assembly");

        // Should contain convolution-specific assembly
        assert!(assembly.contains("convolution"));
        assert!(assembly.contains("3"));
        assert!(assembly.contains("1"));
        assert!(assembly.contains("0"));
    }
}
