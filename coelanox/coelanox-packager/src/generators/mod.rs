//! Code generators for optimized machine code//! 
//! This module provides hardware-specific code generators that convert optimized IR
//! into machine code for different hardware targets.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::HardwareTarget;
use coelanox_core::ir::{OptimizedIR, Operation, OpType, Attribute};
use tracing::{warn, info, trace};

// Include backend translators
pub mod backend_translators;
// Include CPU code generator
pub mod cpu_codegen;

// Include backend translators
use backend_translators::{BackendTranslatorManager, CoelanoxAiOperation, CoelanoxAiOperationParams};
use backend_translators::coelanox_operations::CoelanoxAttribute;

/// Backend translator-based code generator
pub struct BackendTranslatorCodeGenerator {
    /// Generator name
    name: String,
    /// Generator version
    version: String,
    /// Backend translator manager
    translator_manager: BackendTranslatorManager,
    /// Target hardware
    target: HardwareTarget,
}

impl BackendTranslatorCodeGenerator {
    /// Create a new backend translator-based code generator
    pub fn new(target: HardwareTarget) -> Self {
        Self {
            name: format!("backend_translator_generator_{:?}", target),
            version: "1.0.0".to_string(),
            translator_manager: BackendTranslatorManager::new(),
            target,
        }
    }
    
    /// Defuse fused operations before passing to vendor translators (removes vendor pressure!)
    fn defuse_operations(&self, operations: &[Operation]) -> Vec<Operation> {
        let mut defused_ops = Vec::new();
        
        for operation in operations {
            match &operation.op_type {
                OpType::Custom(ref name) => {
                    // Defuse fused ops so vendors only see primitives
                    match name.as_str() {
                        "ConvReLU" => {
                            defused_ops.push(Operation::new(OpType::Convolution));
                            defused_ops.push(Operation::new(OpType::Activation));
                        }
                        "MatMulAdd" => {
                            defused_ops.push(Operation::new(OpType::MatMul));
                            defused_ops.push(Operation::new(OpType::Add));
                        }
                        "EdgeConvReLU" => {
                            defused_ops.push(Operation::new(OpType::Convolution));
                            defused_ops.push(Operation::new(OpType::Activation));
                        }
                        _ => defused_ops.push(operation.clone()), // Vendor experimental primitives pass through
                    }
                }
                _ => defused_ops.push(operation.clone()), // Primitive ops pass through
            }
        }
        
        defused_ops
    }
    
    /// Generate code using backend translators
    fn generate_backend_translator_code(&self, operations: &[Operation]) -> CoelanoxResult<Vec<u8>> {
        let mut machine_code = Vec::new();
        
        // Generate function prologue
        machine_code.extend_from_slice(&self.generate_prologue());
        
        // Defuse fused ops BEFORE sending to vendors (vendors only see primitives!)
        let defused_ops = self.defuse_operations(operations);
        
        // Generate code for each operation using backend translators
        for operation in &defused_ops {
            let coelanox_operation = self.convert_to_coelanox_operation(operation)?;
            let coelanox_params = self.convert_to_coelanox_params(operation)?;
            
            let operation_code = self.translator_manager.translate_operation(
                &coelanox_operation,
                &coelanox_params,
                self.target.clone()
            )?;
            
            // Append binary machine code directly (BackendTranslator returns Vec<u8>)
            machine_code.extend_from_slice(&operation_code);
        }
        
        // Generate function epilogue
        machine_code.extend_from_slice(&self.generate_epilogue());
        
        Ok(machine_code)
    }
    
    /// Convert Operation to CoelanoxAiOperation
    fn convert_to_coelanox_operation(&self, operation: &Operation) -> CoelanoxResult<CoelanoxAiOperation> {
        match operation.op_type {
            // Basic arithmetic
            OpType::Add => Ok(CoelanoxAiOperation::Add),
            OpType::Subtract => Ok(CoelanoxAiOperation::Subtract),
            OpType::Multiply => Ok(CoelanoxAiOperation::Multiply),
            OpType::Divide => Ok(CoelanoxAiOperation::Divide),
            
            // Neural network operations
            OpType::Convolution => Ok(CoelanoxAiOperation::Convolution),
            OpType::Pooling | OpType::MaxPool | OpType::AvgPool | OpType::GlobalMaxPool | OpType::GlobalAvgPool => Ok(CoelanoxAiOperation::Pooling),
            OpType::Activation | OpType::Relu | OpType::Sigmoid | OpType::Tanh | OpType::Softmax | OpType::LogSoftmax | OpType::Gelu | OpType::Swish => Ok(CoelanoxAiOperation::Activation),
            OpType::BatchNorm => Ok(CoelanoxAiOperation::BatchNorm),
            OpType::Dropout => Ok(CoelanoxAiOperation::Dropout),
            
            // Matrix operations
            OpType::MatMul | OpType::Gemm => Ok(CoelanoxAiOperation::MatMul),
            OpType::Transpose => Ok(CoelanoxAiOperation::Transpose),
            OpType::Reshape => Ok(CoelanoxAiOperation::Reshape),
            
            // Attention operations
            OpType::Attention => Ok(CoelanoxAiOperation::Attention),
            OpType::MultiHeadAttention => Ok(CoelanoxAiOperation::MultiHeadAttention),
            OpType::SelfAttention => Ok(CoelanoxAiOperation::SelfAttention),
            
            // Transformer operations
            OpType::Transformer => Ok(CoelanoxAiOperation::Transformer),
            OpType::Encoder => Ok(CoelanoxAiOperation::Encoder),
            OpType::Decoder => Ok(CoelanoxAiOperation::Decoder),
            
            // Custom operations (pass through)
            OpType::Custom(ref name) => Ok(CoelanoxAiOperation::Custom(name.clone())),
            
            // Operations not directly supported by CoelanoxAiOperation enum
            // These should be decomposed or handled via Custom
            OpType::LayerNorm | OpType::Gather | OpType::Scatter | OpType::Slice | OpType::Split | OpType::Concatenate | OpType::Permute |
            OpType::ReduceSum | OpType::ReduceMean | OpType::ReduceMax | OpType::ReduceMin | OpType::ReduceProd |
            OpType::Broadcast | OpType::Expand |
            OpType::Equal | OpType::NotEqual | OpType::Greater | OpType::GreaterEqual | OpType::Less | OpType::LessEqual |
            OpType::And | OpType::Or | OpType::Not |
            OpType::Sqrt | OpType::Pow | OpType::Cos | OpType::Sin | OpType::Exp | OpType::Log => {
                // Map to Custom with operation name for vendor handling
                Ok(CoelanoxAiOperation::Custom(format!("{:?}", operation.op_type)))
            },
        }
    }
    
    /// Convert Operation to CoelanoxAiOperationParams
    fn convert_to_coelanox_params(&self, operation: &Operation) -> CoelanoxResult<CoelanoxAiOperationParams> {
        let mut input_shapes = Vec::new();
        let mut output_shapes = Vec::new();
        let mut input_is_contiguous = Vec::new();
        let mut output_is_contiguous = Vec::new();
        let mut attributes = std::collections::HashMap::new();
        let mut optimization_hints = Vec::new();
        
        // Determine if operation creates non-contiguous output
        let creates_non_contiguous = matches!(
            operation.op_type,
            OpType::Slice | OpType::Transpose | OpType::Permute
        );
        
        // Convert input shapes (placeholder - Operation doesn't have shape info)
        for _input in &operation.inputs {
            input_shapes.push(vec![1, 1]); // Placeholder shape
            // Assume inputs are contiguous unless we track producer operations
            // For now, default to true (vendors can check if needed)
            input_is_contiguous.push(true);
        }
        
        // Convert output shapes (placeholder - Operation doesn't have shape info)
        for _output in &operation.outputs {
            output_shapes.push(vec![1, 1]); // Placeholder shape
            // Set contiguity based on operation type
            output_is_contiguous.push(!creates_non_contiguous);
        }
        
        // Convert attributes
        for (key, value) in &operation.attributes {
            match value {
                Attribute::Int(i) => {
                    attributes.insert(key.clone(), CoelanoxAttribute::Int(*i as i32));
                }
                Attribute::Float(f) => {
                    attributes.insert(key.clone(), CoelanoxAttribute::Float(*f));
                }
                Attribute::String(s) => {
                    attributes.insert(key.clone(), CoelanoxAttribute::String(s.clone()));
                }
                Attribute::Bool(b) => {
                    attributes.insert(key.clone(), CoelanoxAttribute::Bool(*b));
                }
                Attribute::FloatArray(_) | Attribute::IntArray(_) | Attribute::StringArray(_) => {
                    // Skip array attributes for now
                    continue;
                }
            }
        }
        
        // Get optimization hints from the operation
        let coelanox_operation = self.convert_to_coelanox_operation(operation)?;
        optimization_hints = coelanox_operation.optimization_hints();
        
        Ok(CoelanoxAiOperationParams {
            input_shapes,
            output_shapes,
            input_is_contiguous,
            output_is_contiguous,
            attributes,
            optimization_hints,
        })
    }
    
    /// Generate function prologue
    fn generate_prologue(&self) -> Vec<u8> {
        vec![
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x20,
            0x53, 0x56, 0x57,
        ]
    }
    
    /// Generate function epilogue
    fn generate_epilogue(&self) -> Vec<u8> {
        vec![
            0x5F, 0x5E, 0x5B, 0x48, 0x83, 0xC4, 0x20, 0x5D, 0xC3,
        ]
    }
}

/// Code generator trait
pub trait CodeGenerator: Send + Sync {
    /// Generate machine code from optimized IR
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>>;
    
    /// Generate machine code with input shapes information
    /// This allows generating function signatures that support multiple inputs
    /// for models like Transformers (BERT, GPT) which need multiple inputs
    fn generate_with_input_shapes(&self, optimized_ir: &OptimizedIR, input_shapes: &[coelanox_core::ir::Shape]) -> CoelanoxResult<Vec<u8>> {
        // Default implementation just calls generate() for backward compatibility
        self.generate(optimized_ir)
    }
    
    /// Get target hardware
    fn target_hardware(&self) -> HardwareTarget;
    
    /// Get generator name
    fn name(&self) -> &str;
    
    /// Get generator version
    fn version(&self) -> &str;
}


/// GPU code generator
pub struct GpuCodeGenerator {
    /// Generator name
    name: String,
    /// Generator version
    version: String,
}

impl GpuCodeGenerator {
    /// Create a new GPU code generator
    pub fn new() -> Self {
        Self {
            name: "gpu_generator".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Generate GPU machine code (CUDA PTX)
    fn generate_gpu_code(&self, operations: &[Operation]) -> CoelanoxResult<Vec<u8>> {
        let mut ptx_code = String::new();
        
        // PTX header
        ptx_code.push_str(".version 7.0\n");
        ptx_code.push_str(".target sm_70\n");
        ptx_code.push_str(".address_size 64\n\n");
        
        // Generate PTX for each operation
        for operation in operations {
            match operation.op_type {
                OpType::Add => {
                    ptx_code.push_str(&self.generate_add_ptx(operation));
                }
                OpType::Multiply => {
                    ptx_code.push_str(&self.generate_multiply_ptx(operation));
                }
                OpType::Convolution => {
                    ptx_code.push_str(&self.generate_convolution_ptx(operation));
                }
                OpType::Pooling => {
                    ptx_code.push_str(&self.generate_pooling_ptx(operation));
                }
                _ => {
                    ptx_code.push_str("// Unsupported operation\n");
                }
            }
        }
        
        Ok(ptx_code.into_bytes())
    }
    
    /// Generate PTX for addition
    fn generate_add_ptx(&self, _operation: &Operation) -> String {
        let mut ptx = String::new();
        ptx.push_str(".visible .entry add_kernel(\n");
        ptx.push_str("    .param .u64 input1,\n");
        ptx.push_str("    .param .u64 input2,\n");
        ptx.push_str("    .param .u64 output,\n");
        ptx.push_str("    .param .u32 size\n");
        ptx.push_str(") {\n");
        ptx.push_str("    .reg .u32 %tid;\n");
        ptx.push_str("    .reg .u64 %input1_addr, %input2_addr, %output_addr;\n");
        ptx.push_str("    .reg .f32 %val1, %val2, %result;\n\n");
        ptx.push_str("    mov.u32 %tid, %tid.x;\n");
        ptx.push_str("    ld.param.u64 %input1_addr, [input1];\n");
        ptx.push_str("    ld.param.u64 %input2_addr, [input2];\n");
        ptx.push_str("    ld.param.u64 %output_addr, [output];\n\n");
        ptx.push_str("    ld.global.f32 %val1, [%input1_addr + %tid];\n");
        ptx.push_str("    ld.global.f32 %val2, [%input2_addr + %tid];\n");
        ptx.push_str("    add.f32 %result, %val1, %val2;\n");
        ptx.push_str("    st.global.f32 [%output_addr + %tid], %result;\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n\n");
        ptx
    }
    
    /// Generate PTX for multiplication
    fn generate_multiply_ptx(&self, _operation: &Operation) -> String {
        let mut ptx = String::new();
        ptx.push_str(".visible .entry multiply_kernel(\n");
        ptx.push_str("    .param .u64 input1,\n");
        ptx.push_str("    .param .u64 input2,\n");
        ptx.push_str("    .param .u64 output,\n");
        ptx.push_str("    .param .u32 size\n");
        ptx.push_str(") {\n");
        ptx.push_str("    .reg .u32 %tid;\n");
        ptx.push_str("    .reg .u64 %input1_addr, %input2_addr, %output_addr;\n");
        ptx.push_str("    .reg .f32 %val1, %val2, %result;\n\n");
        ptx.push_str("    mov.u32 %tid, %tid.x;\n");
        ptx.push_str("    ld.param.u64 %input1_addr, [input1];\n");
        ptx.push_str("    ld.param.u64 %input2_addr, [input2];\n");
        ptx.push_str("    ld.param.u64 %output_addr, [output];\n\n");
        ptx.push_str("    ld.global.f32 %val1, [%input1_addr + %tid];\n");
        ptx.push_str("    ld.global.f32 %val2, [%input2_addr + %tid];\n");
        ptx.push_str("    mul.f32 %result, %val1, %val2;\n");
        ptx.push_str("    st.global.f32 [%output_addr + %tid], %result;\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n\n");
        ptx
    }
    
    /// Generate PTX for convolution
    fn generate_convolution_ptx(&self, _operation: &Operation) -> String {
        let mut ptx = String::new();
        ptx.push_str(".visible .entry conv_kernel(\n");
        ptx.push_str("    .param .u64 input,\n");
        ptx.push_str("    .param .u64 weights,\n");
        ptx.push_str("    .param .u64 output,\n");
        ptx.push_str("    .param .u32 batch_size,\n");
        ptx.push_str("    .param .u32 channels,\n");
        ptx.push_str("    .param .u32 height,\n");
        ptx.push_str("    .param .u32 width\n");
        ptx.push_str(") {\n");
        ptx.push_str("    .reg .u32 %tid, %batch, %channel, %h, %w;\n");
        ptx.push_str("    .reg .u64 %input_addr, %weights_addr, %output_addr;\n");
        ptx.push_str("    .reg .f32 %sum, %val, %weight;\n\n");
        ptx.push_str("    mov.u32 %tid, %tid.x;\n");
        ptx.push_str("    ld.param.u64 %input_addr, [input];\n");
        ptx.push_str("    ld.param.u64 %weights_addr, [weights];\n");
        ptx.push_str("    ld.param.u64 %output_addr, [output];\n\n");
        ptx.push_str("    // Simple convolution implementation\n");
        ptx.push_str("    mov.f32 %sum, 0.0;\n");
        ptx.push_str("    ld.global.f32 %val, [%input_addr + %tid];\n");
        ptx.push_str("    ld.global.f32 %weight, [%weights_addr + %tid];\n");
        ptx.push_str("    fma.f32 %sum, %val, %weight, %sum;\n");
        ptx.push_str("    st.global.f32 [%output_addr + %tid], %sum;\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n\n");
        ptx
    }
    
    /// Generate PTX for pooling
    fn generate_pooling_ptx(&self, _operation: &Operation) -> String {
        let mut ptx = String::new();
        ptx.push_str(".visible .entry pool_kernel(\n");
        ptx.push_str("    .param .u64 input,\n");
        ptx.push_str("    .param .u64 output,\n");
        ptx.push_str("    .param .u32 batch_size,\n");
        ptx.push_str("    .param .u32 channels,\n");
        ptx.push_str("    .param .u32 height,\n");
        ptx.push_str("    .param .u32 width\n");
        ptx.push_str(") {\n");
        ptx.push_str("    .reg .u32 %tid;\n");
        ptx.push_str("    .reg .u64 %input_addr, %output_addr;\n");
        ptx.push_str("    .reg .f32 %val, %max_val;\n\n");
        ptx.push_str("    mov.u32 %tid, %tid.x;\n");
        ptx.push_str("    ld.param.u64 %input_addr, [input];\n");
        ptx.push_str("    ld.param.u64 %output_addr, [output];\n\n");
        ptx.push_str("    // Simple max pooling implementation\n");
        ptx.push_str("    ld.global.f32 %val, [%input_addr + %tid];\n");
        ptx.push_str("    max.f32 %max_val, %val, %max_val;\n");
        ptx.push_str("    st.global.f32 [%output_addr + %tid], %max_val;\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n\n");
        ptx
    }
}

impl CodeGenerator for GpuCodeGenerator {
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>> {
        self.generate_gpu_code(&optimized_ir.optimized_nodes.iter().map(|node| {
            let mut op = Operation::new(node.op_type.clone());
            op.attributes = node.attributes.clone();
            op
        }).collect::<Vec<_>>())
    }
    
    fn target_hardware(&self) -> HardwareTarget {
        HardwareTarget::Gpu
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

/// Edge code generator
pub struct EdgeCodeGenerator {
    /// Generator name
    name: String,
    /// Generator version
    version: String,
}

impl EdgeCodeGenerator {
    /// Create a new Edge code generator
    pub fn new() -> Self {
        Self {
            name: "edge_generator".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Generate Edge machine code (ARM assembly)
    fn generate_edge_code(&self, operations: &[Operation]) -> CoelanoxResult<Vec<u8>> {
        let mut arm_code = String::new();
        
        // ARM assembly header
        arm_code.push_str(".text\n");
        arm_code.push_str(".global _start\n");
        arm_code.push_str(".align 2\n\n");
        
        // Generate ARM assembly for each operation
        for operation in operations {
            match operation.op_type {
                OpType::Add => {
                    arm_code.push_str(&self.generate_add_arm(operation));
                }
                OpType::Multiply => {
                    arm_code.push_str(&self.generate_multiply_arm(operation));
                }
                OpType::Convolution => {
                    arm_code.push_str(&self.generate_convolution_arm(operation));
                }
                OpType::Pooling => {
                    arm_code.push_str(&self.generate_pooling_arm(operation));
                }
                _ => {
                    arm_code.push_str("    // Unsupported operation\n");
                }
            }
        }
        
        Ok(arm_code.into_bytes())
    }
    
    /// Generate ARM assembly for addition
    fn generate_add_arm(&self, _operation: &Operation) -> String {
        let mut arm = String::new();
        arm.push_str("add_kernel:\n");
        arm.push_str("    // Load input1 address\n");
        arm.push_str("    ldr x0, [x0]\n");
        arm.push_str("    // Load input2 address\n");
        arm.push_str("    ldr x1, [x1]\n");
        arm.push_str("    // Load output address\n");
        arm.push_str("    ldr x2, [x2]\n");
        arm.push_str("    // Load values\n");
        arm.push_str("    ldr s0, [x0]\n");
        arm.push_str("    ldr s1, [x1]\n");
        arm.push_str("    // Add\n");
        arm.push_str("    fadd s0, s0, s1\n");
        arm.push_str("    // Store result\n");
        arm.push_str("    str s0, [x2]\n");
        arm.push_str("    ret\n\n");
        arm
    }
    
    /// Generate ARM assembly for multiplication
    fn generate_multiply_arm(&self, _operation: &Operation) -> String {
        let mut arm = String::new();
        arm.push_str("multiply_kernel:\n");
        arm.push_str("    // Load input1 address\n");
        arm.push_str("    ldr x0, [x0]\n");
        arm.push_str("    // Load input2 address\n");
        arm.push_str("    ldr x1, [x1]\n");
        arm.push_str("    // Load output address\n");
        arm.push_str("    ldr x2, [x2]\n");
        arm.push_str("    // Load values\n");
        arm.push_str("    ldr s0, [x0]\n");
        arm.push_str("    ldr s1, [x1]\n");
        arm.push_str("    // Multiply\n");
        arm.push_str("    fmul s0, s0, s1\n");
        arm.push_str("    // Store result\n");
        arm.push_str("    str s0, [x2]\n");
        arm.push_str("    ret\n\n");
        arm
    }
    
    /// Generate ARM assembly for convolution
    fn generate_convolution_arm(&self, _operation: &Operation) -> String {
        let mut arm = String::new();
        arm.push_str("conv_kernel:\n");
        arm.push_str("    // Load input address\n");
        arm.push_str("    ldr x0, [x0]\n");
        arm.push_str("    // Load weights address\n");
        arm.push_str("    ldr x1, [x1]\n");
        arm.push_str("    // Load output address\n");
        arm.push_str("    ldr x2, [x2]\n");
        arm.push_str("    // Load values\n");
        arm.push_str("    ldr s0, [x0]\n");
        arm.push_str("    ldr s1, [x1]\n");
        arm.push_str("    // Multiply and accumulate\n");
        arm.push_str("    fmul s0, s0, s1\n");
        arm.push_str("    // Store result\n");
        arm.push_str("    str s0, [x2]\n");
        arm.push_str("    ret\n\n");
        arm
    }
    
    /// Generate ARM assembly for pooling
    fn generate_pooling_arm(&self, _operation: &Operation) -> String {
        let mut arm = String::new();
        arm.push_str("pool_kernel:\n");
        arm.push_str("    // Load input address\n");
        arm.push_str("    ldr x0, [x0]\n");
        arm.push_str("    // Load output address\n");
        arm.push_str("    ldr x1, [x1]\n");
        arm.push_str("    // Load value\n");
        arm.push_str("    ldr s0, [x0]\n");
        arm.push_str("    // Store result (simple pass-through for now)\n");
        arm.push_str("    str s0, [x1]\n");
        arm.push_str("    ret\n\n");
        arm
    }
}

impl CodeGenerator for EdgeCodeGenerator {
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>> {
        self.generate_edge_code(&optimized_ir.optimized_nodes.iter().map(|node| {
            let mut op = Operation::new(node.op_type.clone());
            op.attributes = node.attributes.clone();
            op
        }).collect::<Vec<_>>())
    }
    
    fn target_hardware(&self) -> HardwareTarget {
        HardwareTarget::Edge
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

impl CodeGenerator for BackendTranslatorCodeGenerator {
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>> {
        println!("🔧 Generating code using backend translators for {:?}", self.target);
        
        // Generate code using backend translators
        let machine_code = self.generate_backend_translator_code(&optimized_ir.optimized_nodes.iter().map(|node| {
            let mut op = Operation::new(node.op_type.clone());
            op.attributes = node.attributes.clone();
            op
        }).collect::<Vec<_>>())?;
        
        println!("✅ Generated {} bytes of machine code using backend translators", machine_code.len());
        Ok(machine_code)
    }
    
    fn target_hardware(&self) -> HardwareTarget {
        self.target.clone()
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

