//! Main CPU code generator module
//! This module coordinates all operation generators and infrastructure

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::HardwareTarget;
use coelanox_core::ir::{OptimizedIR, GraphNode, OpType, Shape, TensorId};
use iced_x86::Register;
use crate::generators::CodeGenerator;
use std::sync::Mutex;

// Infrastructure
mod infrastructure;

// Helpers
mod helpers;
pub use helpers::elementwise;

// Operations
mod operations;
use operations::{
    activations, arithmetic, comparisons, logical, pooling, normalization,
    convolution, linear, reductions as op_reductions, tensor_ops, tensor_manip, misc
};

/// Operation generator function type
/// Now includes shape information for multi-dimensional support
type OpGenerator = fn(&CpuCodeGenerator, &GraphNode, Option<&Shape>, Option<&Shape>) -> CoelanoxResult<Vec<u8>>;

/// CPU code generator with modular SIMD architecture
pub struct CpuCodeGenerator {
    name: String,
    version: String,
    /// Current operation context (buffer registers for active operation)
    /// This is set per-operation during code generation
    /// Uses Mutex for thread-safe interior mutability (trait requires Send+Sync)
    current_input_buffer: Mutex<Option<Register>>,
    current_output_buffer: Mutex<Option<Register>>,
    /// Current operation shape information (for multi-dimensional support)
    current_input_shape: Mutex<Option<Shape>>,
    current_output_shape: Mutex<Option<Shape>>,
    /// Buffer map for looking up buffer registers for specific tensors
    /// Used for operations that need multiple input buffers (e.g., Gather, Concatenate)
    buffer_map: Mutex<Option<std::collections::HashMap<TensorId, Register>>>,
}

impl CpuCodeGenerator {
    /// Create a new CPU code generator
    pub fn new() -> Self {
        Self {
            name: "cpu_simd_generator".to_string(),
            version: "2.0.0".to_string(),
            current_input_buffer: Mutex::new(None),
            current_output_buffer: Mutex::new(None),
            current_input_shape: Mutex::new(None),
            current_output_shape: Mutex::new(None),
            buffer_map: Mutex::new(None),
        }
    }
    
    /// Get generator function for an operation type
    /// Uses match statement instead of HashMap (OpType doesn't implement Hash)
    fn get_op_generator(&self, op_type: &OpType) -> Option<OpGenerator> {
        match op_type {
            // Basic arithmetic (4) - all implemented
            OpType::Add => Some(arithmetic::generate_add as OpGenerator),
            OpType::Subtract => Some(arithmetic::generate_subtract as OpGenerator),
            OpType::Multiply => Some(arithmetic::generate_multiply as OpGenerator),
            OpType::Divide => Some(arithmetic::generate_divide as OpGenerator),
            
            // Element-wise operations (7)
            OpType::Relu => Some(activations::generate_relu as OpGenerator),
            OpType::Sigmoid => Some(activations::generate_sigmoid as OpGenerator),
            OpType::Tanh => Some(activations::generate_tanh as OpGenerator),
            OpType::Softmax => Some(activations::generate_softmax as OpGenerator),
            OpType::LogSoftmax => Some(activations::generate_logsoftmax as OpGenerator),
            OpType::Gelu => Some(activations::generate_gelu as OpGenerator),
            OpType::Swish => Some(activations::generate_swish as OpGenerator),
            
            // Mathematical primitives (6) - for RMSNorm, RoPE, etc.
            OpType::Sqrt => Some(activations::generate_sqrt as OpGenerator),
            OpType::Pow => Some(activations::generate_pow as OpGenerator),
            OpType::Cos => Some(activations::generate_cos as OpGenerator),
            OpType::Sin => Some(activations::generate_sin as OpGenerator),
            OpType::Exp => Some(activations::generate_exp as OpGenerator),
            OpType::Log => Some(activations::generate_log as OpGenerator),
            
            // Neural network primitives (8)
            OpType::Convolution => Some(convolution::generate_convolution as OpGenerator),
            OpType::MaxPool => Some(pooling::generate_maxpool as OpGenerator),
            OpType::AvgPool => Some(pooling::generate_avgpool as OpGenerator),
            OpType::GlobalMaxPool => Some(pooling::generate_globalmaxpool as OpGenerator),
            OpType::GlobalAvgPool => Some(pooling::generate_globalavgpool as OpGenerator),
            OpType::BatchNorm => Some(normalization::generate_batchnorm as OpGenerator),
            OpType::LayerNorm => Some(normalization::generate_layernorm as OpGenerator),
            OpType::Dropout => Some(misc::generate_dropout as OpGenerator),
            
            // Tensor manipulation (8)
            OpType::Reshape => Some(tensor_ops::generate_reshape as OpGenerator),
            OpType::Transpose => Some(tensor_ops::generate_transpose as OpGenerator),
            OpType::Permute => Some(tensor_ops::generate_permute as OpGenerator),
            OpType::Concatenate => Some(tensor_manip::generate_concatenate as OpGenerator),
            OpType::Split => Some(tensor_manip::generate_split as OpGenerator),
            OpType::Slice => Some(tensor_ops::generate_slice as OpGenerator),
            OpType::Gather => Some(tensor_manip::generate_gather as OpGenerator),
            OpType::Scatter => Some(tensor_manip::generate_scatter as OpGenerator),
            
            // Matrix operations (2)
            OpType::MatMul => Some(linear::generate_matmul as OpGenerator),
            OpType::Gemm => Some(linear::generate_gemm as OpGenerator),
            
            // Reduction operations (5)
            OpType::ReduceSum => Some(op_reductions::generate_reducesum as OpGenerator),
            OpType::ReduceMean => Some(op_reductions::generate_mean as OpGenerator), // generate_mean is the function name
            OpType::ReduceMax => Some(op_reductions::generate_reducemax as OpGenerator),
            OpType::ReduceMin => Some(op_reductions::generate_reducemin as OpGenerator),
            OpType::ReduceProd => Some(op_reductions::generate_reduceprod as OpGenerator),
            
            // Broadcasting operations (2)
            OpType::Broadcast => Some(misc::generate_broadcast as OpGenerator),
            OpType::Expand => Some(misc::generate_expand as OpGenerator),
            
            // Comparison operations (6)
            OpType::Equal => Some(comparisons::generate_equal as OpGenerator),
            OpType::NotEqual => Some(comparisons::generate_notequal as OpGenerator),
            OpType::Greater => Some(comparisons::generate_greater as OpGenerator),
            OpType::GreaterEqual => Some(comparisons::generate_greaterequal as OpGenerator),
            OpType::Less => Some(comparisons::generate_less as OpGenerator),
            OpType::LessEqual => Some(comparisons::generate_lessequal as OpGenerator),
            
            // Logical operations (3)
            OpType::And => Some(logical::generate_and as OpGenerator),
            OpType::Or => Some(logical::generate_or as OpGenerator),
            OpType::Not => Some(logical::generate_not as OpGenerator),
            
            // Custom
            OpType::Custom(_) => Some(misc::generate_custom as OpGenerator),
            
            // Deprecated - not implemented
            _ => None,
        }
    }
}

impl CodeGenerator for CpuCodeGenerator {
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>> {
        let mut code = Vec::new();
        
        // Calculate total intermediate buffer size needed
        let total_intermediate_size = optimized_ir.intermediate_tensors.iter()
            .map(|t| t.size_bytes())
            .sum::<usize>();
        
        // Generate prologue (allocates intermediate buffers on stack)
        code.extend_from_slice(&self.generate_prologue_with_buffers(total_intermediate_size)?);
        
        // Track buffer pointers: TensorId -> Register
        // For intermediate tensors, we use a register rotation strategy:
        // - First 2 tensors: use dedicated registers (R11, R13) for efficiency
        // - Additional tensors: reuse R11/R13 with stack offsets (since tensors are used sequentially)
        // The bit-packed TensorId system allows O(1) lookups regardless of tensor count
        let mut buffer_map: std::collections::HashMap<coelanox_core::ir::TensorId, Register> = std::collections::HashMap::new();
        
        // Calculate stack offsets for intermediate buffers
        let mut stack_offset = 0usize;
        let mut intermediate_registers = vec![Register::R11, Register::R13];
        
        // Map ALL intermediate tensors to buffers
        // For models with >2 intermediate tensors, we reuse registers (R11, R13) in rotation
        // This works because intermediate tensors are typically used sequentially
        for (idx, tensor) in optimized_ir.intermediate_tensors.iter().enumerate() {
            let tensor_id = coelanox_core::ir::TensorId::intermediate(idx);
            // Use register rotation: idx % 2 selects between R11 and R13
            // This ensures all intermediate tensors are in buffer_map
            let register = intermediate_registers[idx % intermediate_registers.len()];
            buffer_map.insert(tensor_id, register);
            stack_offset += tensor.size_bytes();
        }
        
        // Generate code for each operation node
        for (idx, node) in optimized_ir.optimized_nodes.iter().enumerate() {
            // Determine input/output buffers for this operation
            let (input_buffer_reg, output_buffer_reg) = self.determine_buffers(
                node,
                &optimized_ir,
                &buffer_map,
                idx
            )?;
            
            // Get input/output shapes for multi-dimensional support
            let input_shape = self.get_input_shape(node, optimized_ir);
            let output_shape = self.get_output_shape(node, optimized_ir);
            
            // Generate operation with specific buffer pointers and shapes
            let operation_code = self.generate_operation_with_buffers(
                node,
                input_buffer_reg,
                output_buffer_reg,
                input_shape.as_ref(),
                output_shape.as_ref(),
                &buffer_map
            )?;
            code.extend_from_slice(&operation_code);
        }
        
        // Generate epilogue (deallocates stack buffers)
        code.extend_from_slice(&self.generate_epilogue_with_buffers(total_intermediate_size));
        
        Ok(code)
    }
    
    fn target_hardware(&self) -> HardwareTarget {
        HardwareTarget::Cpu
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

// Re-export infrastructure methods
use infrastructure::*;
