//! OpType to Scalar Function Mapping
//!
//! This module provides a 1:1 mapping between OpType enum variants and scalar functions.
//! Custom(String) has no generic implementation and is unsupported.

use crate::ir::Attribute;
use crate::ir::OpType;
use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;

// Re-export all scalar operations
use super::{
    abs,
    // Arithmetic
    add,
    // Logical
    and,
    avg_pool,
    // Normalization
    batch_norm,
    broadcast,
    concatenate,
    convolution,
    cos,
    divide,
    // Misc
    dropout,
    // Comparisons
    equal,
    exp,
    expand,
    gather,
    gelu,
    gemm,
    global_avg_pool,
    global_max_pool,
    greater,
    greater_equal,
    layer_norm,
    less,
    less_equal,
    log,
    log_softmax,
    // Matrix ops
    matmul,
    max,
    // Pooling
    max_pool,
    min,
    multiply,
    not,
    not_equal,
    or,
    permute,
    pow,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    // Reductions
    reduce_sum,
    // Activations
    relu,
    // Tensor ops
    reshape,
    scatter,
    sigmoid,
    sin,
    slice,
    softmax,
    split,
    // Math
    sqrt,
    subtract,
    swish,
    tanh,
    transpose,
};

/// Execute scalar operation based on OpType
///
/// This function provides a 1:1 mapping from OpType to scalar implementations.
///
/// # Arguments
/// * `op_type` - The operation type from OpType enum
/// * `input_a` - First input tensor (or single input for unary ops)
/// * `input_b` - Second input tensor (optional, for binary ops)
/// * `output` - Output tensor buffer
/// * `weights` - Weight tensor (optional, for operations that need weights)
/// * `attributes` - Operation-specific attributes
///
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails or OpType is Custom
pub fn execute_scalar_op(
    op_type: &OpType,
    input_a: &[f32],
    input_b: Option<&[f32]>,
    output: &mut [f32],
    weights: Option<&[f32]>,
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    match op_type {
        // Arithmetic operations
        OpType::Add => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Add requires input_b".to_string()))?;
            add(input_a, input_b, output, attributes)
        }
        OpType::Subtract => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Subtract requires input_b".to_string()))?;
            subtract(input_a, input_b, output, attributes)
        }
        OpType::Multiply => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Multiply requires input_b".to_string()))?;
            multiply(input_a, input_b, output, attributes)
        }
        OpType::Divide => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Divide requires input_b".to_string()))?;
            divide(input_a, input_b, output, attributes)
        }
        OpType::Min => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Min requires input_b".to_string()))?;
            min(input_a, input_b, output, attributes)
        }
        OpType::Max => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Max requires input_b".to_string()))?;
            max(input_a, input_b, output, attributes)
        }

        // Activation functions
        OpType::Relu => relu(input_a, output, attributes),
        OpType::Sigmoid => sigmoid(input_a, output, attributes),
        OpType::Tanh => tanh(input_a, output, attributes),
        OpType::Softmax => softmax(input_a, output, attributes),
        OpType::LogSoftmax => log_softmax(input_a, output, attributes),
        OpType::Gelu => gelu(input_a, output, attributes),
        OpType::Swish => swish(input_a, output, attributes),

        // Mathematical primitives
        OpType::Sqrt => sqrt(input_a, output, attributes),
        OpType::Abs => abs(input_a, output, attributes),
        OpType::Pow => pow(input_a, output, attributes),
        OpType::Cos => cos(input_a, output, attributes),
        OpType::Sin => sin(input_a, output, attributes),
        OpType::Exp => exp(input_a, output, attributes),
        OpType::Log => log(input_a, output, attributes),

        // Neural network primitives
        OpType::Convolution => {
            let weights = weights.ok_or_else(|| {
                CoelanoxError::Internal("Convolution requires weights".to_string())
            })?;
            convolution(input_a, output, weights, attributes)
        }
        OpType::MaxPool => max_pool(input_a, output, attributes),
        OpType::AvgPool => avg_pool(input_a, output, attributes),
        OpType::GlobalMaxPool => global_max_pool(input_a, output, attributes),
        OpType::GlobalAvgPool => global_avg_pool(input_a, output, attributes),
        OpType::BatchNorm => {
            let weights = weights
                .ok_or_else(|| CoelanoxError::Internal("BatchNorm requires weights".to_string()))?;
            batch_norm(input_a, output, weights, attributes)
        }
        OpType::LayerNorm => {
            let weights = weights
                .ok_or_else(|| CoelanoxError::Internal("LayerNorm requires weights".to_string()))?;
            layer_norm(input_a, output, weights, attributes)
        }
        OpType::Dropout => dropout(input_a, output, attributes),

        // Tensor manipulation
        OpType::Reshape => reshape(input_a, output, attributes),
        OpType::Transpose => transpose(input_a, output, attributes),
        OpType::Permute => permute(input_a, output, attributes),
        OpType::Concatenate => {
            let input_b = input_b.ok_or_else(|| {
                CoelanoxError::Internal("Concatenate requires second input (input_b)".to_string())
            })?;
            let inputs = [input_a, input_b];
            concatenate(&inputs, output, attributes)
        }
        OpType::Split => {
            // Single output buffer: copy input into it (logical split is in attributes for consumers)
            split(input_a, &mut [output], attributes)
        }
        OpType::Slice => slice(input_a, output, attributes),
        OpType::Gather => {
            let input_b = input_b.ok_or_else(|| {
                CoelanoxError::Internal("Gather requires indices (input_b)".to_string())
            })?;
            let indices: Vec<i32> = input_b.iter().map(|&f| f as i32).collect();
            gather(input_a, &indices, output, attributes)
        }
        OpType::Scatter => {
            let input_b = input_b.ok_or_else(|| {
                CoelanoxError::Internal("Scatter requires indices (input_b)".to_string())
            })?;
            let indices: Vec<i32> = input_b.iter().map(|&f| f as i32).collect();
            scatter(input_a, &indices, output, attributes)
        }
        OpType::Broadcast => broadcast(input_a, output, attributes),
        OpType::Expand => expand(input_a, output, attributes),

        // Matrix operations
        OpType::MatMul => {
            let w = weights.or(input_b).ok_or_else(|| {
                CoelanoxError::Internal("MatMul requires weights or second input".to_string())
            })?;
            matmul(input_a, output, w, attributes)
        }
        OpType::Gemm => {
            let w = weights.or(input_b).ok_or_else(|| {
                CoelanoxError::Internal("Gemm requires weights or second input".to_string())
            })?;
            gemm(input_a, output, w, attributes)
        }

        // Reduction operations
        OpType::ReduceSum => reduce_sum(input_a, output, attributes),
        OpType::ReduceMean => reduce_mean(input_a, output, attributes),
        OpType::ReduceMax => reduce_max(input_a, output, attributes),
        OpType::ReduceMin => reduce_min(input_a, output, attributes),
        OpType::ReduceProd => reduce_prod(input_a, output, attributes),

        // Comparison operations
        OpType::Equal => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Equal requires input_b".to_string()))?;
            equal(input_a, input_b, output, attributes)
        }
        OpType::NotEqual => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("NotEqual requires input_b".to_string()))?;
            not_equal(input_a, input_b, output, attributes)
        }
        OpType::Greater => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Greater requires input_b".to_string()))?;
            greater(input_a, input_b, output, attributes)
        }
        OpType::GreaterEqual => {
            let input_b = input_b.ok_or_else(|| {
                CoelanoxError::Internal("GreaterEqual requires input_b".to_string())
            })?;
            greater_equal(input_a, input_b, output, attributes)
        }
        OpType::Less => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Less requires input_b".to_string()))?;
            less(input_a, input_b, output, attributes)
        }
        OpType::LessEqual => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("LessEqual requires input_b".to_string()))?;
            less_equal(input_a, input_b, output, attributes)
        }

        // Logical operations
        OpType::And => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("And requires input_b".to_string()))?;
            and(input_a, input_b, output, attributes)
        }
        OpType::Or => {
            let input_b = input_b
                .ok_or_else(|| CoelanoxError::Internal("Or requires input_b".to_string()))?;
            or(input_a, input_b, output, attributes)
        }
        OpType::Not => not(input_a, output, attributes),

        // Custom operations are not supported (no generic implementation)
        OpType::Custom(_) => Err(CoelanoxError::Internal(
            "Custom operations are not supported in scalar backend".to_string(),
        )),
    }
}
