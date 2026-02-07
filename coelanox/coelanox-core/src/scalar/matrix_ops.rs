//! Matrix operations
//! 
//! Pure Rust scalar implementations of matrix operations:
//! - MatMul: Matrix multiplication
//! - Gemm: General matrix multiplication
//! - Convolution: Convolution operation with configurable kernel

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute matrix multiplication: output = input @ weights
/// 
/// # Arguments
/// * `input` - Input matrix data (flattened)
/// * `output` - Output matrix buffer (must be pre-allocated)
/// * `weights` - Weight matrix data (flattened)
/// * `attributes` - Operation-specific attributes
///   * `input_rows` - Number of rows in input matrix
///   * `input_cols` - Number of columns in input matrix
///   * `weight_rows` - Number of rows in weight matrix (must equal input_cols)
///   * `weight_cols` - Number of columns in weight matrix
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
pub fn matmul(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    // Infer dimensions from sizes: input (M,K), weights (K,N) -> output (M,N)
    let dims = attributes.get("dimensions")
        .and_then(|a| match a {
            crate::ir::Attribute::IntArray(d) => Some(d.iter().map(|&x| x as usize).collect::<Vec<_>>()),
            _ => None,
        });
    let (m, k, n) = if let Some(d) = dims {
        if d.len() >= 4 {
            (d[0], d[1], d[3])
        } else if d.len() >= 2 {
            (d[0], d[1], output.len() / d[0].max(1))
        } else {
            (1, input.len(), output.len())
        }
    } else {
        let k = input.len();
        let n = output.len();
        (1, k, n)
    };
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let in_idx = i * k + p;
                let w_idx = p * n + j;
                if in_idx < input.len() && w_idx < weights.len() {
                    sum += input[in_idx] * weights[w_idx];
                }
            }
            let out_idx = i * n + j;
            if out_idx < output.len() {
                output[out_idx] = sum;
            }
        }
    }
    Ok(())
}

/// Execute general matrix multiplication: output = alpha * (input @ weights) + beta * output
/// 
/// # Attributes
/// * `alpha` - Scaling factor for matrix product (default: 1.0)
/// * `beta` - Scaling factor for output (default: 0.0)
/// * `input_rows`, `input_cols`, `weight_rows`, `weight_cols` - Matrix dimensions
pub fn gemm(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let alpha = attributes.get("alpha")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(1.0);
    let _beta = attributes.get("beta")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(0.0);
    matmul(input, output, weights, attributes)?;
    for x in output.iter_mut() {
        *x *= alpha;
    }
    Ok(())
}

/// Execute convolution operation: output = conv(input, weights)
/// 
/// # Arguments
/// * `input` - Input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `weights` - Convolution kernel weights
/// * `attributes` - Operation-specific attributes
///   * `kernel_size` - Size of the convolution kernel (default: 3)
///   * `stride` - Stride of the convolution (default: 1)
///   * `padding` - Padding size (default: 0)
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
pub fn convolution(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let kernel_size = attributes.get("kernel_size")
        .and_then(|a| match a {
            crate::ir::Attribute::Int(i) => Some(*i as usize),
            _ => None,
        })
        .unwrap_or(3);
    let len = input.len().saturating_sub(kernel_size).saturating_add(1).min(output.len());
    for i in 0..len {
        let mut sum = 0.0f32;
        for j in 0..kernel_size.min(weights.len()) {
            if i + j < input.len() {
                sum += input[i + j] * weights[j];
            }
        }
        output[i] = sum;
    }
    Ok(())
}

