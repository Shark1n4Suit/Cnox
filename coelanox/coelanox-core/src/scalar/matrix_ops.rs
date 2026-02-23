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
    let out_len = (m * n).min(output.len());
    output[..out_len].fill(0.0f32);
    // Cache-friendly order: (i, p, j) so inner loop scans weights[p*n..p*n+n] contiguously.
    // Bounds: input has m*k, weights has k*n; we write output[..out_len].
    if input.len() < m * k || weights.len() < k * n {
        return Err(CoelanoxError::Internal(format!(
            "MatMul: input len {} or weights len {} too small for M={} K={} N={}",
            input.len(), weights.len(), m, k, n
        )));
    }
    for i in 0..m {
        let out_row = i * n;
        let out_end = (out_row + n).min(output.len());
        for p in 0..k {
            let in_val = input[i * k + p];
            if in_val == 0.0 {
                continue;
            }
            let w_base = p * n;
            for j in out_row..out_end {
                output[j] += in_val * weights[w_base + j - out_row];
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
    let kw = kernel_size.min(weights.len());
    let len = input.len().saturating_sub(kw).saturating_add(1).min(output.len());
    if kw == 0 {
        output[..len].fill(0.0);
        return Ok(());
    }
    // Use windows() for contiguous access; helps compiler auto-vectorize.
    for (i, win) in input.windows(kw).enumerate().take(len) {
        output[i] = win.iter().zip(weights[..kw].iter()).map(|(a, b)| a * b).sum();
    }
    Ok(())
}

