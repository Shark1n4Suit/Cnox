//! Normalization operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers
//! (same pattern as MatMul: use Rayon when beneficial).

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute batch normalization: output = gamma * (input - mean) / sqrt(var + epsilon) + beta
pub fn batch_norm(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let epsilon = attributes.get("epsilon")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(1e-5);
    let len = input.len().min(output.len());
    let mean = input.iter().sum::<f32>() / len as f32;
    let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len as f32 + epsilon;
    let std_inv = 1.0 / var.sqrt();
    let gamma = weights.get(0).copied().unwrap_or(1.0);
    let beta = weights.get(1).copied().unwrap_or(0.0);
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = gamma * (x - mean) * std_inv + beta);
    } else {
        for i in 0..len {
            output[i] = gamma * (input[i] - mean) * std_inv + beta;
        }
    }
    Ok(())
}

/// Execute layer normalization: output = gamma * (input - mean) / sqrt(var + epsilon) + beta
///
/// Normalizes over the last dimension (standard LayerNorm). Weights are [gamma, beta] concatenated,
/// each of length H (normalized dimension). So weights.len() == 2*H. For each row of H elements
/// we compute mean and variance, then apply scale and shift. Agnostic: used by any model (e.g. CNNs,
/// RNNs, transformers) that uses layer normalization.
///
/// # Arguments
/// * `input` - Input tensor (flattened; total elements must be divisible by H)
/// * `output` - Output buffer (same size as input)
/// * `weights` - [gamma (H elements), beta (H elements)]; H = weights.len() / 2
/// * `attributes` - "epsilon" (Float, default 1e-5); optional "normalized_shape" (IntArray) to specify H
///
/// # Returns
/// * `Ok(())` if operation succeeds
pub fn layer_norm(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let epsilon = attributes.get("epsilon")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(1e-5);

    // Normalized dimension H: from attribute or from weights (gamma + beta each length H => 2*H).
    let h = attributes.get("normalized_shape")
        .and_then(|a| match a {
            crate::ir::Attribute::IntArray(v) => v.first().map(|&x| x as usize),
            _ => None,
        })
        .unwrap_or_else(|| weights.len() / 2);

    if h == 0 || weights.len() < 2 * h {
        return Err(CoelanoxError::Internal(format!(
            "LayerNorm: weights len {} too small for normalized dim H={} (need 2*H)",
            weights.len(), h
        )));
    }
    let gamma = &weights[0..h];
    let beta = &weights[h..2 * h];

    let len = input.len().min(output.len());
    if len % h != 0 {
        return Err(CoelanoxError::Internal(format!(
            "LayerNorm: input len {} not divisible by normalized dim {}",
            len, h
        )));
    }

    let num_rows = len / h;
    if num_rows >= PAR_THRESHOLD / h.max(1) {
        output.par_chunks_mut(h)
            .zip(input.par_chunks(h))
            .for_each(|(row_out, row_in): (&mut [f32], &[f32])| {
                let mean: f32 = row_in.iter().sum::<f32>() / h as f32;
                let var: f32 = row_in.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / h as f32 + epsilon;
                let std_inv = 1.0 / var.sqrt();
                for i in 0..h {
                    row_out[i] = gamma[i] * (row_in[i] - mean) * std_inv + beta[i];
                }
            });
    } else {
        for row in 0..num_rows {
            let start = row * h;
            let end = start + h;
            let row_in = &input[start..end];
            let row_out = &mut output[start..end];

            let mean: f32 = row_in.iter().sum::<f32>() / h as f32;
            let var: f32 = row_in.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / h as f32 + epsilon;
            let std_inv = 1.0 / var.sqrt();

            for i in 0..h {
                row_out[i] = gamma[i] * (row_in[i] - mean) * std_inv + beta[i];
            }
        }
    }
    Ok(())
}

