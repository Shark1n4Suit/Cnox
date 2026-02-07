//! Normalization operations
//! 
//! Pure Rust scalar implementations of normalization operations:
//! - BatchNorm: Batch normalization with learnable parameters
//! - LayerNorm: Layer normalization with learnable parameters

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute batch normalization: output = gamma * (input - mean) / sqrt(var + epsilon) + beta
/// 
/// # Arguments
/// * `input` - Input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `weights` - Weight tensor containing [gamma, beta, mean, variance] in that order
/// * `attributes` - Operation-specific attributes
///   * `epsilon` - Small value for numerical stability (default: 1e-5)
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
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
    for i in 0..len {
        output[i] = gamma * (input[i] - mean) * std_inv + beta;
    }
    Ok(())
}

/// Execute layer normalization: output = gamma * (input - mean) / sqrt(var + epsilon) + beta
/// 
/// # Arguments
/// * `input` - Input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `weights` - Weight tensor containing [gamma, beta] in that order
/// * `attributes` - Operation-specific attributes
///   * `epsilon` - Small value for numerical stability (default: 1e-5)
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
pub fn layer_norm(
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
    for i in 0..len {
        output[i] = gamma * (input[i] - mean) * std_inv + beta;
    }
    Ok(())
}

