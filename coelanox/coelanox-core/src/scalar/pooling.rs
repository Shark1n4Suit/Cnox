//! Pooling operations
//! 
//! Pure Rust scalar implementations of pooling operations:
//! - MaxPool: Maximum pooling with configurable kernel and stride
//! - AvgPool: Average pooling with configurable kernel and stride
//! - GlobalMaxPool: Global maximum pooling
//! - GlobalAvgPool: Global average pooling

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute max pooling operation
/// 
/// # Attributes
/// * `kernel_size` - Size of the pooling kernel (default: 2)
/// * `stride` - Stride of the pooling operation (default: kernel_size)
pub fn max_pool(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let kernel_size = attributes.get("kernel_size")
        .and_then(|a| match a { crate::ir::Attribute::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(2);
    let stride = attributes.get("stride")
        .and_then(|a| match a { crate::ir::Attribute::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(kernel_size);
    let mut out_idx = 0;
    let mut i = 0;
    while i + kernel_size <= input.len() && out_idx < output.len() {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..kernel_size {
            max_val = max_val.max(input[i + j]);
        }
        output[out_idx] = max_val;
        out_idx += 1;
        i += stride;
    }
    Ok(())
}

/// Execute average pooling operation
/// 
/// # Attributes
/// * `kernel_size` - Size of the pooling kernel (default: 2)
/// * `stride` - Stride of the pooling operation (default: kernel_size)
pub fn avg_pool(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let kernel_size = attributes.get("kernel_size")
        .and_then(|a| match a { crate::ir::Attribute::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(2);
    let stride = attributes.get("stride")
        .and_then(|a| match a { crate::ir::Attribute::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(kernel_size);
    let mut out_idx = 0;
    let mut i = 0;
    while i + kernel_size <= input.len() && out_idx < output.len() {
        let mut sum = 0.0f32;
        for j in 0..kernel_size {
            sum += input[i + j];
        }
        output[out_idx] = sum / kernel_size as f32;
        out_idx += 1;
        i += stride;
    }
    Ok(())
}

/// Execute global max pooling: output = max(input)
pub fn global_max_pool(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    }
    Ok(())
}

/// Execute global average pooling: output = mean(input)
pub fn global_avg_pool(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        output[0] = input.iter().sum::<f32>() / input.len() as f32;
    }
    Ok(())
}

