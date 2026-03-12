//! Pooling operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers
//! (same pattern as MatMul: use Rayon when beneficial).

use crate::CoelanoxResult;
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute max pooling operation
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
    let max_out = (input.len().saturating_sub(kernel_size) / stride).saturating_add(1).min(output.len());
    if max_out >= PAR_THRESHOLD {
        output.par_iter_mut()
            .take(max_out)
            .enumerate()
            .for_each(|(out_idx, o)| {
                let i = out_idx * stride;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..kernel_size {
                    max_val = max_val.max(input[i + j]);
                }
                *o = max_val;
            });
    } else {
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
    }
    Ok(())
}

/// Execute average pooling operation
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
    let max_out = (input.len().saturating_sub(kernel_size) / stride).saturating_add(1).min(output.len());
    if max_out >= PAR_THRESHOLD {
        output.par_iter_mut()
            .take(max_out)
            .enumerate()
            .for_each(|(out_idx, o)| {
                let i = out_idx * stride;
                let sum: f32 = input[i..i + kernel_size].iter().sum();
                *o = sum / kernel_size as f32;
            });
    } else {
        let mut out_idx = 0;
        let mut i = 0;
        while i + kernel_size <= input.len() && out_idx < output.len() {
            let sum: f32 = input[i..i + kernel_size].iter().sum();
            output[out_idx] = sum / kernel_size as f32;
            out_idx += 1;
            i += stride;
        }
    }
    Ok(())
}

/// Execute global max pooling: output = max(input)
pub fn global_max_pool(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        output[0] = if input.len() >= PAR_THRESHOLD {
            input.par_iter().fold(|| f32::NEG_INFINITY, |a, &b| a.max(b)).reduce(|| f32::NEG_INFINITY, |a, b| a.max(b))
        } else {
            input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        };
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
        let (sum, n) = if input.len() >= PAR_THRESHOLD {
            let sum: f32 = input.par_iter().sum();
            (sum, input.len())
        } else {
            (input.iter().sum::<f32>(), input.len())
        };
        output[0] = sum / n as f32;
    }
    Ok(())
}

