//! Reduction operations
//!
//! Pure Rust scalar implementations with parallel reduction for large buffers
//! (same pattern as MatMul: use Rayon when beneficial).

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute sum reduction: output = sum(input)
pub fn reduce_sum(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = if input.len() >= PAR_THRESHOLD {
            input.par_iter().sum()
        } else {
            input.iter().sum()
        };
    }
    Ok(())
}

/// Execute mean reduction: output = mean(input)
pub fn reduce_mean(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        let (sum, n) = if input.len() >= PAR_THRESHOLD {
            (input.par_iter().sum::<f32>(), input.len())
        } else {
            (input.iter().sum::<f32>(), input.len())
        };
        output[0] = sum / n as f32;
    }
    Ok(())
}

/// Execute max reduction: output = max(input)
pub fn reduce_max(
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

/// Execute min reduction: output = min(input)
pub fn reduce_min(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        output[0] = if input.len() >= PAR_THRESHOLD {
            input.par_iter().fold(|| f32::INFINITY, |a, &b| a.min(b)).reduce(|| f32::INFINITY, |a, b| a.min(b))
        } else {
            input.iter().fold(f32::INFINITY, |a, &b| a.min(b))
        };
    }
    Ok(())
}

/// Execute product reduction: output = product(input)
pub fn reduce_prod(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        output[0] = if input.len() >= PAR_THRESHOLD {
            input.par_iter().fold(|| 1.0f32, |a, &b| a * b).reduce(|| 1.0f32, |a, b| a * b)
        } else {
            input.iter().product()
        };
    }
    Ok(())
}

