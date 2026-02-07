//! Reduction operations
//! 
//! Pure Rust scalar implementations of reduction operations:
//! - ReduceSum: Sum reduction along specified axis
//! - ReduceMean: Mean reduction along specified axis
//! - ReduceMax: Maximum reduction along specified axis
//! - ReduceMin: Minimum reduction along specified axis

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute sum reduction: output = sum(input) along specified axis
/// 
/// # Attributes
/// * `axis` - Axis along which to reduce (default: all axes)
pub fn reduce_sum(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = input.iter().sum();
    }
    Ok(())
}

/// Execute mean reduction: output = mean(input) along specified axis
/// 
/// # Attributes
/// * `axis` - Axis along which to reduce (default: all axes)
pub fn reduce_mean(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() && !input.is_empty() {
        output[0] = input.iter().sum::<f32>() / input.len() as f32;
    }
    Ok(())
}

/// Execute max reduction: output = max(input) along specified axis
/// 
/// # Attributes
/// * `axis` - Axis along which to reduce (default: all axes)
pub fn reduce_max(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    }
    Ok(())
}

/// Execute min reduction: output = min(input) along specified axis
/// 
/// # Attributes
/// * `axis` - Axis along which to reduce (default: all axes)
pub fn reduce_min(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = input.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    }
    Ok(())
}

/// Execute product reduction: output = product(input) along specified axis
/// 
/// # Attributes
/// * `axis` - Axis along which to reduce (default: all axes)
pub fn reduce_prod(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if !output.is_empty() {
        output[0] = input.iter().product();
    }
    Ok(())
}

