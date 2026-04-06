//! Comparison operations
//!
//! Pure Rust scalar implementations of comparison operations:
//! - Equal: Element-wise equality comparison
//! - NotEqual: Element-wise inequality comparison
//! - Greater: Element-wise greater-than comparison
//! - GreaterEqual: Element-wise greater-than-or-equal comparison
//! - Less: Element-wise less-than comparison
//! - LessEqual: Element-wise less-than-or-equal comparison

use crate::ir::Attribute;
use crate::CoelanoxResult;
use std::collections::HashMap;

/// Execute element-wise equality: output = (input_a == input_b) ? 1.0 : 0.0
pub fn equal(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if (input_a[i] - input_b[i]).abs() < f32::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    Ok(())
}

/// Execute element-wise inequality: output = (input_a != input_b) ? 1.0 : 0.0
pub fn not_equal(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if (input_a[i] - input_b[i]).abs() >= f32::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    Ok(())
}

/// Execute element-wise greater-than: output = (input_a > input_b) ? 1.0 : 0.0
pub fn greater(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] > input_b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Execute element-wise greater-than-or-equal: output = (input_a >= input_b) ? 1.0 : 0.0
pub fn greater_equal(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] >= input_b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Execute element-wise less-than: output = (input_a < input_b) ? 1.0 : 0.0
pub fn less(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] < input_b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}

/// Execute element-wise less-than-or-equal: output = (input_a <= input_b) ? 1.0 : 0.0
pub fn less_equal(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] <= input_b[i] { 1.0 } else { 0.0 };
    }
    Ok(())
}
