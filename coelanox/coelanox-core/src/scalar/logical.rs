//! Logical operations
//!
//! Pure Rust scalar implementations of logical operations:
//! - And: Element-wise logical AND
//! - Or: Element-wise logical OR
//! - Not: Element-wise logical NOT

use crate::ir::Attribute;
use crate::CoelanoxResult;
use std::collections::HashMap;

/// Execute element-wise logical AND: output = (input_a != 0.0 && input_b != 0.0) ? 1.0 : 0.0
pub fn and(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] != 0.0 && input_b[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }
    Ok(())
}

/// Execute element-wise logical OR: output = (input_a != 0.0 || input_b != 0.0) ? 1.0 : 0.0
pub fn or(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = if input_a[i] != 0.0 || input_b[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }
    Ok(())
}

/// Execute element-wise logical NOT: output = (input == 0.0) ? 1.0 : 0.0
pub fn not(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = if input[i] == 0.0 { 1.0 } else { 0.0 };
    }
    Ok(())
}
