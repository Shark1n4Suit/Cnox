//! Arithmetic operations
//! 
//! Pure Rust scalar implementations of basic arithmetic operations:
//! - Add: Element-wise addition
//! - Subtract: Element-wise subtraction
//! - Multiply: Element-wise multiplication
//! - Divide: Element-wise division

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute element-wise addition: output = input_a + input_b
/// 
/// This is the authoritative scalar implementation used for:
/// - Fallback backend execution
/// - Vendor backend verification
/// - Golden reference for correctness
/// 
/// # Arguments
/// * `input_a` - First input tensor data
/// * `input_b` - Second input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `attributes` - Operation-specific attributes (currently unused)
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
pub fn add(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = input_a[i] + input_b[i];
    }
    Ok(())
}

/// Execute element-wise subtraction: output = input_a - input_b
pub fn subtract(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = input_a[i] - input_b[i];
    }
    Ok(())
}

/// Execute element-wise multiplication: output = input_a * input_b
pub fn multiply(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = input_a[i] * input_b[i];
    }
    Ok(())
}

/// Execute element-wise division: output = input_a / input_b
pub fn divide(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    for i in 0..len {
        output[i] = input_a[i] / (input_b[i] + f32::EPSILON);
    }
    Ok(())
}

