//! Mathematical primitive operations
//! 
//! Pure Rust scalar implementations of mathematical primitives:
//! - Sqrt: Square root
//! - Pow: Power (x^y)
//! - Cos: Cosine
//! - Sin: Sine
//! - Exp: Exponential (e^x)
//! - Log: Natural logarithm

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute square root: output = sqrt(input)
pub fn sqrt(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].sqrt();
    }
    Ok(())
}

/// Execute power: output = input^exponent
/// 
/// # Attributes
/// * `exponent` - Power to raise input to (default: 2.0)
pub fn pow(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let exponent = attributes.get("exponent")
        .and_then(|a| match a {
            crate::ir::Attribute::Float(f) => Some(*f),
            crate::ir::Attribute::Int(i) => Some(*i as f32),
            _ => None,
        })
        .unwrap_or(2.0);
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].powf(exponent);
    }
    Ok(())
}

/// Execute cosine: output = cos(input)
pub fn cos(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].cos();
    }
    Ok(())
}

/// Execute sine: output = sin(input)
pub fn sin(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].sin();
    }
    Ok(())
}

/// Execute exponential: output = exp(input)
pub fn exp(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].exp();
    }
    Ok(())
}

/// Execute natural logarithm: output = log(input)
pub fn log(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = (input[i] + f32::EPSILON).ln();
    }
    Ok(())
}

