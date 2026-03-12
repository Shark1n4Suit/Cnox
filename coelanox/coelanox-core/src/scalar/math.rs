//! Mathematical primitive operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers.

use crate::CoelanoxResult;
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute square root: output = sqrt(input)
pub fn sqrt(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.sqrt());
    } else {
        for i in 0..len {
            output[i] = input[i].sqrt();
        }
    }
    Ok(())
}

/// Execute power: output = input^exponent
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.powf(exponent));
    } else {
        for i in 0..len {
            output[i] = input[i].powf(exponent);
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.cos());
    } else {
        for i in 0..len {
            output[i] = input[i].cos();
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.sin());
    } else {
        for i in 0..len {
            output[i] = input[i].sin();
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.exp());
    } else {
        for i in 0..len {
            output[i] = input[i].exp();
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = (x + f32::EPSILON).ln());
    } else {
        for i in 0..len {
            output[i] = (input[i] + f32::EPSILON).ln();
        }
    }
    Ok(())
}

