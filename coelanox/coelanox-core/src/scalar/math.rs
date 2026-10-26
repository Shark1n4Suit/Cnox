//! Mathematical primitive operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers.

use crate::ir::Attribute;
use crate::CoelanoxResult;
use rayon::prelude::*;
use std::collections::HashMap;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute absolute value: output[i] = |input[i]|
pub fn abs(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.abs());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.abs());
    }
    Ok(())
}

/// Execute square root: output = sqrt(input)
pub fn sqrt(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.sqrt());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.sqrt());
    }
    Ok(())
}

/// Execute power: output = input^exponent
pub fn pow(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let exponent = attributes
        .get("exponent")
        .and_then(|a| match a {
            crate::ir::Attribute::Float(f) => Some(*f),
            crate::ir::Attribute::Int(i) => Some(*i as f32),
            _ => None,
        })
        .unwrap_or(2.0);
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.powf(exponent));
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.powf(exponent));
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
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.cos());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.cos());
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
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.sin());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.sin());
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
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.exp());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = x.exp());
    }
    Ok(())
}

/// Execute natural logarithm: output = log(input)
pub fn log(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    const EPS: f32 = 1e-8;
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output
            .par_iter_mut()
            .take(len)
            .zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = (x + EPS).ln());
    } else {
        output[..len]
            .iter_mut()
            .zip(input[..len].iter())
            .for_each(|(o, &x)| *o = (x + EPS).ln());
    }
    Ok(())
}
