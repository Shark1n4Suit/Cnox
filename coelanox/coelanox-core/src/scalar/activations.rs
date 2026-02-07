//! Activation functions
//! 
//! Pure Rust scalar implementations of activation functions:
//! - ReLU: Rectified Linear Unit
//! - Sigmoid: Sigmoid activation
//! - Tanh: Hyperbolic tangent
//! - Softmax: Softmax activation
//! - GELU: Gaussian Error Linear Unit
//! - Swish: Swish activation

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute ReLU activation: output = max(0, input)
pub fn relu(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
    }
    Ok(())
}

/// Execute Sigmoid activation: output = 1 / (1 + exp(-input))
pub fn sigmoid(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
    Ok(())
}

/// Execute Tanh activation: output = tanh(input)
pub fn tanh(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].tanh();
    }
    Ok(())
}

/// Execute Softmax activation: output = exp(input - max) / sum(exp(input - max))
pub fn softmax(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len == 0 {
        return Ok(());
    }
    let max = input[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for i in 0..len {
        let e = (input[i] - max).exp();
        output[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for i in 0..len {
            output[i] /= sum;
        }
    }
    Ok(())
}

/// Execute GELU activation: output = x * 0.5 * (1 + erf(x / sqrt(2)))
pub fn gelu(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        let x = input[i];
        output[i] = 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2));
    }
    Ok(())
}

/// Approximate erf for GELU (Abramowitz & Stegun approximation)
fn erf(x: f32) -> f32 {
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;

    let p = 1.0 / (1.0 + 0.3275911 * x.abs());

    let t = 1.0 - (((((a5 * p + a4) * p) + a3) * p + a2) * p + a1) * p * (-x * x).exp();
    if x >= 0.0 {
        t
    } else {
        -t
    }
}

/// Execute Swish activation: output = x * sigmoid(x)
pub fn swish(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    for i in 0..len {
        let x = input[i];
        output[i] = x / (1.0 + (-x).exp());
    }
    Ok(())
}

/// Execute LogSoftmax activation: output = log(softmax(input))
pub fn log_softmax(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len == 0 {
        return Ok(());
    }
    let max = input[..len].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += (input[i] - max).exp();
    }
    let log_sum = sum.ln() + max;
    for i in 0..len {
        output[i] = input[i] - log_sum;
    }
    Ok(())
}

