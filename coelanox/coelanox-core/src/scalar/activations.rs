//! Activation functions
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers
//! (same pattern as MatMul: use Rayon for multi-core when beneficial).

use crate::CoelanoxResult;
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute ReLU activation: output = max(0, input)
pub fn relu(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = if x > 0.0 { x } else { 0.0 });
    } else {
        for i in 0..len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = 1.0 / (1.0 + (-x).exp()));
    } else {
        for i in 0..len {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x.tanh());
    } else {
        for i in 0..len {
            output[i] = input[i].tanh();
        }
    }
    Ok(())
}

/// Execute Softmax activation: output = exp(input - max) / sum(exp(input - max))
/// Optional attribute "axis" (-1 = last) with "shape" (IntArray): softmax over last dimension per row; else whole-tensor.
/// Per-row path parallelizes over rows when num_rows is large.
pub fn softmax(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    if len == 0 {
        return Ok(());
    }

    let shape: Option<Vec<usize>> = attributes.get("shape").and_then(|a| match a {
        crate::ir::Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect()),
        _ => None,
    });
    let axis: i64 = attributes.get("axis").and_then(|a| a.as_int()).unwrap_or(-1);
    let last_dim = shape.as_ref().and_then(|s| s.last().copied());
    let do_per_row = (axis == -1 || axis == 1) && last_dim.map_or(false, |d| d > 0 && len % d == 0);

    if do_per_row {
        let row_len = last_dim.unwrap();
        let num_rows = len / row_len;
        if num_rows >= PAR_THRESHOLD / row_len.max(1) {
            output.par_chunks_mut(row_len)
                .zip(input.par_chunks(row_len))
                .for_each(|(row_out, row_in): (&mut [f32], &[f32])| {
                    let max = row_in.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;
                    for i in 0..row_len {
                        let e = (row_in[i] - max).exp();
                        row_out[i] = e;
                        sum += e;
                    }
                    if sum > 0.0 {
                        for i in 0..row_len {
                            row_out[i] /= sum;
                        }
                    }
                });
        } else {
            for row in 0..num_rows {
                let start = row * row_len;
                let end = start + row_len;
                let row_in = &input[start..end];
                let row_out = &mut output[start..end];
                let max = row_in.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for i in 0..row_len {
                    let e = (row_in[i] - max).exp();
                    row_out[i] = e;
                    sum += e;
                }
                if sum > 0.0 {
                    for i in 0..row_len {
                        row_out[i] /= sum;
                    }
                }
            }
        }
        return Ok(());
    }

    // Whole-tensor softmax (default).
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2)));
    } else {
        for i in 0..len {
            let x = input[i];
            output[i] = 0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2));
        }
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
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x / (1.0 + (-x).exp()));
    } else {
        for i in 0..len {
            let x = input[i];
            output[i] = x / (1.0 + (-x).exp());
        }
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
    let sum: f32 = if len >= PAR_THRESHOLD {
        input.par_iter().take(len).map(|&x| (x - max).exp()).sum()
    } else {
        input[..len].iter().map(|&x| (x - max).exp()).sum()
    };
    let log_sum = sum.ln() + max;
    if len >= PAR_THRESHOLD {
        output.par_iter_mut().take(len).zip(input.par_iter().take(len))
            .for_each(|(o, &x)| *o = x - log_sum);
    } else {
        for i in 0..len {
            output[i] = input[i] - log_sum;
        }
    }
    Ok(())
}

