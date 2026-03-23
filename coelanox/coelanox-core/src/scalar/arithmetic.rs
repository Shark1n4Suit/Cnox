//! Arithmetic operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers
//! (same pattern as MatMul: use a tuned primitive—here Rayon—for speed).

use crate::CoelanoxResult;
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

/// Threshold above which we use parallel iteration (16K balances parallelism vs rayon overhead).
const PAR_THRESHOLD: usize = 64 * 1024;

#[inline(always)]
fn add_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x + y);
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x + y);
    }
}

/// Execute element-wise addition: output = input_a + input_b
pub fn add(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    add_impl(input_a, input_b, output, len);
    Ok(())
}

#[inline(always)]
fn subtract_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x - y);
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x - y);
    }
}

/// Execute element-wise subtraction: output = input_a - input_b
pub fn subtract(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    subtract_impl(input_a, input_b, output, len);
    Ok(())
}

#[inline(always)]
fn multiply_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x * y);
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x * y);
    }
}

/// Execute element-wise multiplication: output = input_a * input_b
pub fn multiply(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    multiply_impl(input_a, input_b, output, len);
    Ok(())
}

#[inline(always)]
fn divide_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    const EPS: f32 = 1e-8;
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x / (y + EPS));
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x / (y + EPS));
    }
}

/// Execute element-wise division: output = input_a / input_b
pub fn divide(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    divide_impl(input_a, input_b, output, len);
    Ok(())
}

#[inline(always)]
fn min_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x.min(y));
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x.min(y));
    }
}

/// Execute element-wise minimum: output = min(input_a, input_b)
pub fn min(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    min_impl(input_a, input_b, output, len);
    Ok(())
}

#[inline(always)]
fn max_impl(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    if len >= PAR_THRESHOLD {
        out.par_iter_mut()
            .take(len)
            .zip(a.par_iter().take(len))
            .zip(b.par_iter().take(len))
            .for_each(|((o, &x), &y)| *o = x.max(y));
    } else {
        out[..len]
            .iter_mut()
            .zip(a[..len].iter())
            .zip(b[..len].iter())
            .for_each(|((o, &x), &y)| *o = x.max(y));
    }
}

/// Execute element-wise maximum: output = max(input_a, input_b)
pub fn max(
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input_a.len().min(input_b.len()).min(output.len());
    max_impl(input_a, input_b, output, len);
    Ok(())
}
