//! Miscellaneous operations
//!
//! Pure Rust scalar implementations with parallel iteration for large buffers.

use crate::CoelanoxResult;
use std::collections::HashMap;
use crate::ir::Attribute;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 64 * 1024;

/// Execute dropout operation: scale or copy input to output
pub fn dropout(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let training = attributes.get("training")
        .and_then(|a| match a {
            crate::ir::Attribute::Int(i) => Some(*i != 0),
            _ => None,
        })
        .unwrap_or(false);
    let len = input.len().min(output.len());
    if training {
        let rate = attributes.get("rate")
            .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
            .unwrap_or(0.5);
        let scale = 1.0 / (1.0 - rate);
        if len >= PAR_THRESHOLD {
            output.par_iter_mut().take(len).zip(input.par_iter().take(len))
                .for_each(|(o, &x)| *o = x * scale);
        } else {
            for i in 0..len {
                output[i] = input[i] / (1.0 - rate);
            }
        }
    } else {
        output[..len].copy_from_slice(&input[..len]);
    }
    Ok(())
}

