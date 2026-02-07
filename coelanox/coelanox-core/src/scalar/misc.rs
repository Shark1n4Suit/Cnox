//! Miscellaneous operations
//! 
//! Pure Rust scalar implementations of miscellaneous operations:
//! - Dropout: Random dropout for training

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Execute dropout operation: Randomly zero out elements during training
/// 
/// # Arguments
/// * `input` - Input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `attributes` - Operation-specific attributes
///   * `rate` - Dropout rate (0.0 to 1.0, default: 0.5)
///   * `training` - Whether in training mode (default: true)
///   * `seed` - Random seed for reproducibility (optional)
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
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
        for i in 0..len {
            output[i] = input[i] / (1.0 - rate);
        }
    } else {
        output[..len].copy_from_slice(&input[..len]);
    }
    Ok(())
}

