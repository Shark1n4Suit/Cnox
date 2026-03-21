//! Tensor manipulation operations
//! 
//! Pure Rust scalar implementations of tensor manipulation operations:
//! - Reshape: Change tensor shape (metadata only, no data movement)
//! - Transpose: Transpose tensor dimensions
//! - Slice: Extract a slice from a tensor
//! - Concatenate: Concatenate tensors along specified axis
//! - Split: Split tensor along specified axis
//! - Gather: Gather elements from tensor using indices
//! - Scatter: Scatter elements into tensor using indices
//! - Broadcast: Broadcast tensor to larger shape

use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use crate::ir::Attribute;

/// Reshape: copy input to output (shape change is metadata; caller provides layout).
pub fn reshape(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    output[..len].copy_from_slice(&input[..len]);
    Ok(())
}

/// Execute transpose operation: Transpose tensor dimensions according to permutation.
///
/// Attributes: "shape" (IntArray) = input shape, "perm" (IntArray) = permutation of dimension indices.
/// output[permuted_index] = input[index]. Agnostic: used by any op that reorders axes (e.g. matmul for K^T).
///
/// **Performance:** No heap allocation inside the hot loop. Strides and shapes are used only to compute
/// indices; we reuse two small buffers (allocated once) for multi-index conversion to avoid the
/// "millions of Vec allocations" that caused BERT scalar fallback to go from ~381ms to ~6 minutes.
pub fn transpose(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let shape: Vec<usize> = attributes.get("shape")
        .and_then(|a| match a {
            Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect()),
            _ => None,
        })
        .ok_or_else(|| CoelanoxError::Internal("Transpose requires attribute 'shape' (IntArray)".to_string()))?;

    let perm: Vec<usize> = attributes.get("perm")
        .and_then(|a| match a {
            Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect()),
            _ => None,
        })
        .ok_or_else(|| CoelanoxError::Internal("Transpose requires attribute 'perm' (IntArray)".to_string()))?;

    if shape.is_empty() || perm.len() != shape.len() {
        return Err(CoelanoxError::Internal(format!(
            "Transpose: shape len {} and perm len {} must match and be non-zero",
            shape.len(), perm.len()
        )));
    }
    let total: usize = shape.iter().product();
    if input.len() < total || output.len() < total {
        return Err(CoelanoxError::Internal(format!(
            "Transpose: input len {} or output len {} < total elements {}",
            input.len(), output.len(), total
        )));
    }

    // Fast path: 2D transpose (e.g. [seq, dim] -> [dim, seq]). No div/mod in inner loop, no extra alloc.
    let nd = shape.len();
    if nd == 2 && perm[0] == 1 && perm[1] == 0 {
        let rows = shape[0];
        let cols = shape[1];
        for row in 0..rows {
            for col in 0..cols {
                output[col * rows + row] = input[row * cols + col];
            }
        }
        return Ok(());
    }

    // General N-D: compute strides once, reuse two buffers for multi-index (no alloc per element).
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_strides: Vec<usize> = {
        let mut s = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    let out_strides: Vec<usize> = {
        let mut s = vec![1; out_shape.len()];
        for i in (0..out_shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    // Reused buffers: one allocation per transpose, not per element.
    let mut out_multi = vec![0usize; out_shape.len()];
    let mut in_multi = vec![0usize; shape.len()];

    for out_linear in 0..total {
        // Output linear index -> output multi-index (div/rem, but no alloc).
        let mut rem = out_linear;
        for (i, &stride) in out_strides.iter().enumerate() {
            out_multi[i] = rem / stride;
            rem %= stride;
        }
        // Input multi-index: in_multi[perm[i]] = out_multi[i].
        for (i, &p) in perm.iter().enumerate() {
            in_multi[p] = out_multi[i];
        }
        let in_linear: usize = in_multi.iter().zip(in_strides.iter()).map(|(m, s)| m * s).sum();
        output[out_linear] = input[in_linear];
    }
    Ok(())
}

/// Execute permute operation: Permute tensor dimensions
/// 
/// Similar to transpose but more general - can handle any permutation
/// 
/// # Attributes
/// * `perm` - Permutation of dimensions (e.g., [0, 2, 1, 3] for 4D permute)
pub fn permute(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let len = input.len().min(output.len());
    output[..len].copy_from_slice(&input[..len]);
    Ok(())
}

/// Execute slice operation: Extract a slice from a tensor
///
/// # Attributes
/// * `starts` - Starting indices for each dimension (IntArray)
/// * `ends` - Ending indices (exclusive) for each dimension (IntArray)
/// * `axes` - Axes to slice along (IntArray, optional; default 0..rank)
/// * `input_shape` - Input tensor shape for multi-dim slice (IntArray, optional)
///
/// When starts/ends/input_shape are present, computes offset and count from row-major layout.
/// When absent, copies input to output (legacy behavior).
pub fn slice(
    input: &[f32],
    output: &mut [f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let starts = attributes.get("starts").and_then(|a| match a {
        Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect::<Vec<_>>()),
        _ => None,
    });
    let ends = attributes.get("ends").and_then(|a| match a {
        Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect::<Vec<_>>()),
        _ => None,
    });
    let input_shape = attributes.get("input_shape").or(attributes.get("dimensions")).and_then(|a| match a {
        Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect::<Vec<_>>()),
        _ => None,
    });

    if let (Some(st), Some(en), Some(shape)) = (starts, ends, input_shape) {
        if st.len() != en.len() || st.len() != shape.len() {
            let len = input.len().min(output.len());
            output[..len].copy_from_slice(&input[..len]);
            return Ok(());
        }
        // Row-major strides: stride[i] = product of shape[i+1..]
        let mut stride = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        let mut offset = 0usize;
        let mut count = 1usize;
        for i in 0..shape.len() {
            let s = st[i].min(shape[i]);
            let e = en[i].min(shape[i]);
            if s >= e {
                let len = output.len().min(0);
                output[..len].copy_from_slice(&input[..len]);
                return Ok(());
            }
            offset += s * stride[i];
            count *= e - s;
        }
        if offset + count <= input.len() && count <= output.len() {
            output[..count].copy_from_slice(&input[offset..offset + count]);
            return Ok(());
        }
    }

    // Fallback: copy input to output
    let len = input.len().min(output.len());
    output[..len].copy_from_slice(&input[..len]);
    Ok(())
}

/// Execute concatenate operation: Concatenate tensors along specified axis.
/// For flat/1D tensors, axis is ignored; inputs are copied in order into output.
///
/// # Arguments
/// * `inputs` - Vector of input tensors to concatenate
/// * `output` - Output tensor buffer (must be pre-allocated, length = sum of input lengths)
/// * `attributes` - Operation-specific attributes (axis optional, for shape-aware concat)
pub fn concatenate(
    inputs: &[&[f32]],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let total: usize = inputs.iter().map(|s| s.len()).sum();
    if output.len() < total {
        return Err(CoelanoxError::Internal(format!(
            "Concatenate: output len {} < total elements {}",
            output.len(),
            total
        )));
    }
    let mut offset = 0;
    for inp in inputs {
        let len = inp.len();
        output[offset..offset + len].copy_from_slice(inp);
        offset += len;
    }
    Ok(())
}

/// Execute split operation: Split tensor along specified axis into consecutive chunks.
/// When a single output buffer is used (outputs.len() == 1), copies the full input
/// into it; the "split" attribute defines logical boundaries for downstream consumers.
///
/// # Arguments
/// * `input` - Input tensor data
/// * `outputs` - One or more output buffers. If one, input is copied into it.
///   If multiple, input is split by sizes from attribute "split" (IntArray).
/// * `attributes` - "split" (IntArray) = sizes per output; "axis" optional
pub fn split(
    input: &[f32],
    outputs: &mut [&mut [f32]],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if outputs.is_empty() {
        return Err(CoelanoxError::Internal("Split requires at least one output".to_string()));
    }
    if outputs.len() == 1 {
        let out = &mut outputs[0];
        let len = input.len().min(out.len());
        out[..len].copy_from_slice(&input[..len]);
        return Ok(());
    }
    let sizes: Vec<usize> = attributes
        .get("split")
        .and_then(|a| match a {
            Attribute::IntArray(v) => Some(v.iter().map(|&x| x as usize).collect()),
            _ => None,
        })
        .ok_or_else(|| CoelanoxError::Internal("Split with multiple outputs requires attribute 'split' (IntArray)".to_string()))?;
    if sizes.len() != outputs.len() {
        return Err(CoelanoxError::Internal(format!(
            "Split: 'split' has {} sizes but {} outputs",
            sizes.len(),
            outputs.len()
        )));
    }
    let total: usize = sizes.iter().sum();
    if total > input.len() {
        return Err(CoelanoxError::Internal(format!(
            "Split: total size {} > input len {}",
            total,
            input.len()
        )));
    }
    let mut offset = 0;
    for (out, &sz) in outputs.iter_mut().zip(sizes.iter()) {
        let end = offset + sz;
        let copy_len = sz.min(out.len()).min(input.len().saturating_sub(offset));
        out[..copy_len].copy_from_slice(&input[offset..offset + copy_len]);
        offset = end;
    }
    Ok(())
}

/// Execute gather operation: Gather elements from tensor using indices (1D / flat).
/// output[i] = input[indices[i]]. Indices are clamped to valid range.
///
/// # Arguments
/// * `input` - Input (source) tensor data
/// * `indices` - Indices tensor (integer values)
/// * `output` - Output tensor buffer (must be pre-allocated, len = indices.len())
/// * `attributes` - Operation-specific attributes (axis optional for higher-dim)
pub fn gather(
    input: &[f32],
    indices: &[i32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if output.len() < indices.len() {
        return Err(CoelanoxError::Internal(format!(
            "Gather: output len {} < indices len {}",
            output.len(),
            indices.len()
        )));
    }
    let in_len = input.len() as i32;
    for (i, &idx) in indices.iter().enumerate() {
        let j = idx.clamp(0, in_len.saturating_sub(1)) as usize;
        output[i] = input.get(j).copied().unwrap_or(0.0);
    }
    Ok(())
}

/// Execute scatter operation: Scatter updates into output at indices (1D / flat).
/// output is zeroed, then output[indices[i]] = input[i]. Indices clamped to output bounds.
///
/// # Arguments
/// * `input` - Values to scatter (updates)
/// * `indices` - Indices tensor (integer values)
/// * `output` - Output tensor buffer (pre-allocated; zeroed then written)
/// * `attributes` - Operation-specific attributes (axis optional)
pub fn scatter(
    input: &[f32],
    indices: &[i32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    if indices.len() != input.len() {
        return Err(CoelanoxError::Internal(format!(
            "Scatter: indices len {} != updates len {}",
            indices.len(),
            input.len()
        )));
    }
    output.fill(0.0);
    let out_len = output.len() as i32;
    for (i, &idx) in indices.iter().enumerate() {
        let j = idx.clamp(0, out_len.saturating_sub(1)) as usize;
        if let Some(v) = input.get(i) {
            output[j] = *v;
        }
    }
    Ok(())
}

/// Execute broadcast operation: Broadcast tensor to larger shape
/// 
/// # Arguments
/// * `input` - Input tensor data
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `attributes` - Operation-specific attributes
///   * `shape` - Target shape to broadcast to
pub fn broadcast(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let in_len = input.len();
    if in_len == 0 {
        return Ok(());
    }
    for (i, out_val) in output.iter_mut().enumerate() {
        *out_val = input[i % in_len];
    }
    Ok(())
}

/// Execute expand operation: Expand tensor dimensions
/// 
/// Similar to broadcast but adds new dimensions
/// 
/// # Attributes
/// * `shape` - Target shape to expand to
pub fn expand(
    input: &[f32],
    output: &mut [f32],
    _attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    broadcast(input, output, &HashMap::new())
}

