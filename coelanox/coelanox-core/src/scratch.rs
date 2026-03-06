//! Shape-based peak scratch contract for all ops.
//!
//! Every op declares the maximum scratch elements it needs from its input/output/weight
//! shapes (and optional attributes). Used at package time to compute total dynamic
//! memory and enforce --max-workspace.

use crate::ir::{Attribute, OpType, Shape};

/// Returns the peak number of f32 scratch elements this op needs.
/// Input/output shapes are for the node's data tensors (intermediates); weight_shapes
/// is the list of weight tensor shapes for this node (e.g. Conv has one [C_out, C_in, kH, kW]).
/// Attributes (e.g. kernel_size, stride) can override or disambiguate.
pub fn peak_scratch_elements_for_op(
    op_type: &OpType,
    input_shapes: &[Shape],
    output_shapes: &[Shape],
    weight_shapes: &[Shape],
    attributes: &[(String, Attribute)],
) -> usize {
    let attr_map: std::collections::HashMap<_, _> = attributes.iter().cloned().collect();

    match op_type {
        OpType::Convolution => conv_scratch_from_shapes(input_shapes, output_shapes, weight_shapes, &attr_map),
        OpType::MatMul | OpType::Gemm => matmul_scratch_from_shapes(input_shapes, output_shapes, weight_shapes),
        OpType::Softmax | OpType::LogSoftmax => softmax_scratch_from_shapes(input_shapes, &attr_map),
        OpType::LayerNorm => layernorm_scratch_from_shapes(input_shapes, &attr_map),
        OpType::ReduceSum | OpType::ReduceMean | OpType::ReduceMax | OpType::ReduceMin | OpType::ReduceProd => {
            reduce_scratch_from_shapes(input_shapes, output_shapes)
        }
        OpType::MaxPool | OpType::AvgPool => pool_scratch_from_shapes(input_shapes, output_shapes, &attr_map),
        // Element-wise, no scratch
        OpType::Add | OpType::Subtract | OpType::Multiply | OpType::Divide
        | OpType::Relu | OpType::Sigmoid | OpType::Tanh | OpType::Gelu | OpType::Swish
        | OpType::Sqrt | OpType::Pow | OpType::Cos | OpType::Sin | OpType::Exp | OpType::Log
        | OpType::BatchNorm | OpType::Dropout
        | OpType::Reshape | OpType::Transpose | OpType::Permute
        | OpType::Concatenate | OpType::Split | OpType::Slice | OpType::Gather | OpType::Scatter
        | OpType::Broadcast | OpType::Expand
        | OpType::GlobalMaxPool | OpType::GlobalAvgPool
        | OpType::Equal | OpType::NotEqual | OpType::Greater | OpType::GreaterEqual
        | OpType::Less | OpType::LessEqual | OpType::And | OpType::Or | OpType::Not
        | OpType::Min | OpType::Max
        | OpType::Custom(_) => 0,
    }
}

/// im2col-style scratch: C_in * kH * kW * H_out * W_out (one column per output spatial position).
fn conv_scratch_from_shapes(
    input_shapes: &[Shape],
    output_shapes: &[Shape],
    weight_shapes: &[Shape],
    attr_map: &std::collections::HashMap<String, Attribute>,
) -> usize {
    let in_shape = input_shapes.first().and_then(|s| nchw(s));
    let out_shape = output_shapes.first().and_then(|s| nchw(s));
    let (c_in, k_h, k_w) = if let Some(w) = weight_shapes.first() {
        if w.dimensions.len() == 4 {
            let _c_out = w.dimensions[0];
            let c_in = w.dimensions[1];
            let k_h = w.dimensions[2];
            let k_w = w.dimensions[3];
            (c_in, k_h, k_w)
        } else {
            return 0;
        }
    } else {
        let kernel = attr_map
            .get("kernel_size")
            .and_then(Attribute::as_int_array)
            .filter(|a| a.len() >= 2);
        let k_h = kernel.map(|a| a[0] as usize).unwrap_or(1);
        let k_w = kernel.map(|a| a[1] as usize).unwrap_or(1);
        let c_in = in_shape.map(|(_, c, _, _)| c).unwrap_or(1);
        (c_in, k_h, k_w)
    };
    let (_, _, h_out, w_out) = out_shape.unwrap_or((1, 1, 1, 1));
    c_in * k_h * k_w * h_out * w_out
}

#[inline]
fn nchw(shape: &Shape) -> Option<(usize, usize, usize, usize)> {
    let d = &shape.dimensions[..];
    match d.len() {
        4 => Some((d[0], d[1], d[2], d[3])),
        3 => Some((1, d[0], d[1], d[2])),
        2 => Some((d[0], d[1], 1, 1)),
        1 => Some((1, d[0], 1, 1)),
        _ => None,
    }
}

/// GEMM scratch: often M*K or K*N for transpose; use max(M*K, K*N) as upper bound.
fn matmul_scratch_from_shapes(
    input_shapes: &[Shape],
    output_shapes: &[Shape],
    weight_shapes: &[Shape],
) -> usize {
    let in_shape = input_shapes.first().map(|s| s.dimensions.as_slice()).unwrap_or(&[]);
    let w_shape = weight_shapes.first().map(|s| s.dimensions.as_slice()).unwrap_or(&[]);
    let out_shape = output_shapes.first().map(|s| s.dimensions.as_slice()).unwrap_or(&[]);
    if in_shape.len() >= 2 && w_shape.len() >= 2 && out_shape.len() >= 2 {
        let (m, k_in) = (in_shape[0], in_shape[1]);
        let (k_w, n) = (w_shape[0], w_shape[1]);
        let mk = m * k_in;
        let kn = k_w * n;
        mk.max(kn)
    } else {
        0
    }
}

/// Softmax: temp buffer same size as reduced axis (or full tensor for last-axis).
fn softmax_scratch_from_shapes(
    input_shapes: &[Shape],
    _attr_map: &std::collections::HashMap<String, Attribute>,
) -> usize {
    input_shapes
        .first()
        .map(|s| s.total_elements())
        .unwrap_or(0)
}

/// LayerNorm: normalized_shape determines reduction size; temp for mean/var and normalized.
fn layernorm_scratch_from_shapes(
    input_shapes: &[Shape],
    _attr_map: &std::collections::HashMap<String, Attribute>,
) -> usize {
    input_shapes
        .first()
        .map(|s| s.total_elements())
        .unwrap_or(0)
}

/// Reductions: accumulator over reduced dimensions.
fn reduce_scratch_from_shapes(input_shapes: &[Shape], output_shapes: &[Shape]) -> usize {
    let in_el = input_shapes.first().map(|s| s.total_elements()).unwrap_or(0);
    let out_el = output_shapes.first().map(|s| s.total_elements()).unwrap_or(0);
    in_el.max(out_el)
}

/// Pool: temp for windowed op; use output size * kernel as upper bound.
fn pool_scratch_from_shapes(
    _input_shapes: &[Shape],
    output_shapes: &[Shape],
    attr_map: &std::collections::HashMap<String, Attribute>,
) -> usize {
    let kernel = attr_map
        .get("kernel_size")
        .and_then(Attribute::as_int_array)
        .filter(|a| !a.is_empty());
    let k_prod: usize = kernel
        .map(|a| a.iter().map(|&x| x as usize).product())
        .unwrap_or(1);
    let out_el = output_shapes.first().map(|s| s.total_elements()).unwrap_or(0);
    out_el * k_prod
}
