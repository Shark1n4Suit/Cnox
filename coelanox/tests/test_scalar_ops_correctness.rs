//! Scalar operation correctness tests: known input → expected output.
//!
//! Each test calls the scalar backend (execute_scalar_op) with fixed inputs
//! and asserts the output matches the expected values. This verifies that
//! the scalar implementations produce correct results regardless of container/IR.
//!
//! Coverage: The ISA has 51 non-Custom primitive OpTypes; the scalar backend
//! implements all 51. These tests cover each op callable via execute_scalar_op
//! (including Concatenate, Split, Gather, Scatter which use input_b or attributes).

use coelanox_core::ir::{Attribute, OpType};
use coelanox_core::scalar::execute_scalar_op;
use std::collections::HashMap;

const TOL: f32 = 1e-5;

fn assert_float_eq(a: f32, b: f32, msg: &str) {
    assert!(
        (a - b).abs() < TOL,
        "{}: expected {} ≈ {} (diff {})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

fn assert_slice_eq(actual: &[f32], expected: &[f32], op: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: output length {} != expected {}",
        op,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_float_eq(a, e, &format!("{}[{}]", op, i));
    }
}

fn empty_attrs() -> HashMap<String, Attribute> {
    HashMap::new()
}

// --- Arithmetic ---

#[test]
fn scalar_add_correctness() {
    let a = [1.0f32, 2.0f32, -3.0f32];
    let b = [3.0f32, 4.0f32, 1.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(&OpType::Add, &a, Some(&b), &mut out, None, &empty_attrs())
        .expect("Add should succeed");
    assert_slice_eq(&out, &[4.0, 6.0, -2.0], "Add");
}

#[test]
fn scalar_subtract_correctness() {
    let a = [5.0f32, 3.0f32, 0.0f32];
    let b = [2.0f32, 1.0f32, -1.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Subtract,
        &a,
        Some(&b),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Subtract should succeed");
    assert_slice_eq(&out, &[3.0, 2.0, 1.0], "Subtract");
}

#[test]
fn scalar_multiply_correctness() {
    let a = [2.0f32, 3.0f32, -1.0f32];
    let b = [4.0f32, 5.0f32, -2.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Multiply,
        &a,
        Some(&b),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Multiply should succeed");
    assert_slice_eq(&out, &[8.0, 15.0, 2.0], "Multiply");
}

#[test]
fn scalar_divide_correctness() {
    let a = [10.0f32, 6.0f32, 1.0f32];
    let b = [2.0f32, 3.0f32, 1.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Divide,
        &a,
        Some(&b),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Divide should succeed");
    assert_float_eq(out[0], 5.0, "Divide[0]");
    assert_float_eq(out[1], 2.0, "Divide[1]");
    assert_float_eq(out[2], 1.0, "Divide[2]");
}

// --- Activations ---

#[test]
fn scalar_relu_correctness() {
    let input = [-1.0f32, 0.0f32, 1.0f32, 2.0f32];
    let mut out = [0.0f32; 4];
    execute_scalar_op(&OpType::Relu, &input, None, &mut out, None, &empty_attrs())
        .expect("Relu should succeed");
    assert_slice_eq(&out, &[0.0, 0.0, 1.0, 2.0], "Relu");
}

#[test]
fn scalar_sigmoid_correctness() {
    let input = [0.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(
        &OpType::Sigmoid,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Sigmoid should succeed");
    assert_float_eq(out[0], 0.5, "Sigmoid(0)");
}

#[test]
fn scalar_tanh_correctness() {
    let input = [0.0f32, 1.0f32];
    let mut out = [0.0f32; 2];
    execute_scalar_op(&OpType::Tanh, &input, None, &mut out, None, &empty_attrs())
        .expect("Tanh should succeed");
    assert_float_eq(out[0], 0.0, "Tanh(0)");
    assert_float_eq(out[1], 1.0f32.tanh(), "Tanh(1)");
}

#[test]
fn scalar_softmax_correctness() {
    let input = [1.0f32, 1.0f32, 1.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Softmax,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Softmax should succeed");
    let third = 1.0 / 3.0;
    assert_slice_eq(&out, &[third, third, third], "Softmax");
}

#[test]
fn scalar_log_softmax_correctness() {
    // log_softmax: output[i] = input[i] - log(sum(exp(input - max)))
    // For [0,0,0]: max=0, sum=3, log_sum=ln(3), output = 0 - ln(3) = -ln(3)
    let input = [0.0f32, 0.0f32, 0.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::LogSoftmax,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("LogSoftmax should succeed");
    let expected = -(3.0f32.ln());
    for (i, &v) in out.iter().enumerate() {
        assert_float_eq(v, expected, &format!("LogSoftmax[{}]", i));
    }
}

#[test]
fn scalar_gelu_correctness() {
    // GELU(0) = 0
    let input = [0.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(&OpType::Gelu, &input, None, &mut out, None, &empty_attrs())
        .expect("Gelu should succeed");
    assert_float_eq(out[0], 0.0, "Gelu(0)");
}

#[test]
fn scalar_swish_correctness() {
    // Swish(0) = 0
    let input = [0.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(&OpType::Swish, &input, None, &mut out, None, &empty_attrs())
        .expect("Swish should succeed");
    assert_float_eq(out[0], 0.0, "Swish(0)");
}

// --- Math ---

#[test]
fn scalar_sqrt_correctness() {
    let input = [4.0f32, 9.0f32, 0.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(&OpType::Sqrt, &input, None, &mut out, None, &empty_attrs())
        .expect("Sqrt should succeed");
    assert_slice_eq(&out, &[2.0, 3.0, 0.0], "Sqrt");
}

#[test]
fn scalar_abs_correctness() {
    let input = [2.0f32, -3.0f32, 0.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(&OpType::Abs, &input, None, &mut out, None, &empty_attrs())
        .expect("Abs should succeed");
    assert_slice_eq(&out, &[2.0, 3.0, 0.0], "Abs");
}

#[test]
fn scalar_pow_correctness() {
    let input = [2.0f32, 3.0f32];
    let mut out = [0.0f32; 2];
    let mut attrs = empty_attrs();
    attrs.insert("exponent".to_string(), Attribute::Float(2.0));
    execute_scalar_op(&OpType::Pow, &input, None, &mut out, None, &attrs)
        .expect("Pow should succeed");
    assert_slice_eq(&out, &[4.0, 9.0], "Pow(x^2)");
}

#[test]
fn scalar_cos_sin_correctness() {
    let input = [0.0f32, std::f32::consts::FRAC_PI_2];
    let mut out_cos = [0.0f32; 2];
    let mut out_sin = [0.0f32; 2];
    execute_scalar_op(
        &OpType::Cos,
        &input,
        None,
        &mut out_cos,
        None,
        &empty_attrs(),
    )
    .expect("Cos should succeed");
    execute_scalar_op(
        &OpType::Sin,
        &input,
        None,
        &mut out_sin,
        None,
        &empty_attrs(),
    )
    .expect("Sin should succeed");
    assert_float_eq(out_cos[0], 1.0, "Cos(0)");
    assert_float_eq(out_cos[1], 0.0, "Cos(π/2)");
    assert_float_eq(out_sin[0], 0.0, "Sin(0)");
    assert_float_eq(out_sin[1], 1.0, "Sin(π/2)");
}

#[test]
fn scalar_exp_log_correctness() {
    let input_exp = [0.0f32, 1.0f32];
    let mut out_exp = [0.0f32; 2];
    execute_scalar_op(
        &OpType::Exp,
        &input_exp,
        None,
        &mut out_exp,
        None,
        &empty_attrs(),
    )
    .expect("Exp should succeed");
    assert_float_eq(out_exp[0], 1.0, "Exp(0)");
    assert_float_eq(out_exp[1], std::f32::consts::E, "Exp(1)");

    let input_log = [1.0f32, std::f32::consts::E];
    let mut out_log = [0.0f32; 2];
    execute_scalar_op(
        &OpType::Log,
        &input_log,
        None,
        &mut out_log,
        None,
        &empty_attrs(),
    )
    .expect("Log should succeed");
    assert_float_eq(out_log[0], 0.0, "Log(1)");
    assert_float_eq(out_log[1], 1.0, "Log(e)");
}

// --- Reductions ---

#[test]
fn scalar_reduce_sum_correctness() {
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(
        &OpType::ReduceSum,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("ReduceSum should succeed");
    assert_float_eq(out[0], 6.0, "ReduceSum");
}

#[test]
fn scalar_reduce_mean_correctness() {
    let input = [2.0f32, 4.0f32, 6.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(
        &OpType::ReduceMean,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("ReduceMean should succeed");
    assert_float_eq(out[0], 4.0, "ReduceMean");
}

#[test]
fn scalar_reduce_max_min_correctness() {
    let input = [1.0f32, 5.0f32, 3.0f32, -2.0f32];
    let mut out_max = [0.0f32; 1];
    let mut out_min = [0.0f32; 1];
    execute_scalar_op(
        &OpType::ReduceMax,
        &input,
        None,
        &mut out_max,
        None,
        &empty_attrs(),
    )
    .expect("ReduceMax should succeed");
    execute_scalar_op(
        &OpType::ReduceMin,
        &input,
        None,
        &mut out_min,
        None,
        &empty_attrs(),
    )
    .expect("ReduceMin should succeed");
    assert_float_eq(out_max[0], 5.0, "ReduceMax");
    assert_float_eq(out_min[0], -2.0, "ReduceMin");
}

#[test]
fn scalar_reduce_prod_correctness() {
    let input = [2.0f32, 3.0f32, 4.0f32];
    let mut out = [0.0f32; 1];
    execute_scalar_op(
        &OpType::ReduceProd,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("ReduceProd should succeed");
    assert_float_eq(out[0], 24.0, "ReduceProd");
}

// --- Comparisons ---

#[test]
fn scalar_equal_not_equal_correctness() {
    let a = [1.0f32, 2.0f32, 3.0f32];
    let b = [1.0f32, 2.1f32, 3.0f32];
    let mut out_eq = [0.0f32; 3];
    let mut out_ne = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Equal,
        &a,
        Some(&b),
        &mut out_eq,
        None,
        &empty_attrs(),
    )
    .expect("Equal should succeed");
    execute_scalar_op(
        &OpType::NotEqual,
        &a,
        Some(&b),
        &mut out_ne,
        None,
        &empty_attrs(),
    )
    .expect("NotEqual should succeed");
    assert_slice_eq(&out_eq, &[1.0, 0.0, 1.0], "Equal");
    assert_slice_eq(&out_ne, &[0.0, 1.0, 0.0], "NotEqual");
}

#[test]
fn scalar_greater_less_correctness() {
    let a = [3.0f32, 1.0f32, 2.0f32];
    let b = [2.0f32, 2.0f32, 2.0f32];
    let mut out_gt = [0.0f32; 3];
    let mut out_ge = [0.0f32; 3];
    let mut out_lt = [0.0f32; 3];
    let mut out_le = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Greater,
        &a,
        Some(&b),
        &mut out_gt,
        None,
        &empty_attrs(),
    )
    .expect("Greater should succeed");
    execute_scalar_op(
        &OpType::GreaterEqual,
        &a,
        Some(&b),
        &mut out_ge,
        None,
        &empty_attrs(),
    )
    .expect("GreaterEqual should succeed");
    execute_scalar_op(
        &OpType::Less,
        &a,
        Some(&b),
        &mut out_lt,
        None,
        &empty_attrs(),
    )
    .expect("Less should succeed");
    execute_scalar_op(
        &OpType::LessEqual,
        &a,
        Some(&b),
        &mut out_le,
        None,
        &empty_attrs(),
    )
    .expect("LessEqual should succeed");
    assert_slice_eq(&out_gt, &[1.0, 0.0, 0.0], "Greater");
    assert_slice_eq(&out_ge, &[1.0, 0.0, 1.0], "GreaterEqual");
    assert_slice_eq(&out_lt, &[0.0, 1.0, 0.0], "Less");
    assert_slice_eq(&out_le, &[0.0, 1.0, 1.0], "LessEqual");
}

// --- Logical ---

#[test]
fn scalar_and_or_not_correctness() {
    let a = [1.0f32, 0.0f32, 1.0f32];
    let b = [1.0f32, 1.0f32, 0.0f32];
    let mut out_and = [0.0f32; 3];
    let mut out_or = [0.0f32; 3];
    execute_scalar_op(
        &OpType::And,
        &a,
        Some(&b),
        &mut out_and,
        None,
        &empty_attrs(),
    )
    .expect("And should succeed");
    execute_scalar_op(&OpType::Or, &a, Some(&b), &mut out_or, None, &empty_attrs())
        .expect("Or should succeed");
    assert_slice_eq(&out_and, &[1.0, 0.0, 0.0], "And");
    assert_slice_eq(&out_or, &[1.0, 1.0, 1.0], "Or");

    let input_not = [0.0f32, 1.0f32, -1.0f32];
    let mut out_not = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Not,
        &input_not,
        None,
        &mut out_not,
        None,
        &empty_attrs(),
    )
    .expect("Not should succeed");
    assert_slice_eq(&out_not, &[1.0, 0.0, 0.0], "Not");
}

// --- Pooling ---

#[test]
fn scalar_maxpool_avgpool_correctness() {
    // kernel_size=2, stride=2: [1,2,3,4] -> 2 windows -> max [2,4], avg [1.5, 3.5]
    let input = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let mut out_max = [0.0f32; 2];
    let mut out_avg = [0.0f32; 2];
    execute_scalar_op(
        &OpType::MaxPool,
        &input,
        None,
        &mut out_max,
        None,
        &empty_attrs(),
    )
    .expect("MaxPool should succeed");
    execute_scalar_op(
        &OpType::AvgPool,
        &input,
        None,
        &mut out_avg,
        None,
        &empty_attrs(),
    )
    .expect("AvgPool should succeed");
    assert_slice_eq(&out_max, &[2.0, 4.0], "MaxPool");
    assert_slice_eq(&out_avg, &[1.5, 3.5], "AvgPool");
}

#[test]
fn scalar_global_maxpool_avgpool_correctness() {
    let input = [1.0f32, 5.0f32, 3.0f32];
    let mut out_max = [0.0f32; 1];
    let mut out_avg = [0.0f32; 1];
    execute_scalar_op(
        &OpType::GlobalMaxPool,
        &input,
        None,
        &mut out_max,
        None,
        &empty_attrs(),
    )
    .expect("GlobalMaxPool should succeed");
    execute_scalar_op(
        &OpType::GlobalAvgPool,
        &input,
        None,
        &mut out_avg,
        None,
        &empty_attrs(),
    )
    .expect("GlobalAvgPool should succeed");
    assert_float_eq(out_max[0], 5.0, "GlobalMaxPool");
    assert_float_eq(out_avg[0], 3.0, "GlobalAvgPool"); // (1+5+3)/3
}

// --- Dropout ---

#[test]
fn scalar_dropout_correctness() {
    // training=false: output = input
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 3];
    let mut attrs = empty_attrs();
    attrs.insert("training".to_string(), Attribute::Int(0));
    execute_scalar_op(&OpType::Dropout, &input, None, &mut out, None, &attrs)
        .expect("Dropout should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Dropout(training=false)");
}

// --- Normalization (with minimal weights) ---

#[test]
fn scalar_batchnorm_correctness() {
    // input [2,4,6]: mean=4, var = (4+0+4)/3 + eps, normalized then gamma* x + beta
    // gamma=1, beta=0 -> output is (x-4)/std
    let input = [2.0f32, 4.0f32, 6.0f32];
    let weights = [1.0f32, 0.0f32]; // gamma, beta
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::BatchNorm,
        &input,
        None,
        &mut out,
        Some(&weights),
        &empty_attrs(),
    )
    .expect("BatchNorm should succeed");
    let mean = 4.0f32;
    let var = (2.0f32 * 2.0f32 + 0.0 + 2.0f32 * 2.0f32) / 3.0f32 + 1e-5;
    let std_inv = 1.0 / var.sqrt();
    assert_float_eq(out[0], (2.0 - mean) * std_inv, "BatchNorm[0]");
    assert_float_eq(out[1], (4.0 - mean) * std_inv, "BatchNorm[1]");
    assert_float_eq(out[2], (6.0 - mean) * std_inv, "BatchNorm[2]");
}

#[test]
fn scalar_layernorm_correctness() {
    // LayerNorm normalizes over last dim H; weights = [gamma (H), beta (H)]. Use H=3 for one row of 3.
    let input = [1.0f32, 2.0f32, 3.0f32];
    let weights = [1.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32]; // gamma=[1,1,1], beta=[0,0,0]
    let mut out = [0.0f32; 3];
    let mut attrs = empty_attrs();
    attrs.insert("normalized_shape".to_string(), Attribute::IntArray(vec![3]));
    execute_scalar_op(
        &OpType::LayerNorm,
        &input,
        None,
        &mut out,
        Some(&weights),
        &attrs,
    )
    .expect("LayerNorm should succeed");
    let mean = 2.0f32;
    let var = (1.0f32 + 0.0 + 1.0f32) / 3.0f32 + 1e-5;
    let std_inv = 1.0 / var.sqrt();
    assert_float_eq(out[0], (1.0 - mean) * std_inv, "LayerNorm[0]");
    assert_float_eq(out[1], (2.0 - mean) * std_inv, "LayerNorm[1]");
    assert_float_eq(out[2], (3.0 - mean) * std_inv, "LayerNorm[2]");
}

// --- Matrix (minimal shapes) ---

#[test]
fn scalar_matmul_correctness() {
    // 1x2 @ 2x1 -> 1x1: input [a,b], weights (K,N)=(2,1) = [w0, w1], output = a*w0 + b*w1
    let input = [1.0f32, 2.0f32];
    let weights = [1.0f32, 0.0f32]; // 2x1
    let mut out = [0.0f32; 1];
    execute_scalar_op(
        &OpType::MatMul,
        &input,
        None,
        &mut out,
        Some(&weights),
        &empty_attrs(),
    )
    .expect("MatMul should succeed");
    assert_float_eq(out[0], 1.0 * 1.0 + 2.0 * 0.0, "MatMul");
}

#[test]
fn scalar_gemm_correctness() {
    // Same as MatMul then * alpha (default 1.0)
    let input = [1.0f32, 2.0f32];
    let weights = [1.0f32, 1.0f32];
    let mut out = [0.0f32; 1];
    let mut attrs = empty_attrs();
    attrs.insert("alpha".to_string(), Attribute::Float(2.0));
    execute_scalar_op(
        &OpType::Gemm,
        &input,
        None,
        &mut out,
        Some(&weights),
        &attrs,
    )
    .expect("Gemm should succeed");
    assert_float_eq(out[0], 2.0 * (1.0 * 1.0 + 2.0 * 1.0), "Gemm"); // alpha * (1+2) = 6
}

// --- Convolution (1D, kernel_size=3) ---

#[test]
fn scalar_convolution_correctness() {
    // input len 5, kernel 3 -> output len 3. weights [1,0,0]: out[i] = input[i]
    let input = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
    let weights = [1.0f32, 0.0f32, 0.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Convolution,
        &input,
        None,
        &mut out,
        Some(&weights),
        &empty_attrs(),
    )
    .expect("Convolution should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Convolution");
}

// --- Tensor ops (copy / layout) ---

#[test]
fn scalar_reshape_correctness() {
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Reshape,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Reshape should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Reshape");
}

#[test]
fn scalar_transpose_correctness() {
    // Transpose requires "shape" and "perm" attributes. Identity perm [0] for 1D leaves order unchanged.
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 3];
    let mut attrs = empty_attrs();
    attrs.insert("shape".to_string(), Attribute::IntArray(vec![3]));
    attrs.insert("perm".to_string(), Attribute::IntArray(vec![0]));
    execute_scalar_op(&OpType::Transpose, &input, None, &mut out, None, &attrs)
        .expect("Transpose should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Transpose");
}

#[test]
fn scalar_permute_correctness() {
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Permute,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Permute should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Permute");
}

#[test]
fn scalar_slice_correctness() {
    let input = [1.0f32, 2.0f32, 3.0f32];
    let mut out = [0.0f32; 3];
    execute_scalar_op(&OpType::Slice, &input, None, &mut out, None, &empty_attrs())
        .expect("Slice should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0], "Slice");
}

#[test]
fn scalar_broadcast_expand_correctness() {
    // broadcast: output[i] = input[i % input.len()]
    let input = [1.0f32, 2.0f32];
    let mut out = [0.0f32; 4];
    execute_scalar_op(
        &OpType::Broadcast,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Broadcast should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 1.0, 2.0], "Broadcast");
    out = [0.0f32; 4];
    execute_scalar_op(
        &OpType::Expand,
        &input,
        None,
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Expand should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 1.0, 2.0], "Expand");
}

#[test]
fn scalar_concatenate_correctness() {
    let a = [1.0f32, 2.0f32, 3.0f32];
    let b = [4.0f32, 5.0f32];
    let mut out = [0.0f32; 5];
    execute_scalar_op(
        &OpType::Concatenate,
        &a,
        Some(&b),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Concatenate should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0, 4.0, 5.0], "Concatenate");
}

#[test]
fn scalar_split_correctness() {
    let input = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let mut out = [0.0f32; 4];
    execute_scalar_op(&OpType::Split, &input, None, &mut out, None, &empty_attrs())
        .expect("Split should succeed");
    assert_slice_eq(&out, &[1.0, 2.0, 3.0, 4.0], "Split");
}

#[test]
fn scalar_gather_correctness() {
    let data = [10.0f32, 20.0f32, 30.0f32, 40.0f32];
    let indices = [0.0f32, 2.0f32, 1.0f32]; // interpreted as i32: 0, 2, 1
    let mut out = [0.0f32; 3];
    execute_scalar_op(
        &OpType::Gather,
        &data,
        Some(&indices),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Gather should succeed");
    assert_slice_eq(&out, &[10.0, 30.0, 20.0], "Gather");
}

#[test]
fn scalar_scatter_correctness() {
    let updates = [100.0f32, 200.0f32];
    let indices = [1.0f32, 3.0f32]; // interpreted as i32: 1, 3
    let mut out = [0.0f32; 4];
    execute_scalar_op(
        &OpType::Scatter,
        &updates,
        Some(&indices),
        &mut out,
        None,
        &empty_attrs(),
    )
    .expect("Scatter should succeed");
    assert_slice_eq(&out, &[0.0, 100.0, 0.0, 200.0], "Scatter");
}
