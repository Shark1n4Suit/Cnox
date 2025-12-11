//! Tests for tensor manipulation operations
//! 
//! For 1D tensors, these operations are effectively copies

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_transpose() {
    // Transpose: for 1D, just copy
    let test = OperationTest::unary(
        OpType::Transpose,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Transpose test failed");
}

#[test]
fn test_permute() {
    // Permute: for 1D, just copy
    let test = OperationTest::unary(
        OpType::Permute,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Permute test failed");
}

#[test]
fn test_concatenate() {
    // Concatenate: for 1D single input, just copy
    let test = OperationTest::unary(
        OpType::Concatenate,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Concatenate test failed");
}

#[test]
fn test_split() {
    // Split: for 1D single output, just copy
    let test = OperationTest::unary(
        OpType::Split,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Split test failed");
}

#[test]
fn test_slice() {
    // Slice: for 1D full slice, just copy
    let test = OperationTest::unary(
        OpType::Slice,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Slice test failed");
}

#[test]
fn test_gather() {
    // Gather: for 1D without indices, just copy
    let test = OperationTest::unary(
        OpType::Gather,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Gather test failed");
}

#[test]
fn test_scatter() {
    // Scatter: for 1D without indices, just copy
    let test = OperationTest::unary(
        OpType::Scatter,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Scatter test failed");
}

