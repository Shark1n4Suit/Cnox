//! Tests for MatMul and Gemm operations
//! 
//! These compute dot product of two vectors

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_matmul() {
    // MatMul: dot product of two vectors
    // Input A: [1.0, 2.0, 3.0, 4.0]
    // Input B: [5.0, 6.0, 7.0, 8.0]
    // Output: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70.0
    let test = OperationTest::binary(
        OpType::MatMul,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![5.0f32, 6.0f32, 7.0f32, 8.0f32],
        vec![70.0f32],
    );
    test.run().expect("MatMul test failed");
}

#[test]
fn test_gemm() {
    // Gemm: same as MatMul for now
    // Input A: [2.0, 3.0]
    // Input B: [4.0, 5.0]
    // Output: 2*4 + 3*5 = 8 + 15 = 23.0
    let test = OperationTest::binary(
        OpType::Gemm,
        vec![2.0f32, 3.0f32],
        vec![4.0f32, 5.0f32],
        vec![23.0f32],
    );
    test.run().expect("Gemm test failed");
}

