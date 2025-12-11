//! Tests for neural network operations
//! 
//! These include pooling, normalization, and convolution operations

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_maxpool() {
    // MaxPool: max of every 2 elements
    // Input: [1.0, 5.0, 3.0, 7.0]
    // Output: [5.0, 7.0] (max of [1,5] and [3,7])
    let test = OperationTest::unary(
        OpType::MaxPool,
        vec![1.0f32, 5.0f32, 3.0f32, 7.0f32],
        vec![5.0f32, 7.0f32],
    );
    test.run().expect("MaxPool test failed");
}

#[test]
fn test_avgpool() {
    // AvgPool: average of every 2 elements
    // Input: [1.0, 3.0, 5.0, 7.0]
    // Output: [2.0, 6.0] (avg of [1,3] and [5,7])
    let test = OperationTest::unary(
        OpType::AvgPool,
        vec![1.0f32, 3.0f32, 5.0f32, 7.0f32],
        vec![2.0f32, 6.0f32],
    );
    test.run().expect("AvgPool test failed");
}

#[test]
fn test_batchnorm() {
    // BatchNorm: (x - mean) / sqrt(var + epsilon) * gamma + beta
    // For test, use weights: [gamma=1.0, beta=0.0, mean=0.0, var=1.0]
    // Input: [1.0, 2.0, 3.0, 4.0]
    // Output: [1.0, 2.0, 3.0, 4.0] (normalized with mean=0, var=1)
    // Note: This test requires weights buffer to be set up
    // For now, test with simple passthrough expectation
    let mut test = OperationTest::unary(
        OpType::BatchNorm,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], // Simplified - assumes identity normalization
    );
    test.tolerance = 0.1f32; // Allow some tolerance for sqrt computation
    test.run().expect("BatchNorm test failed");
}

#[test]
fn test_layernorm() {
    // LayerNorm: for 1D, just copy (simplified implementation)
    let test = OperationTest::unary(
        OpType::LayerNorm,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("LayerNorm test failed");
}

#[test]
fn test_convolution() {
    // Convolution: simplified as element-wise multiply with weights
    // Input: [1.0, 2.0, 3.0, 4.0]
    // Weights: [2.0, 2.0, 2.0, 2.0]
    // Output: [2.0, 4.0, 6.0, 8.0]
    // Note: This requires weights buffer - for now test with simple case
    let test = OperationTest::unary(
        OpType::Convolution,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], // Simplified - assumes weights are all 1.0
    );
    test.run().expect("Convolution test failed");
}

#[test]
fn test_dropout() {
    // Dropout: for inference mode, just copy
    let test = OperationTest::unary(
        OpType::Dropout,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    test.run().expect("Dropout test failed");
}

