//! Test for ReLU operation
//! 
//! Input: [1.0, -1.0, 0.0, 2.5, -3.0]
//! Expected: [1.0, 0.0, 0.0, 2.5, 0.0] (max(0, x))

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_relu() {
    let test = OperationTest::unary(
        OpType::Relu,
        vec![1.0f32, -1.0f32, 0.0f32, 2.5f32, -3.0f32],
        vec![1.0f32, 0.0f32, 0.0f32, 2.5f32, 0.0f32],
    );
    
    test.run().expect("ReLU test failed");
}

#[test]
fn test_relu_all_positive() {
    let test = OperationTest::unary(
        OpType::Relu,
        vec![1.0f32, 2.0f32, 3.0f32],
        vec![1.0f32, 2.0f32, 3.0f32],
    );
    
    test.run().expect("ReLU all positive test failed");
}

#[test]
fn test_relu_all_negative() {
    let test = OperationTest::unary(
        OpType::Relu,
        vec![-1.0f32, -2.0f32, -3.0f32],
        vec![0.0f32, 0.0f32, 0.0f32],
    );
    
    test.run().expect("ReLU all negative test failed");
}

