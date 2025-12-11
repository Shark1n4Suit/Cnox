//! Test for ADD operation
//! 
//! Input: input_a = [1.0, 2.0, 3.0, 4.0]
//!        input_b = [1.0, 1.0, 1.0, 1.0]
//! Expected: [2.0, 3.0, 4.0, 5.0]

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_add() {
    let test = OperationTest::binary(
        OpType::Add,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 1.0f32, 1.0f32, 1.0f32],
        vec![2.0f32, 3.0f32, 4.0f32, 5.0f32],
    );
    
    test.run().expect("ADD test failed");
}

#[test]
fn test_add_negative() {
    let test = OperationTest::binary(
        OpType::Add,
        vec![5.0f32, -3.0f32, 0.0f32],
        vec![2.0f32, 4.0f32, -1.0f32],
        vec![7.0f32, 1.0f32, -1.0f32],
    );
    
    test.run().expect("ADD negative test failed");
}

