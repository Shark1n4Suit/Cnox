//! Test for SUBTRACT operation
//! 
//! Input: input_a = [5.0, 4.0, 3.0, 2.0]
//!        input_b = [1.0, 2.0, 1.0, 1.0]
//! Expected: [4.0, 2.0, 2.0, 1.0]

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_subtract() {
    let test = OperationTest::binary(
        OpType::Subtract,
        vec![5.0f32, 4.0f32, 3.0f32, 2.0f32],
        vec![1.0f32, 2.0f32, 1.0f32, 1.0f32],
        vec![4.0f32, 2.0f32, 2.0f32, 1.0f32],
    );
    
    test.run().expect("SUBTRACT test failed");
}

#[test]
fn test_subtract_negative() {
    let test = OperationTest::binary(
        OpType::Subtract,
        vec![3.0f32, -2.0f32, 0.0f32],
        vec![5.0f32, -1.0f32, 2.0f32],
        vec![-2.0f32, -1.0f32, -2.0f32],
    );
    
    test.run().expect("SUBTRACT negative test failed");
}

