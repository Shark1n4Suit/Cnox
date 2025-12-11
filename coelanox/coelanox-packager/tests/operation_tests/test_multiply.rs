//! Test for MULTIPLY operation
//! 
//! Input: input_a = [2.0, 3.0, 4.0, 5.0]
//!        input_b = [1.5, 2.0, 0.5, 2.0]
//! Expected: [3.0, 6.0, 2.0, 10.0]

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_multiply() {
    let test = OperationTest::binary(
        OpType::Multiply,
        vec![2.0f32, 3.0f32, 4.0f32, 5.0f32],
        vec![1.5f32, 2.0f32, 0.5f32, 2.0f32],
        vec![3.0f32, 6.0f32, 2.0f32, 10.0f32],
    );
    
    test.run().expect("MULTIPLY test failed");
}

#[test]
fn test_multiply_zero() {
    let test = OperationTest::binary(
        OpType::Multiply,
        vec![5.0f32, 3.0f32, 2.0f32],
        vec![0.0f32, 1.0f32, 0.0f32],
        vec![0.0f32, 3.0f32, 0.0f32],
    );
    
    test.run().expect("MULTIPLY zero test failed");
}

