//! Test for DIVIDE operation
//! 
//! Input: input_a = [6.0, 8.0, 10.0, 12.0]
//!        input_b = [2.0, 4.0, 5.0, 3.0]
//! Expected: [3.0, 2.0, 2.0, 4.0]

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_divide() {
    let test = OperationTest::binary(
        OpType::Divide,
        vec![6.0f32, 8.0f32, 10.0f32, 12.0f32],
        vec![2.0f32, 4.0f32, 5.0f32, 3.0f32],
        vec![3.0f32, 2.0f32, 2.0f32, 4.0f32],
    );
    
    test.run().expect("DIVIDE test failed");
}

#[test]
fn test_divide_fractional() {
    let test = OperationTest::binary(
        OpType::Divide,
        vec![1.0f32, 3.0f32, 5.0f32],
        vec![2.0f32, 2.0f32, 4.0f32],
        vec![0.5f32, 1.5f32, 1.25f32],
    );
    
    test.run().expect("DIVIDE fractional test failed");
}

