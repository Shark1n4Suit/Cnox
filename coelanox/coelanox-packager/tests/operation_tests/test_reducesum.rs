//! Test for ReduceSum operation
//! 
//! Input: [1.0, 2.0, 3.0, 4.0]
//! Expected: [10.0] (sum of all elements)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reducesum() {
    let test = OperationTest::unary(
        OpType::ReduceSum,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![10.0f32],
    );
    
    test.run().expect("ReduceSum test failed");
}

#[test]
fn test_reducesum_negative() {
    let test = OperationTest::unary(
        OpType::ReduceSum,
        vec![5.0f32, -3.0f32, 2.0f32, -1.0f32],
        vec![3.0f32],
    );
    
    test.run().expect("ReduceSum negative test failed");
}

#[test]
fn test_reducesum_single() {
    let test = OperationTest::unary(
        OpType::ReduceSum,
        vec![42.0f32],
        vec![42.0f32],
    );
    
    test.run().expect("ReduceSum single element test failed");
}

