//! Test for ReduceMax operation
//! 
//! Input: [1.0, 5.0, 3.0, 2.0]
//! Expected: [5.0] (maximum of all elements)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reducemax() {
    let test = OperationTest::unary(
        OpType::ReduceMax,
        vec![1.0f32, 5.0f32, 3.0f32, 2.0f32],
        vec![5.0f32],
    );
    
    test.run().expect("ReduceMax test failed");
}

#[test]
fn test_reducemax_negative() {
    let test = OperationTest::unary(
        OpType::ReduceMax,
        vec![-5.0f32, -1.0f32, -3.0f32],
        vec![-1.0f32],
    );
    
    test.run().expect("ReduceMax negative test failed");
}

#[test]
fn test_reducemax_single() {
    let test = OperationTest::unary(
        OpType::ReduceMax,
        vec![7.5f32],
        vec![7.5f32],
    );
    
    test.run().expect("ReduceMax single element test failed");
}

