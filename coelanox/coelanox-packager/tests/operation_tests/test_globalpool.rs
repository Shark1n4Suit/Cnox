//! Tests for GlobalMaxPool and GlobalAvgPool operations
//! 
//! These are essentially the same as ReduceMax and ReduceMean

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_globalmaxpool() {
    let test = OperationTest::unary(
        OpType::GlobalMaxPool,
        vec![1.0f32, 5.0f32, 3.0f32, 2.0f32, 4.0f32],
        vec![5.0f32], // Max of all elements
    );
    test.run().expect("GlobalMaxPool test failed");
}

#[test]
fn test_globalavgpool() {
    let test = OperationTest::unary(
        OpType::GlobalAvgPool,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![2.5f32], // Mean: (1+2+3+4)/4 = 2.5
    );
    test.run().expect("GlobalAvgPool test failed");
}

