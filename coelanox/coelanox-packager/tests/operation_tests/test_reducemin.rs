//! Test for ReduceMin operation
//! 
//! Input: [5.0, 1.0, 3.0, 2.0]
//! Expected: [1.0] (minimum of all elements)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reducemin() {
    let test = OperationTest::unary(
        OpType::ReduceMin,
        vec![5.0f32, 1.0f32, 3.0f32, 2.0f32],
        vec![1.0f32],
    );
    
    test.run().expect("ReduceMin test failed");
}

