//! Test for ReduceMean operation
//! 
//! Input: [1.0, 2.0, 3.0, 4.0]
//! Expected: [2.5] (mean: (1+2+3+4)/4)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reducemean() {
    let test = OperationTest::unary(
        OpType::ReduceMean,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![2.5f32],
    );
    
    test.run().expect("ReduceMean test failed");
}

