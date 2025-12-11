//! Test for ReduceProd operation
//! 
//! Input: [2.0, 3.0, 4.0]
//! Expected: [24.0] (product of all elements: 2*3*4)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reduceprod() {
    let test = OperationTest::unary(
        OpType::ReduceProd,
        vec![2.0f32, 3.0f32, 4.0f32],
        vec![24.0f32],
    );
    
    test.run().expect("ReduceProd test failed");
}

