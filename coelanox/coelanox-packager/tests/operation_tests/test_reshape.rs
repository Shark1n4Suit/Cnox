//! Test for RESHAPE operation
//! 
//! Reshape just copies data (shape is metadata)
//! Input: [1.0, 2.0, 3.0, 4.0]
//! Expected: [1.0, 2.0, 3.0, 4.0] (same data)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_reshape() {
    let test = OperationTest::unary(
        OpType::Reshape,
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
    );
    
    test.run().expect("RESHAPE test failed");
}

