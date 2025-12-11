//! Tests for Broadcast and Expand operations
//! 
//! These replicate a single value to multiple output positions

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_broadcast() {
    // Broadcast single value to multiple outputs
    let test = OperationTest::unary(
        OpType::Broadcast,
        vec![2.5f32], // Single input value
        vec![2.5f32, 2.5f32, 2.5f32, 2.5f32], // Replicated to output
    );
    test.run().expect("Broadcast test failed");
}

#[test]
fn test_expand() {
    // Expand single value to multiple outputs
    let test = OperationTest::unary(
        OpType::Expand,
        vec![3.0f32], // Single input value
        vec![3.0f32, 3.0f32, 3.0f32], // Replicated to output
    );
    test.run().expect("Expand test failed");
}

