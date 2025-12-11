//! Tests for logical operations
//! 
//! Logical operations treat non-zero as true (1.0), zero as false (0.0)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_and() {
    let test = OperationTest::binary(
        OpType::And,
        vec![1.0f32, 0.0f32, 2.0f32, 0.0f32],
        vec![1.0f32, 1.0f32, 0.0f32, 0.0f32],
        vec![1.0f32, 0.0f32, 0.0f32, 0.0f32], // both non-zero = true
    );
    test.run().expect("And test failed");
}

#[test]
fn test_or() {
    let test = OperationTest::binary(
        OpType::Or,
        vec![1.0f32, 0.0f32, 0.0f32, 2.0f32],
        vec![0.0f32, 1.0f32, 0.0f32, 0.0f32],
        vec![1.0f32, 1.0f32, 0.0f32, 1.0f32], // either non-zero = true
    );
    test.run().expect("Or test failed");
}

#[test]
fn test_not() {
    let test = OperationTest::unary(
        OpType::Not,
        vec![0.0f32, 1.0f32, 2.0f32, 0.0f32],
        vec![1.0f32, 0.0f32, 0.0f32, 1.0f32], // zero = true, non-zero = false
    );
    test.run().expect("Not test failed");
}

