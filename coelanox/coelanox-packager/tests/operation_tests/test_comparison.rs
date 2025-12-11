//! Tests for comparison operations
//! 
//! All comparison operations return 1.0 for true, 0.0 for false

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_equal() {
    let test = OperationTest::binary(
        OpType::Equal,
        vec![1.0f32, 2.0f32, 3.0f32],
        vec![1.0f32, 2.0f32, 4.0f32],
        vec![1.0f32, 1.0f32, 0.0f32],
    );
    test.run().expect("Equal test failed");
}

#[test]
fn test_notequal() {
    let test = OperationTest::binary(
        OpType::NotEqual,
        vec![1.0f32, 2.0f32, 3.0f32],
        vec![1.0f32, 2.0f32, 4.0f32],
        vec![0.0f32, 0.0f32, 1.0f32],
    );
    test.run().expect("NotEqual test failed");
}

#[test]
fn test_less() {
    let test = OperationTest::binary(
        OpType::Less,
        vec![1.0f32, 3.0f32, 2.0f32],
        vec![2.0f32, 2.0f32, 2.0f32],
        vec![1.0f32, 0.0f32, 0.0f32],
    );
    test.run().expect("Less test failed");
}

#[test]
fn test_lessequal() {
    let test = OperationTest::binary(
        OpType::LessEqual,
        vec![1.0f32, 2.0f32, 3.0f32],
        vec![2.0f32, 2.0f32, 2.0f32],
        vec![1.0f32, 1.0f32, 0.0f32],
    );
    test.run().expect("LessEqual test failed");
}

#[test]
fn test_greater() {
    let test = OperationTest::binary(
        OpType::Greater,
        vec![3.0f32, 2.0f32, 1.0f32],
        vec![2.0f32, 2.0f32, 2.0f32],
        vec![1.0f32, 0.0f32, 0.0f32],
    );
    test.run().expect("Greater test failed");
}

#[test]
fn test_greaterequal() {
    let test = OperationTest::binary(
        OpType::GreaterEqual,
        vec![3.0f32, 2.0f32, 1.0f32],
        vec![2.0f32, 2.0f32, 2.0f32],
        vec![1.0f32, 1.0f32, 0.0f32],
    );
    test.run().expect("GreaterEqual test failed");
}

