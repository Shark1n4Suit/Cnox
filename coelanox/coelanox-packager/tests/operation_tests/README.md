# Operation Test Suite

This directory contains comprehensive tests for all 46 operations in the CPU codegen.

## Structure

- `test_helpers.rs` - Shared test utilities and `OperationTest` struct
- `test_*.rs` - Individual test files for each operation

## How It Works

Each test file defines:
1. **Input data** - Specific test cases with known values
2. **Expected output** - The correct result for that operation
3. **Tolerance** - Allowed floating-point error (default: 0.001)

The `OperationTest` helper:
- Creates the IR for the operation
- Generates CPU code
- Executes the code
- Verifies output matches expected result

## Running Tests

```bash
# Run all operation tests
cargo test --test test_operation_tests

# Run specific operation
cargo test test_add --test test_operation_tests

# Run with output
cargo test --test test_operation_tests -- --nocapture
```

## Test Coverage

### ✅ Fully Tested (8 operations)
- Add
- Subtract  
- Multiply
- Divide
- Relu
- Reshape
- ReduceSum
- ReduceMax

### ⚠️ Needs Implementation (38 operations)
All other operations are registered but return errors or are stubs.

## Adding New Tests

To add a test for a new operation:

1. Create `test_operationname.rs` in this directory
2. Add the module to `mod.rs`
3. Use `OperationTest::binary()` or `OperationTest::unary()` helper
4. Define input and expected output

Example:
```rust
use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;

#[test]
fn test_operationname() {
    let test = OperationTest::binary(
        OpType::OperationName,
        vec![1.0f32, 2.0f32],
        vec![3.0f32, 4.0f32],
        vec![4.0f32, 6.0f32], // expected
    );
    test.run().expect("Test failed");
}
```

