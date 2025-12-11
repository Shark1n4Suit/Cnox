//! Operation test suite - tests all 46 operations with known inputs/outputs
//! 
//! Each operation has a test file that:
//! - Provides specific input data
//! - Generates code for that operation
//! - Executes the code
//! - Verifies output matches expected result

pub mod test_helpers;
pub mod test_add;
pub mod test_subtract;
pub mod test_multiply;
pub mod test_divide;
pub mod test_relu;
pub mod test_reshape;
pub mod test_reducesum;
pub mod test_reducemax;
pub mod test_reducemin;
pub mod test_reduceprod;
pub mod test_reducemean;
pub mod test_comparison;
pub mod test_logical;
pub mod test_sigmoid;
pub mod test_tanh;
pub mod test_swish;
pub mod test_globalpool;
pub mod test_broadcast;
pub mod test_matmul;
pub mod test_tensor_manipulation;
pub mod test_neural_network;
pub mod test_all_operations;

// Add more test modules as operations are implemented

