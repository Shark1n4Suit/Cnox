//! Comprehensive operation test suite
//! 
//! Tests all operations with known inputs and expected outputs
//! Each operation has specific test cases to verify correctness

mod operation_tests {
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
}

// Re-export tests
use operation_tests::test_add::*;
use operation_tests::test_subtract::*;
use operation_tests::test_multiply::*;
use operation_tests::test_divide::*;
use operation_tests::test_relu::*;
use operation_tests::test_reshape::*;
use operation_tests::test_reducesum::*;
use operation_tests::test_reducemax::*;
use operation_tests::test_all_operations::*;

