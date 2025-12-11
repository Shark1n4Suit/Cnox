//! GlobalMaxPool operation
//! Max over all spatial dimensions - identical to ReduceMax

use coelanox_core::CoelanoxResult;
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::reductions;

pub fn generate_globalmaxpool(generator: &CpuCodeGenerator, node: &GraphNode, input_shape: Option<&Shape>, output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // GlobalMaxPool: max over all spatial dimensions
    // Implementation is identical to ReduceMax - reduce all elements to max
    reductions::generate_reducemax(generator, node, input_shape, output_shape)
}

