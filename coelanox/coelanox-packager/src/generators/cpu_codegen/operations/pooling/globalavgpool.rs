//! GlobalAvgPool operation
//! Mean over all spatial dimensions - identical to ReduceMean

use coelanox_core::CoelanoxResult;
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::reductions;

pub fn generate_globalavgpool(generator: &CpuCodeGenerator, node: &GraphNode, input_shape: Option<&Shape>, output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // GlobalAvgPool: mean over all spatial dimensions
    // Implementation is identical to ReduceMean - reduce all elements to mean
    reductions::generate_mean(generator, node, input_shape, output_shape)
}

