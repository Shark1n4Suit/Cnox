//! Expand operation
//! Similar to broadcast - replicate tensor to larger shape

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::misc::broadcast;

pub fn generate_expand(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Expand: similar to broadcast - replicate tensor to larger shape
    // Same implementation as broadcast for simple cases
    broadcast::generate_broadcast(generator, _node, _input_shape, _output_shape)
}

