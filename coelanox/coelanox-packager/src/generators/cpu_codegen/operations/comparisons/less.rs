//! Less operation
//! Element-wise less than: output = (input_a < input_b) ? 1.0 : 0.0

use coelanox_core::{CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use super::helpers::generate_comparison_binary;

pub fn generate_less(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Less: output = (input_a < input_b) ? 1.0 : 0.0
    generate_comparison_binary(generator, 2, _output_shape)
}

