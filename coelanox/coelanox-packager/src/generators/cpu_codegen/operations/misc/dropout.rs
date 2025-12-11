//! Dropout operation
//! Randomly zero out elements (requires RNG)
//! For inference mode (no dropout), just copy input to output

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::tensor_ops;

pub fn generate_dropout(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Dropout: randomly zero out elements (requires RNG)
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D) via reshape
    // For inference mode (no dropout), just copy input to output
    // For training mode, would need RNG to randomly zero elements
    // For now, implement as passthrough (copy) - uses reshape which has 2D support
    tensor_ops::generate_reshape(generator, _node, _input_shape, _output_shape)
}

