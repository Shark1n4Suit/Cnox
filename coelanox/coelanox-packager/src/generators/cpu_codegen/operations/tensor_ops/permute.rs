//! Permute operation
//! Permute tensor dimensions according to permutation array

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use super::{reshape, transpose};

pub fn generate_permute(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Permute: permute tensor dimensions according to permutation
    // For 1D tensors, permute is a no-op (just copy) - correct behavior
    // For 2D: fully implemented - handles [0,1] (identity) and [1,0] (transpose)
    // For 2D, there are only 2 valid permutations:
    //   [0, 1]: identity (no change) - uses reshape (copy)
    //   [1, 0]: transpose - uses transpose implementation
    // For 3D+: requires multi-dimensional index remapping (not yet implemented)
    // Attributes: "perm" (IntArray) - e.g., [1, 0] for 2D transpose, [1, 0, 2] for 3D
    // Shape-aware: Supports 1D and 2D (fully implemented)
    
    // Check if 1D (rank <= 1) - permute is no-op
    if let Some(shape) = _input_shape {
        if shape.rank() <= 1 {
            // 1D permute is a no-op - just copy
            return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
        }
        
        // For 2D: handle all permutations
        if shape.rank() == 2 {
            if let Some(perm_attr) = _node.attributes.get("perm") {
                if let coelanox_core::ir::Attribute::IntArray(perm) = perm_attr {
                    if perm.len() == 2 {
                        // For 2D, there are only 2 valid permutations:
                        // [0, 1]: identity (no-op, just copy)
                        // [1, 0]: transpose
                        
                        if perm[0] == 1 && perm[1] == 0 {
                            // This is a 2D transpose - use transpose implementation
                            return transpose::generate_transpose(generator, _node, _input_shape, _output_shape);
                        } else if perm[0] == 0 && perm[1] == 1 {
                            // Identity permutation - just copy (reshape handles this)
                            return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
                        } else {
                            // Invalid permutation for 2D - should not happen, but handle gracefully
                            return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
                        }
                    }
                }
            }
            // No permutation specified - treat as identity (copy)
            return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
        }
        
        // For 3D+: need permutation array
        if let Some(perm_attr) = _node.attributes.get("perm") {
            if let coelanox_core::ir::Attribute::IntArray(_perm) = perm_attr {
                // We have permutation array, but implementing multi-dimensional
                // index remapping is complex - requires calculating source indices
                // from destination indices based on permutation
                return Err(CoelanoxError::Internal("3D+ Permute requires multi-dimensional index remapping - use Reshape for 1D/2D or implement 3D+ indexing".to_string()));
            }
        }
    }
    
    // Default: 1D case (no-op)
    reshape::generate_reshape(generator, _node, _input_shape, _output_shape)
}

