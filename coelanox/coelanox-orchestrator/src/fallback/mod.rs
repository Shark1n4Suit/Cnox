//! Scalar Fallback Backend for COELANOX
//!
//! This module provides a pure Rust scalar fallback implementation that uses
//! the authoritative scalar backend from `coelanox-core`. It serves as a
//! 100% uptime guarantee when vendor backends are unavailable or fail.
//!
//! **Key Features:**
//! - 1:1 mapping with Universal IR operations via `coelanox-core::scalar::execute_scalar_op`
//! - Pure Rust implementation (no SIMD, no external dependencies)
//! - Bit-exact correctness (golden reference)
//! - Always available (100% uptime guarantee)
//!
//! **Architecture:** ScalarFallbackBackend is registered as an ExecutionBackend in the
//! RuntimeBackendRegistry. The runtime tries CPU native machine code first, then falls back
//! to this backend via the registry when machine code execution fails.

use coelanox_core::format::CoelanoxContainer;
use coelanox_core::ir::{OpType, GraphNode, Attribute, TensorId};
use coelanox_core::scalar::execute_scalar_op;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::{HashMap, HashSet};

use crate::execution_backend::ExecutionBackend;

/// Scalar fallback backend
/// 
/// This backend uses the authoritative scalar operations from `coelanox-core`
/// to execute operations when vendor backends are unavailable or fail.
#[derive(Debug, Clone)]
pub struct ScalarFallbackBackend {
    /// Backend name
    name: String,
    /// Backend version
    version: String,
}

impl ScalarFallbackBackend {
    /// Create a new scalar fallback backend
    pub fn new() -> Self {
        Self {
            name: "scalar-fallback".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
    
    /// Get backend name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get backend version
    pub fn version(&self) -> &str {
        &self.version
    }
    
    /// Execute a single operation using scalar fallback
    /// 
    /// This method provides a 1:1 mapping with Universal IR operations
    /// by delegating to `coelanox-core::scalar::execute_scalar_op`.
    /// 
    /// # Arguments
    /// * `op_type` - The operation type from OpType enum
    /// * `input_a` - First input tensor (or single input for unary ops)
    /// * `input_b` - Second input tensor (optional, for binary ops)
    /// * `output` - Output tensor buffer (must be pre-allocated)
    /// * `weights` - Weight tensor (optional, for operations that need weights)
    /// * `attributes` - Operation-specific attributes
    /// 
    /// # Returns
    /// * `Ok(())` if operation succeeds
    /// * `Err(CoelanoxError)` if operation fails
    pub fn execute_operation(
        &self,
        op_type: &OpType,
        input_a: &[f32],
        input_b: Option<&[f32]>,
        output: &mut [f32],
        weights: Option<&[f32]>,
        attributes: &HashMap<String, Attribute>,
    ) -> CoelanoxResult<()> {
        // Delegate to core scalar operations for 1:1 mapping
        coelanox_core::scalar::execute_scalar_op(
            op_type,
            input_a,
            input_b,
            output,
            weights,
            attributes,
        )
    }
    
    /// Execute a graph node using scalar fallback
    /// 
    /// This method extracts operation information from a GraphNode and
    /// executes it using the scalar fallback.
    /// 
    /// # Arguments
    /// * `node` - Graph node containing operation information
    /// * `input_a` - First input tensor
    /// * `input_b` - Second input tensor (optional)
    /// * `output` - Output tensor buffer
    /// * `weights` - Weight tensor (optional)
    /// 
    /// # Returns
    /// * `Ok(())` if operation succeeds
    /// * `Err(CoelanoxError)` if operation fails
    pub fn execute_node(
        &self,
        node: &GraphNode,
        input_a: &[f32],
        input_b: Option<&[f32]>,
        output: &mut [f32],
        weights: Option<&[f32]>,
    ) -> CoelanoxResult<()> {
        self.execute_operation(
            &node.op_type,
            input_a,
            input_b,
            output,
            weights,
            &node.attributes,
        )
    }
    
    /// Check if this backend supports an operation
    /// 
    /// The scalar fallback supports all operations that have a corresponding
    /// implementation in `coelanox-core::scalar` (1:1 mapping with OpType).
    /// 
    /// # Arguments
    /// * `op_type` - The operation type to check
    /// 
    /// # Returns
    /// * `true` if operation is supported
    /// * `false` if operation is not supported (Custom only)
    pub fn supports_operation(&self, op_type: &OpType) -> bool {
        match op_type {
            OpType::Custom(_) => false,
            _ => true,
        }
    }
    
    /// Get list of supported operations
    /// 
    /// Returns a list of all OpType variants that are supported by the scalar fallback.
    /// This matches the 1:1 mapping in `coelanox-core::scalar::execute_scalar_op`.
    pub fn supported_operations(&self) -> Vec<OpType> {
        // Return all non-Custom OpType variants
        vec![
            // Arithmetic
            OpType::Add,
            OpType::Subtract,
            OpType::Multiply,
            OpType::Divide,
            // Activations
            OpType::Relu,
            OpType::Sigmoid,
            OpType::Tanh,
            OpType::Softmax,
            OpType::LogSoftmax,
            OpType::Gelu,
            OpType::Swish,
            // Math
            OpType::Sqrt,
            OpType::Pow,
            OpType::Cos,
            OpType::Sin,
            OpType::Exp,
            OpType::Log,
            // Pooling
            OpType::MaxPool,
            OpType::AvgPool,
            OpType::GlobalMaxPool,
            OpType::GlobalAvgPool,
            // Normalization
            OpType::BatchNorm,
            OpType::LayerNorm,
            // Reductions
            OpType::ReduceSum,
            OpType::ReduceMean,
            OpType::ReduceMax,
            OpType::ReduceMin,
            OpType::ReduceProd,
            // Matrix ops
            OpType::MatMul,
            OpType::Gemm,
            OpType::Convolution,
            // Tensor ops
            OpType::Reshape,
            OpType::Transpose,
            OpType::Permute,
            OpType::Concatenate,
            OpType::Split,
            OpType::Slice,
            OpType::Gather,
            OpType::Scatter,
            OpType::Broadcast,
            OpType::Expand,
            // Comparisons
            OpType::Equal,
            OpType::NotEqual,
            OpType::Greater,
            OpType::GreaterEqual,
            OpType::Less,
            OpType::LessEqual,
            // Logical
            OpType::And,
            OpType::Or,
            OpType::Not,
            // Misc
            OpType::Dropout,
        ]
    }
}

impl Default for ScalarFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar execution backend that implements ExecutionBackend for the runtime registry.
///
/// Executes the full IR graph using the scalar fallback. Registered as "scalar-fallback"
/// in the RuntimeBackendRegistry. The runtime routes to this backend when CPU native
/// machine code execution fails.
#[derive(Debug)]
pub struct ScalarExecutionBackend {
    scalar: ScalarFallbackBackend,
}

impl ScalarExecutionBackend {
    /// Create a new scalar execution backend.
    pub fn new() -> Self {
        Self {
            scalar: ScalarFallbackBackend::new(),
        }
    }

    /// Execute the full IR graph using scalar operations.
    fn execute_ir(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        use bincode;
        use coelanox_core::ir::{GraphNode, IntermediateTensor, WeightTensor};
        use coelanox_core::format::HardwareTarget;

        #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
        struct SerializableIR {
            target: HardwareTarget,
            optimized_nodes: Vec<GraphNode>,
            intermediate_tensors: Vec<IntermediateTensor>,
            optimization_metadata: HashMap<String, String>,
            original_model_name: String,
            weight_tensors: Vec<WeightTensor>,
        }

        let serialized: SerializableIR = bincode::deserialize(&container.ir_data)
            .map_err(|e| CoelanoxError::Runtime(format!("Failed to deserialize IR: {}", e)))?;

        let ir = coelanox_core::ir::OptimizedIR::from_deserialized(
            serialized.target,
            serialized.optimized_nodes,
            serialized.intermediate_tensors,
            serialized.optimization_metadata,
            serialized.original_model_name,
            serialized.weight_tensors,
        );

        let weights = container.decompress_weights()?;

        let mut tensor_data: HashMap<TensorId, Vec<f32>> = HashMap::new();

        let produced: HashSet<TensorId> = ir
            .optimized_nodes
            .iter()
            .flat_map(|n| n.output_tensors.iter().copied())
            .collect();

        let graph_inputs: Vec<TensorId> = ir
            .optimized_nodes
            .iter()
            .flat_map(|n| n.input_tensors.iter().copied())
            .filter(|tid| tid.is_intermediate() && !produced.contains(tid))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let expected_input: usize = container
            .manifest
            .input_shapes
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum();
        if input_data.len() < expected_input {
            return Err(CoelanoxError::Runtime(format!(
                "Input size mismatch: expected {} elements, got {}",
                expected_input,
                input_data.len()
            )));
        }
        let input_slice = &input_data[..expected_input];

        if graph_inputs.is_empty() {
            if let Some(node) = ir.optimized_nodes.first() {
                for &tid in &node.input_tensors {
                    if tid.is_intermediate() && !tensor_data.contains_key(&tid) {
                        tensor_data.insert(tid, input_slice.to_vec());
                        break;
                    }
                }
            }
        } else {
            let per_input = input_slice.len() / graph_inputs.len().max(1);
            for (i, &tid) in graph_inputs.iter().enumerate() {
                let start = i * per_input;
                let end = if i == graph_inputs.len() - 1 {
                    input_slice.len()
                } else {
                    start + per_input
                };
                tensor_data.insert(tid, input_slice[start..end].to_vec());
            }
        }

        let mut weight_offsets: Vec<usize> = vec![0];
        for wt in ir.shared_weights.iter() {
            let next = weight_offsets.last().unwrap() + wt.elements();
            weight_offsets.push(next);
        }

        for node in &ir.optimized_nodes {
            let input_a = node.input_tensors.first().and_then(|tid| {
                if tid.is_weight() {
                    let idx = tid.index();
                    let start = *weight_offsets.get(idx)?;
                    let end = *weight_offsets.get(idx + 1)?;
                    Some(weights.get(start..end)?.to_vec())
                } else {
                    tensor_data.get(tid).cloned()
                }
            }).ok_or_else(|| CoelanoxError::Runtime("Missing first input for node".to_string()))?;

            let input_b = node.input_tensors.get(1).and_then(|tid| {
                if tid.is_weight() {
                    let idx = tid.index();
                    let start = *weight_offsets.get(idx)?;
                    let end = *weight_offsets.get(idx + 1)?;
                    Some(weights.get(start..end)?.to_vec())
                } else {
                    tensor_data.get(tid).cloned()
                }
            });

            let weight_slice = node.input_tensors.iter().find(|tid| tid.is_weight()).and_then(|tid| {
                let idx = tid.index();
                let start = *weight_offsets.get(idx)?;
                let end = *weight_offsets.get(idx + 1)?;
                Some(weights.get(start..end)?.to_vec())
            });

            let output_size = node.output_tensors.first().and_then(|tid| {
                if tid.is_intermediate() {
                    ir.intermediate_tensors.get(tid.index()).map(|t| t.elements())
                } else {
                    None
                }
            }).unwrap_or_else(|| {
                input_a.len().max(input_b.as_ref().map(|b| b.len()).unwrap_or(0))
            });

            let mut output = vec![0.0f32; output_size];

            execute_scalar_op(
                &node.op_type,
                &input_a,
                input_b.as_deref(),
                &mut output,
                weight_slice.as_deref(),
                &node.attributes,
            )?;

            for &tid in &node.output_tensors {
                if tid.is_intermediate() {
                    tensor_data.insert(tid, output.clone());
                }
            }
        }

        let output_size: usize = container
            .manifest
            .output_shapes
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum();
        if output_size == 0 {
            return Ok(vec![]);
        }

        let result = if let Some(last) = ir.optimized_nodes.last() {
            last.output_tensors.first().and_then(|tid| tensor_data.get(tid)).cloned()
        } else {
            None
        };

        match result {
            Some(v) if v.len() >= output_size => Ok(v[..output_size].to_vec()),
            Some(v) => Ok(v),
            None => tensor_data
                .values()
                .find(|v| v.len() == output_size)
                .cloned()
                .ok_or_else(|| CoelanoxError::Runtime("Could not resolve output tensor".to_string())),
        }
    }
}

impl ExecutionBackend for ScalarExecutionBackend {
    fn execute(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        self.execute_ir(container, input_data)
    }

    fn name(&self) -> &str {
        self.scalar.name()
    }

    fn version(&self) -> &str {
        self.scalar.version()
    }
}

impl Default for ScalarExecutionBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coelanox_core::ir::NodeId;
    use std::collections::HashMap;
    
    #[test]
    fn test_scalar_fallback_creation() {
        let backend = ScalarFallbackBackend::new();
        assert_eq!(backend.name(), "scalar-fallback");
        assert!(!backend.version().is_empty());
    }
    
    #[test]
    fn test_supports_operation() {
        let backend = ScalarFallbackBackend::new();
        
        // Supported operations
        assert!(backend.supports_operation(&OpType::Add));
        assert!(backend.supports_operation(&OpType::Relu));
        assert!(backend.supports_operation(&OpType::MatMul));
        
        // Unsupported operations
        assert!(!backend.supports_operation(&OpType::Custom("test".to_string())));
    }
    
    #[test]
    fn test_supported_operations_list() {
        let backend = ScalarFallbackBackend::new();
        let supported = backend.supported_operations();
        
        // Should include common operations
        assert!(supported.contains(&OpType::Add));
        assert!(supported.contains(&OpType::Relu));
        assert!(supported.contains(&OpType::MatMul));
        
        // Should not include custom
        assert!(!supported.contains(&OpType::Custom("test".to_string())));
    }
    
    #[test]
    fn test_execute_operation_structure() {
        let backend = ScalarFallbackBackend::new();
        let mut output = vec![0.0f32; 4];
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![5.0, 6.0, 7.0, 8.0];
        let attributes = HashMap::new();

        let result = backend.execute_operation(
            &OpType::Add,
            &input_a,
            Some(&input_b),
            &mut output,
            None,
            &attributes,
        );

        assert!(result.is_ok());
        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_execute_node_structure() {
        let backend = ScalarFallbackBackend::new();
        let mut node = GraphNode::new(NodeId(0), OpType::Relu);
        node.attributes = HashMap::new();

        let mut output = vec![0.0f32; 4];
        let input = vec![-1.0, 0.0, 1.0, 2.0];

        let result = backend.execute_node(&node, &input, None, &mut output, None);

        assert!(result.is_ok());
        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0]);
    }
}

