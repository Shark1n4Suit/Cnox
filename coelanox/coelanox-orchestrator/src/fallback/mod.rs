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

use crate::execution_backend::{ContainerWarmData, ExecutionBackend};

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
    /// When warm_data is Some, uses cached IR and decompressed weights; otherwise deserializes and decompresses.
    fn execute_ir(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        warm_data: Option<&ContainerWarmData>,
    ) -> CoelanoxResult<Vec<f32>> {
        let (ir, weights): (&coelanox_core::ir::OptimizedIR, &[f32]) = match warm_data {
            Some(warm) => (&warm.optimized_ir, warm.decompressed_weights.as_slice()),
            None => {
                use coelanox_core::ir::deserialize_optimized_ir;
                let ir_owned = deserialize_optimized_ir(&container.ir_data)?;
                let weights_owned = container.decompress_weights()?;
                return self.execute_ir_with(container, input_data, &ir_owned, &weights_owned);
            }
        };
        self.execute_ir_with(container, input_data, ir, weights)
    }

    /// Shared execution path: run graph with given IR and weight slice.
    fn execute_ir_with(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        ir: &coelanox_core::ir::OptimizedIR,
        weights: &[f32],
    ) -> CoelanoxResult<Vec<f32>> {

        let mut tensor_data: HashMap<TensorId, Vec<f32>> =
            HashMap::with_capacity(ir.intermediate_tensors.len().saturating_add(ir.optimized_nodes.len()));

        let produced: HashSet<TensorId> = ir
            .optimized_nodes
            .iter()
            .flat_map(|n| n.output_tensors.iter().copied())
            .collect();

        // Build execution order: topological sort so every node runs after producers of its input intermediates.
        let node_order = {
            let nodes = &ir.optimized_nodes;
            let mut producer: HashMap<TensorId, usize> = HashMap::new();
            for (i, node) in nodes.iter().enumerate() {
                for &tid in &node.output_tensors {
                    if tid.is_intermediate() {
                        producer.insert(tid, i);
                    }
                }
            }
            let mut in_degree: Vec<usize> = vec![0; nodes.len()];
            let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
            for (i, node) in nodes.iter().enumerate() {
                for &tid in &node.input_tensors {
                    if tid.is_intermediate() && produced.contains(&tid) {
                        if let Some(&p) = producer.get(&tid) {
                            in_degree[i] += 1;
                            dependents[p].push(i);
                        }
                    }
                }
            }
            let mut queue: std::collections::VecDeque<usize> = (0..nodes.len()).filter(|&i| in_degree[i] == 0).collect();
            let mut order = Vec::with_capacity(nodes.len());
            while let Some(i) = queue.pop_front() {
                order.push(i);
                for &j in &dependents[i] {
                    in_degree[j] -= 1;
                    if in_degree[j] == 0 {
                        queue.push_back(j);
                    }
                }
            }
            if order.len() != nodes.len() {
                return Err(CoelanoxError::Runtime(format!(
                    "Graph has cycle or missing producers: sorted {} of {} nodes",
                    order.len(),
                    nodes.len()
                )));
            }
            order
        };

        // Collect graph inputs (intermediates consumed but never produced); sort by TensorId for deterministic assignment.
        let mut graph_inputs: Vec<TensorId> = ir
            .optimized_nodes
            .iter()
            .flat_map(|n| n.input_tensors.iter().copied())
            .filter(|tid| tid.is_intermediate() && !produced.contains(tid))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        graph_inputs.sort_by_key(|tid| tid.0);

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
        let total_weight_elements = weight_offsets.last().copied().unwrap_or(0);
        if weights.len() != total_weight_elements {
            return Err(CoelanoxError::Runtime(format!(
                "Weight buffer size mismatch: decompressed {} elements, IR expects {}",
                weights.len(),
                total_weight_elements
            )));
        }

        /// Resolve first input as a slice (no clone): from tensor_data or weights.
        fn input_a_slice<'a>(
            tid_a: TensorId,
            tensor_data: &'a HashMap<TensorId, Vec<f32>>,
            weights: &'a [f32],
            weight_offsets: &'a [usize],
            node_index: usize,
            ir_weights_len: usize,
        ) -> CoelanoxResult<&'a [f32]> {
            if tid_a.is_weight() {
                let idx = tid_a.index();
                let start = *weight_offsets.get(idx).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Missing first input for node {}: weight index {} out of range (weights count {})",
                    node_index, idx, ir_weights_len
                )))?;
                let end = *weight_offsets.get(idx + 1).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Missing first input for node {}: weight index {} end out of range",
                    node_index, idx
                )))?;
                weights.get(start..end).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Missing first input for node {}: weight slice [{}, {}) out of bounds (weights len {})",
                    node_index, start, end, weights.len()
                )))
            } else {
                tensor_data.get(&tid_a).map(|v| v.as_slice()).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Missing first input for node {}: tensor {:?} not in tensor_data",
                    node_index, tid_a
                )))
            }
        }

        /// Resolve second input as a slice (no clone): from tensor_data or weights. Errors if missing.
        fn input_b_slice<'a>(
            tid: TensorId,
            tensor_data: &'a HashMap<TensorId, Vec<f32>>,
            weights: &'a [f32],
            weight_offsets: &'a [usize],
            node_index: usize,
        ) -> CoelanoxResult<&'a [f32]> {
            if tid.is_weight() {
                let idx = tid.index();
                let start = *weight_offsets.get(idx).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Node {}: weight index {} out of range", node_index, idx
                )))?;
                let end = *weight_offsets.get(idx + 1).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Node {}: weight index {} end out of range", node_index, idx
                )))?;
                weights.get(start..end).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Node {}: weight slice [{}, {}) out of bounds", node_index, start, end
                )))
            } else {
                tensor_data.get(&tid).map(|v| v.as_slice()).ok_or_else(|| CoelanoxError::Runtime(format!(
                    "Node {}: intermediate tensor {:?} not in tensor_data", node_index, tid
                )))
            }
        }

        // Reusable output buffer to avoid per-node allocations.
        let mut output_buf = Vec::<f32>::new();

        for &node_index in node_order.iter() {
            let node = &ir.optimized_nodes[node_index];
            let tid_a = match node.input_tensors.first() {
                Some(t) => *t,
                None => {
                    return Err(CoelanoxError::Runtime(format!(
                        "Missing first input for node {} (node has no inputs)",
                        node_index
                    )));
                }
            };
            // Defensive: if first input is intermediate and missing, and we have a single logical input, assign it (handles graph_inputs ordering/empty edge cases).
            if tid_a.is_intermediate() && !tensor_data.contains_key(&tid_a) && graph_inputs.len() <= 1 && input_slice.len() == expected_input {
                tensor_data.insert(tid_a, input_slice.to_vec());
            }
            let input_a: &[f32] = input_a_slice(tid_a, &tensor_data, &weights, &weight_offsets, node_index, ir.shared_weights.len())?;

            let input_b: Option<&[f32]> = node.input_tensors.get(1).map(|&tid| {
                input_b_slice(tid, &tensor_data, &weights, &weight_offsets, node_index)
            }).transpose()?;

            // Pass weight slice by reference to avoid cloning (Conv/MatMul/BatchNorm are weight-heavy).
            let weight_slice: Option<&[f32]> = node.input_tensors.iter().find(|tid| tid.is_weight()).and_then(|tid| {
                let idx = tid.index();
                let start = *weight_offsets.get(idx)?;
                let end = *weight_offsets.get(idx + 1)?;
                Some(weights.get(start..end)?)
            });

            let output_size = node.output_tensors.first().and_then(|tid| {
                if tid.is_intermediate() {
                    ir.intermediate_tensors.get(tid.index()).map(|t| t.elements())
                } else {
                    None
                }
            }).unwrap_or_else(|| {
                input_a.len().max(input_b.map(|b: &[f32]| b.len()).unwrap_or(0))
            });

            // Single output: allocate once, execute, move into map (no clone). Multi-output: reuse buffer and clone.
            match node.output_tensors.as_slice() {
                [tid] if tid.is_intermediate() => {
                    let mut output = vec![0.0f32; output_size];
                    execute_scalar_op(
                        &node.op_type,
                        input_a,
                        input_b,
                        &mut output,
                        weight_slice,
                        &node.attributes,
                    )?;
                    if crate::audit::is_enabled() {
                        crate::audit::log_after_op(node_index, &node.op_type, output_size, &output);
                    }
                    tensor_data.insert(*tid, output);
                }
                _ => {
                    output_buf.resize(output_size, 0.0f32);
                    let output = &mut output_buf[..output_size];
                    execute_scalar_op(
                        &node.op_type,
                        input_a,
                        input_b,
                        output,
                        weight_slice,
                        &node.attributes,
                    )?;
                    if crate::audit::is_enabled() {
                        crate::audit::log_after_op(node_index, &node.op_type, output_size, output);
                    }
                    for &tid in &node.output_tensors {
                        if tid.is_intermediate() {
                            tensor_data.insert(tid, output.to_vec());
                        }
                    }
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

        // Prefer remove() so we return the last node's output Vec without cloning.
        let result = node_order.last().and_then(|&idx| {
            let node = &ir.optimized_nodes[idx];
            node.output_tensors.first().and_then(|tid| tensor_data.remove(tid))
        });

        match result {
            Some(mut v) if v.len() >= output_size => {
                v.truncate(output_size);
                Ok(v)
            }
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
    fn execute(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        warm_data: Option<&ContainerWarmData>,
    ) -> CoelanoxResult<Vec<f32>> {
        self.execute_ir(container, input_data, warm_data)
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

