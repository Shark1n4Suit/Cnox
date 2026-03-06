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
use std::sync::Arc;
use rayon::prelude::*;

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
            // Element-wise min/max
            OpType::Min,
            OpType::Max,
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

        // Arc so we can share tensor buffers across parallel node execution.
        let mut tensor_data: HashMap<TensorId, Arc<Vec<f32>>> =
            HashMap::with_capacity(ir.intermediate_tensors.len().saturating_add(ir.optimized_nodes.len()));

        let produced: HashSet<TensorId> = ir
            .optimized_nodes
            .iter()
            .flat_map(|n| n.output_tensors.iter().copied())
            .collect();

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
        let dependents: Vec<Vec<usize>> = {
            let mut dep = vec![Vec::new(); nodes.len()];
            for (i, node) in nodes.iter().enumerate() {
                for &tid in &node.input_tensors {
                    if tid.is_intermediate() && produced.contains(&tid) {
                        if let Some(&p) = producer.get(&tid) {
                            in_degree[i] += 1;
                            dep[p].push(i);
                        }
                    }
                }
            }
            dep
        };
        let node_order: Vec<usize> = {
            let mut deg = in_degree.clone();
            let mut queue: std::collections::VecDeque<usize> = (0..nodes.len()).filter(|&i| deg[i] == 0).collect();
            let mut order = Vec::with_capacity(nodes.len());
            while let Some(i) = queue.pop_front() {
                order.push(i);
                for &j in &dependents[i] {
                    deg[j] -= 1;
                    if deg[j] == 0 {
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
            if let Some(node) = nodes.first() {
                for &tid in &node.input_tensors {
                    if tid.is_intermediate() && !tensor_data.contains_key(&tid) {
                        tensor_data.insert(tid, Arc::new(input_slice.to_vec()));
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
                tensor_data.insert(tid, Arc::new(input_slice[start..end].to_vec()));
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

        // Wave-based parallel execution: run all ready nodes (in_degree 0) in parallel each round.
        let weights_ref = weights;
        let weight_offsets_ref = &weight_offsets;
        let audit_enabled = crate::audit::is_enabled();

        while in_degree.iter().any(|&d| d == 0) {
            if crate::runtime::execution_deadline_expired() {
                return Err(CoelanoxError::Runtime(
                    "Execution timeout exceeded (COELANOX_EXECUTION_TIMEOUT_MS)".to_string(),
                ));
            }
            let ready: Vec<usize> = (0..nodes.len()).filter(|&i| in_degree[i] == 0).collect();
            if ready.is_empty() {
                break;
            }
            // Mark these nodes as processed so they don't appear ready again.
            for &i in &ready {
                in_degree[i] = usize::MAX;
            }

            #[allow(clippy::type_complexity)]
            let jobs: Vec<(usize, Arc<Vec<f32>>, Option<Arc<Vec<f32>>>, Option<(usize, usize)>, usize, OpType, HashMap<String, Attribute>, Vec<TensorId>)> = ready
                .iter()
                .filter_map(|&node_index| {
                    let node = &nodes[node_index];
                    let tid_a = node.input_tensors.first()?;
                    let input_a = if tid_a.is_weight() {
                        let idx = tid_a.index();
                        let start = *weight_offsets_ref.get(idx)?;
                        let end = *weight_offsets_ref.get(idx + 1)?;
                        Some(Arc::new(weights_ref[start..end].to_vec()))
                    } else {
                        tensor_data.get(tid_a).cloned()
                    }?;
                    let input_b = node.input_tensors.get(1).and_then(|&tid| {
                        if tid.is_weight() {
                            let idx = tid.index();
                            let start = *weight_offsets_ref.get(idx)?;
                            let end = *weight_offsets_ref.get(idx + 1)?;
                            Some(Arc::new(weights_ref[start..end].to_vec()))
                        } else {
                            tensor_data.get(&tid).cloned()
                        }
                    });
                    let weight_range = node.input_tensors.iter().find(|t| t.is_weight()).and_then(|tid| {
                        let idx = tid.index();
                        let start = weight_offsets_ref.get(idx).copied()?;
                        let end = weight_offsets_ref.get(idx + 1).copied()?;
                        Some((start, end))
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
                    let out_tids: Vec<TensorId> = node.output_tensors.iter()
                        .filter(|tid| tid.is_intermediate())
                        .copied()
                        .collect();
                    if out_tids.is_empty() {
                        return None;
                    }
                    // For Conv, add input_shape so scalar can use im2col+sgemm (2D NCHW).
                    let mut attrs = node.attributes.clone();
                    if matches!(node.op_type, OpType::Convolution) {
                        let in_tid = node.input_tensors.iter().find(|t| !t.is_weight())?;
                        if in_tid.is_intermediate() {
                            if let Some(t) = ir.intermediate_tensors.get(in_tid.index()) {
                                if t.shape.dimensions.len() >= 4 {
                                    attrs.insert(
                                        "input_shape".to_string(),
                                        Attribute::IntArray(t.shape.dimensions.iter().map(|&d| d as i64).collect()),
                                    );
                                }
                            }
                        }
                    }
                    Some((
                        node_index,
                        input_a,
                        input_b,
                        weight_range,
                        output_size,
                        node.op_type.clone(),
                        attrs,
                        out_tids,
                    ))
                })
                .collect();

            // Return audit data from workers so we log from the main thread (thread_local audit only sees main thread otherwise)
            type JobResult = Result<(usize, Vec<f32>, Vec<TensorId>, Option<(OpType, usize, Vec<f32>)>), CoelanoxError>;
            let results: Vec<JobResult> = jobs
                .into_par_iter()
                .map(|(node_index, input_a, input_b, weight_range, output_size, op_type, attributes, out_tids)| {
                    let input_a_slice = input_a.as_slice();
                    let input_b_slice = input_b.as_ref().map(|b| b.as_slice());
                    let weight_slice = weight_range.and_then(|(s, e)| weights_ref.get(s..e));
                    let mut output = vec![0.0f32; output_size];
                    execute_scalar_op(
                        &op_type,
                        input_a_slice,
                        input_b_slice,
                        &mut output,
                        weight_slice,
                        &attributes,
                    )?;
                    let audit_data = if audit_enabled {
                        let sample: Vec<f32> = output.iter().take(4).copied().collect();
                        Some((op_type, output_size, sample))
                    } else {
                        None
                    };
                    Ok((node_index, output, out_tids, audit_data))
                })
                .collect();

            for res in results {
                let (node_index, output, out_tids, audit_data) = res?;
                if let Some((ref op_type, output_size, ref sample)) = audit_data {
                    crate::audit::log_after_op(node_index, op_type, output_size, sample);
                }
                let out_arc = Arc::new(output);
                for tid in &out_tids {
                    tensor_data.insert(*tid, Arc::clone(&out_arc));
                }
                for &j in &dependents[node_index] {
                    in_degree[j] = in_degree[j].saturating_sub(1);
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

        // Prefer remove() so we return the last node's output Vec.
        let result = node_order.last().and_then(|&idx| {
            let node = &ir.optimized_nodes[idx];
            node.output_tensors.first().and_then(|tid| tensor_data.remove(tid))
        });

        match result {
            Some(arc) if arc.len() >= output_size => {
                let mut v = (*arc).clone();
                v.truncate(output_size);
                Ok(v)
            }
            Some(arc) => Ok((*arc).clone()),
            None => tensor_data
                .values()
                .find(|v| v.len() == output_size)
                .cloned()
                .map(|arc| (*arc).clone())
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

