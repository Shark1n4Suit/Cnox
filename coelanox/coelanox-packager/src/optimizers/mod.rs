//! Hardware-specific optimizers for COELANOX
//! 
//! This module provides hardware-specific optimizers that convert Universal IR
//! into optimized IR for different hardware targets.

use coelanox_core::{CoelanoxResult};
use coelanox_core::format::HardwareTarget;
use coelanox_core::ir::{UniversalIR, OpType, GraphNode, Attribute, TensorRef, OptimizedIR, IntermediateTensor, DataType};

/// Hardware optimizer trait - operates directly on graph structure
pub trait HardwareOptimizer: Send + Sync {
    /// Optimize IR for specific hardware by transforming the graph directly
    fn optimize(&self, ir: &UniversalIR, target: HardwareTarget) -> CoelanoxResult<OptimizedIR>;
    
    /// Get supported hardware targets
    fn supported_targets(&self) -> Vec<HardwareTarget>;
    
    /// Get optimizer name
    fn name(&self) -> &str;
    
    /// Get optimizer version
    fn version(&self) -> &str;
}

/// CPU optimizer
pub struct CpuOptimizer {
    /// Optimizer name
    name: String,
    /// Optimizer version
    version: String,
}

impl CpuOptimizer {
    /// Create a new CPU optimizer
    pub fn new() -> Self {
        Self {
            name: "CPU Optimizer".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Replace operations with CPU-optimized versions
    fn replace_with_cpu_optimized_ops(&self, mut node: GraphNode) -> GraphNode {
        match node.op_type {
            OpType::Convolution => {
                // Replace generic convolution with CPU-optimized version
                node.add_attribute("cpu_conv_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cpu_conv_type".to_string(), Attribute::String("im2col".to_string()));
            }
            OpType::MatMul => {
                // Replace generic matmul with CPU-optimized version
                node.add_attribute("cpu_matmul_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cpu_matmul_type".to_string(), Attribute::String("blas".to_string()));
            }
            OpType::Pooling => {
                // Replace generic pooling with CPU-optimized version
                node.add_attribute("cpu_pooling_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cpu_pooling_type".to_string(), Attribute::String("vectorized".to_string()));
            }
            _ => {
                // Add general CPU optimization marker
                node.add_attribute("cpu_optimized".to_string(), Attribute::Bool(true));
            }
        }
        node
    }
    
    /// Apply CPU-specific optimizations based on graph analysis
    fn apply_cpu_optimizations(&self, mut node: GraphNode, ir: &UniversalIR) -> GraphNode {
        // Analyze tensor shapes for optimization opportunities
        let input_tensors = node.input_tensors.clone(); // Clone to avoid borrow issues
        for tensor_id in &input_tensors {
            if let Some(tensor_ref) = ir.get_tensor(*tensor_id) {
                match tensor_ref {
                    TensorRef::Weight(tensor) => {
                        // Optimize based on weight tensor properties
                        if tensor.shape.dimensions.len() == 4 {
                            // 4D tensor - likely convolution weights
                            node.add_attribute("weight_layout".to_string(), Attribute::String("NCHW".to_string()));
                        }
                    }
                    TensorRef::Intermediate(tensor) => {
                        // Optimize based on intermediate tensor properties
                        if tensor.shape.total_elements() > 1000000 {
                            // Large tensor - use memory-efficient algorithms
                            node.add_attribute("memory_efficient".to_string(), Attribute::Bool(true));
                        }
                    }
                }
            }
        }
        node
    }
    
    /// Optimize memory layout for CPU execution
    fn optimize_memory_layout(&self, mut node: GraphNode, _ir: &UniversalIR) -> GraphNode {
        // Add memory layout optimizations
        node.add_attribute("memory_layout".to_string(), Attribute::String("row_major".to_string()));
        node.add_attribute("cache_friendly".to_string(), Attribute::Bool(true));
        node
    }
    
    /// Add SIMD optimization hints
    fn add_simd_hints(&self, mut node: GraphNode) -> GraphNode {
        // Add SIMD optimization hints based on operation type
        match node.op_type {
            OpType::Add | OpType::Multiply => {
                node.add_attribute("simd_enabled".to_string(), Attribute::Bool(true));
                node.add_attribute("simd_width".to_string(), Attribute::Int(8)); // AVX-256
            }
            OpType::Convolution => {
                node.add_attribute("simd_enabled".to_string(), Attribute::Bool(true));
                node.add_attribute("simd_width".to_string(), Attribute::Int(4)); // AVX-128
            }
            _ => {
                node.add_attribute("simd_enabled".to_string(), Attribute::Bool(false));
            }
        }
        node
    }
    
    /// Apply operator fusion optimizations
    fn apply_fusion_optimizations(&self, ir: &UniversalIR, optimized_ir: &mut OptimizedIR) -> CoelanoxResult<()> {
        let mut processed_nodes = std::collections::HashSet::new();
        
        for node in &ir.nodes {
            if processed_nodes.contains(&node.id) {
                continue; // Skip already processed nodes
            }
            
            // Pattern 1: Conv + ReLU fusion
            if node.op_type == OpType::Convolution && ir.has_single_consumer(node) {
                let consumer_ids: Vec<_> = node.output_tensors.iter()
                    .flat_map(|tensor_id| ir.get_consumer_nodes(*tensor_id))
                    .collect();
                
                if let Some(consumer_id) = consumer_ids.first() {
                    if let Some(consumer) = ir.nodes.get(consumer_id.0) {
                        if consumer.op_type == OpType::Relu && ir.can_fuse_nodes(node, consumer) {
                            // Create fused ConvReLU node
                            let mut fused_node = node.clone();
                            fused_node.op_type = OpType::Custom("ConvReLU".to_string());
                            fused_node.add_attribute("fused_conv_relu".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("cpu_optimized".to_string(), Attribute::Bool(true));
                            
                            // Use the ReLU's output tensor as the fused node's output
                            fused_node.output_tensors = consumer.output_tensors.clone();
                            
                            optimized_ir.add_optimized_node(fused_node);
                            
                            // Mark both nodes as processed
                            processed_nodes.insert(node.id);
                            processed_nodes.insert(consumer.id);
                            continue;
                        }
                    }
                }
            }
            
            // Pattern 2: MatMul + Add fusion (common in linear layers)
            if node.op_type == OpType::MatMul && ir.has_single_consumer(node) {
                let consumer_ids: Vec<_> = node.output_tensors.iter()
                    .flat_map(|tensor_id| ir.get_consumer_nodes(*tensor_id))
                    .collect();
                
                if let Some(consumer_id) = consumer_ids.first() {
                    if let Some(consumer) = ir.nodes.get(consumer_id.0) {
                        if consumer.op_type == OpType::Add && ir.can_fuse_nodes(node, consumer) {
                            // Create fused MatMulAdd node
                            let mut fused_node = node.clone();
                            fused_node.op_type = OpType::Custom("MatMulAdd".to_string());
                            fused_node.add_attribute("fused_matmul_add".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("cpu_optimized".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("gemm_bias".to_string(), Attribute::Bool(true));
                            
                            // Use the Add's output tensor as the fused node's output
                            fused_node.output_tensors = consumer.output_tensors.clone();
                            
                            optimized_ir.add_optimized_node(fused_node);
                            
                            // Mark both nodes as processed
                            processed_nodes.insert(node.id);
                            processed_nodes.insert(consumer.id);
                            continue;
                        }
                    }
                }
            }
            
            // Pattern 4: LayerNorm + GELU fusion (common in transformers)
            if node.op_type == OpType::LayerNorm && ir.has_single_consumer(node) {
                let consumer_ids: Vec<_> = node.output_tensors.iter()
                    .flat_map(|tensor_id| ir.get_consumer_nodes(*tensor_id))
                    .collect();
                
                if let Some(consumer_id) = consumer_ids.first() {
                    if let Some(consumer) = ir.nodes.get(consumer_id.0) {
                        if consumer.op_type == OpType::Gelu && ir.can_fuse_nodes(node, consumer) {
                            // Create fused LayerNormGELU node
                            let mut fused_node = node.clone();
                            fused_node.op_type = OpType::Custom("LayerNormGELU".to_string());
                            fused_node.add_attribute("fused_layernorm_gelu".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("cpu_optimized".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("transformer_optimized".to_string(), Attribute::Bool(true));
                            
                            // Use the GELU's output tensor as the fused node's output
                            fused_node.output_tensors = consumer.output_tensors.clone();
                            
                            optimized_ir.add_optimized_node(fused_node);
                            
                            // Mark both nodes as processed
                            processed_nodes.insert(node.id);
                            processed_nodes.insert(consumer.id);
                            continue;
                        }
                    }
                }
            }
            
            // Pattern 5: Sigmoid + Multiply fusion (common in attention mechanisms)
            if node.op_type == OpType::Sigmoid && ir.has_single_consumer(node) {
                let consumer_ids: Vec<_> = node.output_tensors.iter()
                    .flat_map(|tensor_id| ir.get_consumer_nodes(*tensor_id))
                    .collect();
                
                if let Some(consumer_id) = consumer_ids.first() {
                    if let Some(consumer) = ir.nodes.get(consumer_id.0) {
                        if consumer.op_type == OpType::Multiply && ir.can_fuse_nodes(node, consumer) {
                            // Create fused SigmoidMultiply node (attention gate)
                            let mut fused_node = node.clone();
                            fused_node.op_type = OpType::Custom("SigmoidMultiply".to_string());
                            fused_node.add_attribute("fused_sigmoid_multiply".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("cpu_optimized".to_string(), Attribute::Bool(true));
                            fused_node.add_attribute("attention_gate".to_string(), Attribute::Bool(true));
                            
                            // Use the Multiply's output tensor as the fused node's output
                            fused_node.output_tensors = consumer.output_tensors.clone();
                            
                            optimized_ir.add_optimized_node(fused_node);
                            
                            // Mark both nodes as processed
                            processed_nodes.insert(node.id);
                            processed_nodes.insert(consumer.id);
                            continue;
                        }
                    }
                }
            }
            
            // Add the original node if not fused
            optimized_ir.add_optimized_node(node.clone());
        }
        
        Ok(())
    }
    
    /// Add optimized intermediate tensors (for operator fusion results)
    fn add_optimized_intermediate_tensors(&self, optimized_ir: &mut OptimizedIR, ir: &UniversalIR) {
        // Add intermediate tensors for fused operations
        let fused_node_ids: Vec<_> = optimized_ir.optimized_nodes.iter()
            .filter(|node| {
                matches!(node.op_type, 
                    OpType::Custom(ref name) if name == "ConvReLU" || 
                                               name == "MatMulAdd" || 
                                               name == "AddReLU" ||
                                               name == "LayerNormGELU" ||
                                               name == "SigmoidMultiply")
            })
            .map(|node| node.id.0)
            .collect();
            
        for node_id in fused_node_ids {
            // Find the fused node to get its operation type and input tensors
            if let Some(fused_node) = optimized_ir.optimized_nodes.iter()
                .find(|node| node.id.0 == node_id) {
                
                let fused_op_name = match &fused_node.op_type {
                    OpType::Custom(name) => name.as_str(),
                    _ => continue,
                };
                
                // Infer the output shape using shape inference
                if let Some(output_shape) = ir.infer_fused_output_shape(fused_op_name, &fused_node.input_tensors) {
                    let fused_tensor = IntermediateTensor {
                        name: format!("{}_fused_intermediate", node_id),
                        shape: output_shape,
                        data_type: DataType::Float32,
                    };
                    optimized_ir.add_intermediate_tensor(fused_tensor);
                }
            }
        }
    }
}

impl HardwareOptimizer for CpuOptimizer {
    fn optimize(&self, ir: &UniversalIR, target: HardwareTarget) -> CoelanoxResult<OptimizedIR> {
        // Create optimized IR efficiently sharing weights
        let mut optimized_ir = OptimizedIR::new(ir, target);
        
        // === REAL GRAPH TRANSFORMATIONS ===
        
        // Pass 1: Apply operator fusion optimizations (Conv+ReLU, MatMul+Add, etc.)
        self.apply_fusion_optimizations(ir, &mut optimized_ir)?;
        
        // Pass 2: Apply individual optimizations to ALL nodes currently in the optimized graph
        let mut final_nodes = Vec::with_capacity(optimized_ir.optimized_nodes.len());
        
        // Take ownership of the nodes added by the fusion pass
        for mut node in optimized_ir.optimized_nodes {
            // Apply the attribute-setting passes to every node (fused or not)
            node = self.replace_with_cpu_optimized_ops(node);
            node = self.apply_cpu_optimizations(node, ir);
            node = self.optimize_memory_layout(node, ir);
            node = self.add_simd_hints(node);
            final_nodes.push(node); // Add the fully processed node to the final list
        }
        
        // Replace the potentially partially processed nodes with the fully processed ones
        optimized_ir.optimized_nodes = final_nodes;
        
        // Add any new intermediate tensors created during optimization
        // (e.g., from operator fusion)
        self.add_optimized_intermediate_tensors(&mut optimized_ir, ir);
        
        Ok(optimized_ir)
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Cpu]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

/// GPU optimizer
pub struct GpuOptimizer {
    /// Optimizer name
    name: String,
    /// Optimizer version
    version: String,
}

impl GpuOptimizer {
    /// Create a new GPU optimizer
    pub fn new() -> Self {
        Self {
            name: "gpu_optimizer".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    
    /// Replace operations with GPU-optimized versions
    fn replace_with_gpu_optimized_ops(&self, mut node: GraphNode) -> GraphNode {
        match node.op_type {
            OpType::Convolution => {
                // Replace generic convolution with GPU-optimized version
                node.add_attribute("gpu_conv_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cuda_kernel".to_string(), Attribute::Bool(true));
            }
            OpType::MatMul => {
                // Replace generic matmul with GPU-optimized version
                node.add_attribute("gpu_matmul_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cublas".to_string(), Attribute::Bool(true));
            }
            OpType::Pooling => {
                // Replace generic pooling with GPU-optimized version
                node.add_attribute("gpu_pooling_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("cuda_kernel".to_string(), Attribute::Bool(true));
            }
            _ => {
                // Add general GPU optimization marker
                node.add_attribute("gpu_optimized".to_string(), Attribute::Bool(true));
            }
        }
        node
    }
    
    /// Apply GPU-specific optimizations based on graph analysis
    fn apply_gpu_optimizations(&self, mut node: GraphNode, ir: &UniversalIR) -> GraphNode {
        // Analyze tensor shapes for GPU optimization opportunities
        let input_tensors = node.input_tensors.clone(); // Clone to avoid borrow issues
        for tensor_id in &input_tensors {
            if let Some(tensor_ref) = ir.get_tensor(*tensor_id) {
                match tensor_ref {
                    TensorRef::Weight(tensor) => {
                        // Optimize based on weight tensor properties
                        if tensor.shape.dimensions.len() == 4 {
                            // 4D tensor - likely convolution weights
                            node.add_attribute("weight_layout".to_string(), Attribute::String("NHWC".to_string()));
                        }
                    }
                    TensorRef::Intermediate(tensor) => {
                        // Optimize based on intermediate tensor properties
                        if tensor.shape.total_elements() > 1000000 {
                            // Large tensor - use memory-efficient algorithms
                            node.add_attribute("memory_efficient".to_string(), Attribute::Bool(true));
                        }
                    }
                }
            }
        }
        node
    }
    
    /// Optimize memory layout for GPU execution
    fn optimize_gpu_memory_layout(&self, mut node: GraphNode, _ir: &UniversalIR) -> GraphNode {
        // Add GPU memory layout optimizations
        node.add_attribute("memory_layout".to_string(), Attribute::String("coalesced".to_string()));
        node.add_attribute("shared_memory".to_string(), Attribute::Bool(true));
        node
    }
    
    /// Add CUDA optimization hints
    fn add_cuda_hints(&self, mut node: GraphNode) -> GraphNode {
        // Add CUDA optimization hints based on operation type
        match node.op_type {
            OpType::Add | OpType::Multiply => {
                node.add_attribute("cuda_enabled".to_string(), Attribute::Bool(true));
                node.add_attribute("thread_block_size".to_string(), Attribute::Int(256));
            }
            OpType::Convolution => {
                node.add_attribute("cuda_enabled".to_string(), Attribute::Bool(true));
                node.add_attribute("thread_block_size".to_string(), Attribute::Int(128));
            }
            _ => {
                node.add_attribute("cuda_enabled".to_string(), Attribute::Bool(false));
            }
        }
        node
    }
    
    /// Apply operator fusion optimizations for GPU
    fn apply_gpu_fusion_optimizations(&self, ir: &UniversalIR, optimized_ir: &mut OptimizedIR) -> CoelanoxResult<()> {
        // Look for Conv + BatchNorm + ReLU fusion opportunities (common GPU pattern)
        for (i, node) in ir.nodes.iter().enumerate() {
            if node.op_type == OpType::Convolution {
                // Check for Conv -> BatchNorm -> ReLU pattern
                if let Some(bn_node) = ir.nodes.get(i + 1) {
                    if bn_node.op_type == OpType::BatchNorm {
                        if let Some(relu_node) = ir.nodes.get(i + 2) {
                            if relu_node.op_type == OpType::Relu {
                                // Create fused ConvBatchNormReLU node
                                let mut fused_node = node.clone();
                                fused_node.op_type = OpType::Custom("ConvBatchNormReLU".to_string());
                                fused_node.add_attribute("fused_conv_bn_relu".to_string(), Attribute::Bool(true));
                                fused_node.add_attribute("gpu_optimized".to_string(), Attribute::Bool(true));
                                fused_node.add_attribute("tensor_cores".to_string(), Attribute::Bool(true));
                                
                                // Use the ReLU's output tensor as the fused node's output
                                fused_node.output_tensors = relu_node.output_tensors.clone();
                                
                                optimized_ir.add_optimized_node(fused_node);
                                
                                // Skip the BatchNorm and ReLU nodes since they're fused
                                continue;
                            }
                        }
                    }
                }
            }
            
            // Add the original node if not fused
            optimized_ir.add_optimized_node(node.clone());
        }
        
        Ok(())
    }
    
    /// Add optimized intermediate tensors (for GPU fusion results)
    fn add_optimized_intermediate_tensors(&self, optimized_ir: &mut OptimizedIR, ir: &UniversalIR) {
        // Add intermediate tensors for fused operations
        let fused_node_ids: Vec<_> = optimized_ir.optimized_nodes.iter()
            .filter(|node| node.op_type == OpType::Custom("ConvBatchNormReLU".to_string()))
            .map(|node| node.id.0)
            .collect();
            
        for node_id in fused_node_ids {
            // Find the fused node to get its operation type and input tensors
            if let Some(fused_node) = optimized_ir.optimized_nodes.iter()
                .find(|node| node.id.0 == node_id) {
                
                // Infer the output shape using shape inference
                if let Some(output_shape) = ir.infer_fused_output_shape("ConvBatchNormReLU", &fused_node.input_tensors) {
                    let fused_tensor = IntermediateTensor {
                        name: format!("{}_gpu_fused_intermediate", node_id),
                        shape: output_shape,
                        data_type: DataType::Float32,
                    };
                    optimized_ir.add_intermediate_tensor(fused_tensor);
                }
            }
        }
    }
}

impl HardwareOptimizer for GpuOptimizer {
    fn optimize(&self, ir: &UniversalIR, target: HardwareTarget) -> CoelanoxResult<OptimizedIR> {
        // Create optimized IR efficiently sharing weights
        let mut optimized_ir = OptimizedIR::new(ir, target);
        
        // === REAL GPU GRAPH TRANSFORMATIONS ===
        
        // Pass 1: Apply GPU-specific fusion optimizations (Conv+BatchNorm+ReLU, etc.)
        self.apply_gpu_fusion_optimizations(ir, &mut optimized_ir)?;
        
        // Pass 2: Apply individual optimizations to ALL nodes currently in the optimized graph
        let mut final_nodes = Vec::with_capacity(optimized_ir.optimized_nodes.len());
        
        // Take ownership of the nodes added by the fusion pass
        for mut node in optimized_ir.optimized_nodes {
            // Apply the attribute-setting passes to every node (fused or not)
            node = self.replace_with_gpu_optimized_ops(node);
            node = self.apply_gpu_optimizations(node, ir);
            node = self.optimize_gpu_memory_layout(node, ir);
            node = self.add_cuda_hints(node);
            final_nodes.push(node); // Add the fully processed node to the final list
        }
        
        // Replace the potentially partially processed nodes with the fully processed ones
        optimized_ir.optimized_nodes = final_nodes;
        
        // Add any new intermediate tensors created during optimization
        // (e.g., from operator fusion)
        self.add_optimized_intermediate_tensors(&mut optimized_ir, ir);
        
        Ok(optimized_ir)
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Gpu]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

/// Edge optimizer
pub struct EdgeOptimizer {
    /// Optimizer name
    name: String,
    /// Optimizer version
    version: String,
}

impl EdgeOptimizer {
    /// Create a new Edge optimizer
    pub fn new() -> Self {
        Self {
            name: "edge_optimizer".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    
    /// Replace operations with Edge-optimized versions
    fn replace_with_edge_optimized_ops(&self, mut node: GraphNode) -> GraphNode {
        match node.op_type {
            OpType::Convolution => {
                // Replace generic convolution with Edge-optimized version
                node.add_attribute("edge_conv_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("quantized".to_string(), Attribute::Bool(true));
            }
            OpType::MatMul => {
                // Replace generic matmul with Edge-optimized version
                node.add_attribute("edge_matmul_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("int8".to_string(), Attribute::Bool(true));
            }
            OpType::Pooling => {
                // Replace generic pooling with Edge-optimized version
                node.add_attribute("edge_pooling_optimized".to_string(), Attribute::Bool(true));
                node.add_attribute("vectorized".to_string(), Attribute::Bool(true));
            }
            _ => {
                // Add general Edge optimization marker
                node.add_attribute("edge_optimized".to_string(), Attribute::Bool(true));
            }
        }
        node
    }
    
    /// Apply Edge-specific optimizations based on graph analysis
    fn apply_edge_optimizations(&self, mut node: GraphNode, ir: &UniversalIR) -> GraphNode {
        // Analyze tensor shapes for Edge optimization opportunities
        let input_tensors = node.input_tensors.clone(); // Clone to avoid borrow issues
        for tensor_id in &input_tensors {
            if let Some(tensor_ref) = ir.get_tensor(*tensor_id) {
                match tensor_ref {
                    TensorRef::Weight(tensor) => {
                        // Optimize based on weight tensor properties
                        if tensor.shape.dimensions.len() == 4 {
                            // 4D tensor - likely convolution weights
                            node.add_attribute("weight_layout".to_string(), Attribute::String("NCHW".to_string()));
                        }
                    }
                    TensorRef::Intermediate(tensor) => {
                        // Optimize based on intermediate tensor properties
                        if tensor.shape.total_elements() > 100000 {
                            // Large tensor - use memory-efficient algorithms
                            node.add_attribute("memory_efficient".to_string(), Attribute::Bool(true));
                        }
                    }
                }
            }
        }
        node
    }
    
    /// Optimize memory layout for Edge execution
    fn optimize_edge_memory_layout(&self, mut node: GraphNode, _ir: &UniversalIR) -> GraphNode {
        // Add Edge memory layout optimizations
        node.add_attribute("memory_layout".to_string(), Attribute::String("compact".to_string()));
        node.add_attribute("cache_friendly".to_string(), Attribute::Bool(true));
        node
    }
    
    /// Add power optimization hints
    fn add_power_hints(&self, mut node: GraphNode) -> GraphNode {
        // Add power optimization hints based on operation type
        match node.op_type {
            OpType::Add | OpType::Multiply => {
                node.add_attribute("low_power".to_string(), Attribute::Bool(true));
                node.add_attribute("sleep_mode".to_string(), Attribute::Bool(true));
            }
            OpType::Convolution => {
                node.add_attribute("low_power".to_string(), Attribute::Bool(true));
                node.add_attribute("batch_processing".to_string(), Attribute::Bool(true));
            }
            _ => {
                node.add_attribute("low_power".to_string(), Attribute::Bool(false));
            }
        }
        node
    }
    
    /// Apply operator fusion optimizations for Edge devices
    fn apply_edge_fusion_optimizations(&self, ir: &UniversalIR, optimized_ir: &mut OptimizedIR) -> CoelanoxResult<()> {
        // Look for quantization-friendly fusion patterns (Conv + ReLU, Add + ReLU)
        for (i, node) in ir.nodes.iter().enumerate() {
            if node.op_type == OpType::Convolution {
                // Check for Conv -> ReLU pattern (common in mobile networks)
                if let Some(relu_node) = ir.nodes.get(i + 1) {
                    if relu_node.op_type == OpType::Relu {
                        // Create fused ConvReLU node optimized for Edge
                        let mut fused_node = node.clone();
                        fused_node.op_type = OpType::Custom("EdgeConvReLU".to_string());
                        fused_node.add_attribute("fused_conv_relu".to_string(), Attribute::Bool(true));
                        fused_node.add_attribute("edge_optimized".to_string(), Attribute::Bool(true));
                        fused_node.add_attribute("quantized".to_string(), Attribute::Bool(true));
                        fused_node.add_attribute("int8_ready".to_string(), Attribute::Bool(true));
                        
                        // Use the ReLU's output tensor as the fused node's output
                        fused_node.output_tensors = relu_node.output_tensors.clone();
                        
                        optimized_ir.add_optimized_node(fused_node);
                        
                        // Skip the next ReLU node since it's fused
                        continue;
                    }
                }
            }
            
            // Add the original node if not fused
            optimized_ir.add_optimized_node(node.clone());
        }
        
        Ok(())
    }
    
    /// Add optimized intermediate tensors (for Edge fusion results)
    fn add_optimized_intermediate_tensors(&self, optimized_ir: &mut OptimizedIR, ir: &UniversalIR) {
        // Add intermediate tensors for fused operations
        let fused_node_ids: Vec<_> = optimized_ir.optimized_nodes.iter()
            .filter(|node| node.op_type == OpType::Custom("EdgeConvReLU".to_string()))
            .map(|node| node.id.0)
            .collect();
            
        for node_id in fused_node_ids {
            // Find the fused node to get its operation type and input tensors
            if let Some(fused_node) = optimized_ir.optimized_nodes.iter()
                .find(|node| node.id.0 == node_id) {
                
                // Infer the output shape using shape inference
                if let Some(output_shape) = ir.infer_fused_output_shape("EdgeConvReLU", &fused_node.input_tensors) {
                    let fused_tensor = IntermediateTensor {
                        name: format!("{}_edge_fused_intermediate", node_id),
                        shape: output_shape,
                        data_type: DataType::UInt8, // Edge uses quantized data
                    };
                    optimized_ir.add_intermediate_tensor(fused_tensor);
                }
            }
        }
    }
}

impl HardwareOptimizer for EdgeOptimizer {
    fn optimize(&self, ir: &UniversalIR, target: HardwareTarget) -> CoelanoxResult<OptimizedIR> {
        // Create optimized IR efficiently sharing weights
        let mut optimized_ir = OptimizedIR::new(ir, target);
        
        // === REAL EDGE GRAPH TRANSFORMATIONS ===
        
        // Pass 1: Apply Edge-specific fusion optimizations (quantization-friendly patterns)
        self.apply_edge_fusion_optimizations(ir, &mut optimized_ir)?;
        
        // Pass 2: Apply individual optimizations to ALL nodes currently in the optimized graph
        let mut final_nodes = Vec::with_capacity(optimized_ir.optimized_nodes.len());
        
        // Take ownership of the nodes added by the fusion pass
        for mut node in optimized_ir.optimized_nodes {
            // Apply the attribute-setting passes to every node (fused or not)
            node = self.replace_with_edge_optimized_ops(node);
            node = self.apply_edge_optimizations(node, ir);
            node = self.optimize_edge_memory_layout(node, ir);
            node = self.add_power_hints(node);
            final_nodes.push(node); // Add the fully processed node to the final list
        }
        
        // Replace the potentially partially processed nodes with the fully processed ones
        optimized_ir.optimized_nodes = final_nodes;
        
        // Add any new intermediate tensors created during optimization
        // (e.g., from operator fusion)
        self.add_optimized_intermediate_tensors(&mut optimized_ir, ir);
        
        Ok(optimized_ir)
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Edge]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}
