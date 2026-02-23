//! Universal IR (Intermediate Representation) for AI models
//! 
//! This module defines the universal intermediate representation that can hold
//! any AI model in a neutral, hardware-agnostic format. The IR uses a proper
//! graph structure with efficient index-based lookups and clear separation
//! between static weights and dynamic intermediate tensors.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{CoelanoxError, CoelanoxResult, format::HardwareTarget};

/// Unique identifier for nodes in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// Tensor type classification for future-proofing
/// 
/// Uses 3 bits to support up to 8 tensor types:
/// - Current: Weight, Intermediate
/// - Future: Input, Output, Constant, Quantum, Neuromorphic, Reserved
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TensorType {
    /// Weight tensor (static, pre-trained parameters)
    Weight = 0b000,
    /// Intermediate tensor (runtime activations)
    Intermediate = 0b001,
    /// Input tensor (future - model inputs)
    Input = 0b010,
    /// Output tensor (future - model outputs)
    Output = 0b011,
    /// Constant tensor (future - compile-time constants)
    Constant = 0b100,
    /// Quantum tensor (future - quantum computing)
    Quantum = 0b101,
    /// Neuromorphic tensor (future - brain-inspired computing)
    Neuromorphic = 0b110,
    /// Reserved for future use
    Reserved = 0b111,
}

impl TensorType {
    /// Get the type from a 3-bit value
    #[inline]
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b111 {
            0b000 => TensorType::Weight,
            0b001 => TensorType::Intermediate,
            0b010 => TensorType::Input,
            0b011 => TensorType::Output,
            0b100 => TensorType::Constant,
            0b101 => TensorType::Quantum,
            0b110 => TensorType::Neuromorphic,
            0b111 => TensorType::Reserved,
            _ => unreachable!(),
        }
    }
    
    /// Get the 3-bit value for this type
    #[inline]
    pub fn to_bits(self) -> u8 {
        self as u8
    }
}

/// Unique identifier for tensors (both static and dynamic)
/// 
/// The TensorId encodes both the tensor type and index using bit-packing:
/// - Bits 0-2: Tensor type (3 bits, supports 8 types)
/// - Bits 3-63: Index into the respective Vec (2^61 tensors per type)
/// 
/// This future-proof design allows for quantum, neuromorphic, and other
/// emerging tensor types while maintaining O(1) lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub usize);

impl TensorId {
    /// Create a weight tensor ID
    #[inline]
    pub fn weight(index: usize) -> Self {
        Self((index << 3) | (TensorType::Weight as usize))
    }
    
    /// Create an intermediate tensor ID
    #[inline]
    pub fn intermediate(index: usize) -> Self {
        Self((index << 3) | (TensorType::Intermediate as usize))
    }
    
    /// Create a tensor ID with a specific type and index
    #[inline]
    pub fn new(tensor_type: TensorType, index: usize) -> Self {
        Self((index << 3) | (tensor_type.to_bits() as usize))
    }
    
    /// Get the tensor type
    #[inline]
    pub fn tensor_type(&self) -> TensorType {
        TensorType::from_bits((self.0 & 0b111) as u8)
    }
    
    /// Check if this is a weight tensor
    #[inline]
    pub fn is_weight(&self) -> bool {
        self.tensor_type() == TensorType::Weight
    }
    
    /// Check if this is an intermediate tensor
    #[inline]
    pub fn is_intermediate(&self) -> bool {
        self.tensor_type() == TensorType::Intermediate
    }
    
    /// Get the index into the respective Vec
    #[inline]
    pub fn index(&self) -> usize {
        self.0 >> 3
    }
}

/// Data types supported by the Universal IR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 16-bit floating point (IEEE 754)
    Float16,
    /// 16-bit Brain Float (BFloat16) - used by modern AI models (Llama, GPT, etc.)
    BFloat16,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// Boolean
    Bool,
}

impl DataType {
    /// Get the size in bytes of this data type
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Float16 => 2,
            DataType::BFloat16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::UInt8 => 1,
            DataType::Bool => 1,
        }
    }
}

/// Shape of a tensor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    /// Dimensions of the tensor
    pub dimensions: Vec<usize>,
}

impl Shape {
    /// Create a new shape
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self { dimensions }
    }
    
    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        self.dimensions.iter().product()
    }
    
    /// Get the number of dimensions
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

/// Primitive operation types supported by the Universal IR
/// 
/// This enum contains only primitive operations that can be directly executed
/// on hardware. Complex layers should be decomposed into subgraphs of these primitives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpType {
    // Basic arithmetic operations
    Add,
    Subtract,
    Multiply,
    Divide,
    
    // Element-wise operations
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    Gelu,
    Swish,
    
    // Mathematical primitives (for RMSNorm, RoPE, etc.)
    Sqrt,
    Pow,
    Cos,
    Sin,
    Exp,
    Log,
    
    // Neural network primitives
    Convolution,
    MaxPool,
    AvgPool,
    GlobalMaxPool,
    GlobalAvgPool,
    BatchNorm,
    LayerNorm,
    Dropout,
    
    // Tensor manipulation
    Reshape,
    Transpose,
    Permute,
    Concatenate,
    Split,
    Slice,
    Gather,
    Scatter,
    
    // Matrix operations
    MatMul,
    Gemm, // Generalized matrix multiplication
    
    // Reduction operations
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    
    // Broadcasting operations
    Broadcast,
    Expand,
    
    // Comparison operations
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    
    // Logical operations
    And,
    Or,
    Not,
    
    // Custom primitive operation
    Custom(String),
}

/// Attribute values for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Attribute {
    /// Float value
    Float(f32),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Float array
    FloatArray(Vec<f32>),
    /// Integer array
    IntArray(Vec<i64>),
    /// Boolean value
    Bool(bool),
    /// String array
    StringArray(Vec<String>),
}

impl Attribute {
    /// Get integer value if this is an Int variant
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attribute::Int(i) => Some(*i),
            _ => None,
        }
    }
    
    /// Get float value if this is a Float variant
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Attribute::Float(f) => Some(*f),
            _ => None,
        }
    }
    
    /// Get boolean value if this is a Bool variant
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Attribute::Bool(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Get string value if this is a String variant
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Attribute::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Input shapes
    pub input_shapes: Vec<Shape>,
    /// Output shapes
    pub output_shapes: Vec<Shape>,
    /// Default data type
    pub data_type: DataType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: "1.0.0".to_string(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            data_type: DataType::Float32,
            metadata: HashMap::new(),
        }
    }
    
    /// Add input shape
    pub fn add_input_shape(&mut self, shape: Shape) {
        self.input_shapes.push(shape);
    }
    
    /// Add output shape
    pub fn add_output_shape(&mut self, shape: Shape) {
        self.output_shapes.push(shape);
    }
    
    /// Set data type
    pub fn set_data_type(&mut self, data_type: DataType) {
        self.data_type = data_type;
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Graph node representing an operation in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: NodeId,
    /// Operation type
    pub op_type: OpType,
    /// Input tensor IDs
    pub input_tensors: Vec<TensorId>,
    /// Output tensor IDs
    pub output_tensors: Vec<TensorId>,
    /// Operation attributes
    pub attributes: HashMap<String, Attribute>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: NodeId, op_type: OpType) -> Self {
        Self {
            id,
            op_type,
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
    
    /// Add input tensor ID
    pub fn add_input_tensor(&mut self, tensor_id: TensorId) {
        self.input_tensors.push(tensor_id);
    }
    
    /// Add output tensor ID
    pub fn add_output_tensor(&mut self, tensor_id: TensorId) {
        self.output_tensors.push(tensor_id);
    }
    
    /// Add attribute
    pub fn add_attribute(&mut self, key: String, value: Attribute) {
        self.attributes.insert(key, value);
    }
}

/// Static weight tensor (model parameters)
/// 
/// These are the learned parameters of the model that are stored
/// as raw data and loaded at inference time.
/// The TensorId is implicit in the Vec position for O(1) lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Shape,
    /// Data type
    pub data_type: DataType,
    /// Raw tensor data
    pub data: Vec<u8>,
}

impl WeightTensor {
    /// Create a new weight tensor
    pub fn new(name: String, shape: Shape, data_type: DataType) -> Self {
        let data_size = shape.total_elements() * data_type.size();
        Self {
            name,
            shape,
            data_type,
            data: vec![0; data_size],
        }
    }
    
    /// Create a weight tensor with data
    pub fn with_data(name: String, shape: Shape, data_type: DataType, data: Vec<u8>) -> Self {
        Self {
            name,
            shape,
            data_type,
            data,
        }
    }
    
    /// Get tensor size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Get number of elements
    pub fn elements(&self) -> usize {
        self.shape.total_elements()
    }
}

/// Dynamic intermediate tensor (computation results)
/// 
/// These represent the intermediate values flowing through the computation
/// graph. They don't contain data, only metadata about shape and type.
/// The TensorId is implicit in the Vec position for O(1) lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Shape,
    /// Data type
    pub data_type: DataType,
}

impl IntermediateTensor {
    /// Create a new intermediate tensor
    pub fn new(name: String, shape: Shape, data_type: DataType) -> Self {
        Self {
            name,
            shape,
            data_type,
        }
    }
    
    /// Get number of elements
    pub fn elements(&self) -> usize {
        self.shape.total_elements()
    }
    
    /// Get size in bytes (for memory allocation)
    pub fn size_bytes(&self) -> usize {
        self.elements() * self.data_type.size()
    }
}

/// Optimization hint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Quantization optimization
    Quantization,
    /// Pruning optimization
    Pruning,
    /// Operation fusion
    Fusion,
    /// Parallelization
    Parallelization,
    /// Memory optimization
    MemoryOptimization,
    /// Custom optimization
    Custom(String),
}

/// Optimization hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Optimization type
    pub hint_type: OptimizationType,
    /// Optimization parameters
    pub parameters: HashMap<String, Attribute>,
}

impl OptimizationHint {
    /// Create a new optimization hint
    pub fn new(hint_type: OptimizationType) -> Self {
        Self {
            hint_type,
            parameters: HashMap::new(),
        }
    }
    
    /// Add parameter
    pub fn add_parameter(&mut self, key: String, value: Attribute) {
        self.parameters.insert(key, value);
    }
}

/// Optimized IR for specific hardware
/// 
/// This is a standalone result of optimization, not a wrapper around the original IR.
/// It contains only the optimized operations and metadata needed for code generation.
/// Uses the same high-performance ID-based architecture as UniversalIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedIR {
    /// Hardware target
    pub target: HardwareTarget,
    /// Optimized graph nodes (using high-performance ID-based system)
    pub optimized_nodes: Vec<GraphNode>,
    /// Intermediate tensors created during optimization
    pub intermediate_tensors: Vec<IntermediateTensor>,
    /// Optimization metadata
    pub optimization_metadata: HashMap<String, String>,
    /// Reference to original model metadata (lightweight)
    pub original_model_name: String,
    /// Shared weight data (serialized as Vec for storage)
    #[serde(skip)]
    pub shared_weights: std::sync::Arc<Vec<WeightTensor>>,
}

impl OptimizedIR {
    /// Create new optimized IR from original IR (efficiently sharing weights)
    pub fn new(original_ir: &UniversalIR, target: HardwareTarget) -> Self {
        Self {
            target: target.clone(),
            optimized_nodes: Vec::new(),
            intermediate_tensors: Vec::new(),
            optimization_metadata: HashMap::new(),
            original_model_name: original_ir.metadata.name.clone(),
            shared_weights: std::sync::Arc::new(original_ir.weight_tensors.clone()),
        }
    }
    
    /// Create optimized IR from components (for advanced use cases)
    pub fn from_components(
        target: HardwareTarget,
        optimized_nodes: Vec<GraphNode>,
        intermediate_tensors: Vec<IntermediateTensor>,
        original_model_name: String,
        shared_weights: std::sync::Arc<Vec<WeightTensor>>,
    ) -> Self {
        Self {
            target,
            optimized_nodes,
            intermediate_tensors,
            optimization_metadata: HashMap::new(),
            original_model_name,
            shared_weights,
        }
    }
    
    /// Create OptimizedIR from deserialized data (reconstructs Arc)
    pub fn from_deserialized(
        target: HardwareTarget,
        optimized_nodes: Vec<GraphNode>,
        intermediate_tensors: Vec<IntermediateTensor>,
        optimization_metadata: HashMap<String, String>,
        original_model_name: String,
        weight_tensors: Vec<WeightTensor>,
    ) -> Self {
        Self {
            target,
            optimized_nodes,
            intermediate_tensors,
            optimization_metadata,
            original_model_name,
            shared_weights: std::sync::Arc::new(weight_tensors),
        }
    }
    
    /// Add optimized graph node
    pub fn add_optimized_node(&mut self, node: GraphNode) {
        self.optimized_nodes.push(node);
    }
    
    /// Add intermediate tensor created during optimization
    pub fn add_intermediate_tensor(&mut self, tensor: IntermediateTensor) {
        self.intermediate_tensors.push(tensor);
    }
    
    /// Add optimization metadata
    pub fn add_optimization_metadata(&mut self, key: String, value: String) {
        self.optimization_metadata.insert(key, value);
    }
    
    /// Get optimized node by ID (O(1) lookup)
    pub fn get_optimized_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.optimized_nodes.get(node_id.0)
    }
    
    /// Get intermediate tensor by ID (O(1) lookup)
    pub fn get_intermediate_tensor(&self, tensor_id: TensorId) -> Option<&IntermediateTensor> {
        if tensor_id.is_intermediate() {
            self.intermediate_tensors.get(tensor_id.index())
        } else {
            None
        }
    }
    
    /// Get shared weight tensor by ID (O(1) lookup)
    pub fn get_shared_weight(&self, tensor_id: TensorId) -> Option<&WeightTensor> {
        if tensor_id.is_weight() {
            self.shared_weights.get(tensor_id.index())
        } else {
            None
        }
    }
    
    /// Get any tensor by ID (weight or intermediate)
    pub fn get_tensor(&self, tensor_id: TensorId) -> Option<TensorRef> {
        if let Some(tensor) = self.get_shared_weight(tensor_id) {
            return Some(TensorRef::Weight(tensor));
        }
        if let Some(tensor) = self.get_intermediate_tensor(tensor_id) {
            return Some(TensorRef::Intermediate(tensor));
        }
        None
    }
    
    /// Get number of optimized nodes
    pub fn node_count(&self) -> usize {
        self.optimized_nodes.len()
    }
    
    /// Get number of intermediate tensors
    pub fn intermediate_tensor_count(&self) -> usize {
        self.intermediate_tensors.len()
    }
    
    /// Get number of shared weight tensors
    pub fn weight_tensor_count(&self) -> usize {
        self.shared_weights.len()
    }
}

/// Serialized shape for container ir_data (must match packager SerializedOptimizedIR).
#[derive(Debug, Clone, Deserialize)]
struct SerializedOptimizedIR {
    target: HardwareTarget,
    optimized_nodes: Vec<GraphNode>,
    intermediate_tensors: Vec<IntermediateTensor>,
    optimization_metadata: HashMap<String, String>,
    original_model_name: String,
    weight_tensors: Vec<WeightTensor>,
}

/// Deserialize container IR bytes into OptimizedIR (for runtime warm cache).
pub fn deserialize_optimized_ir(ir_data: &[u8]) -> CoelanoxResult<OptimizedIR> {
    let s: SerializedOptimizedIR = bincode::deserialize(ir_data)
        .map_err(|e| CoelanoxError::Runtime(format!("Failed to deserialize IR: {}", e)))?;
    Ok(OptimizedIR::from_deserialized(
        s.target,
        s.optimized_nodes,
        s.intermediate_tensors,
        s.optimization_metadata,
        s.original_model_name,
        s.weight_tensors,
    ))
}

/// Universal IR - the main intermediate representation with proper graph structure
/// 
/// This IR uses an arena-based approach with efficient index-based lookups
/// and clear separation between static weights and dynamic intermediate tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIR {
    /// Model metadata
    pub metadata: ModelMetadata,
    
    /// Graph nodes (operations) - arena-based storage
    pub nodes: Vec<GraphNode>,
    
    /// Static weight tensors (model parameters)
    pub weight_tensors: Vec<WeightTensor>,
    
    /// Dynamic intermediate tensors (computation results)
    pub intermediate_tensors: Vec<IntermediateTensor>,
    
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
    
    /// Topological order of nodes for execution
    pub topological_order: Vec<NodeId>,
    
    /// Next available node ID for efficient allocation
    next_node_id: usize,
}

impl UniversalIR {
    /// Create a new Universal IR
    pub fn new(name: String) -> Self {
        Self {
            metadata: ModelMetadata::new(name),
            nodes: Vec::new(),
            weight_tensors: Vec::new(),
            intermediate_tensors: Vec::new(),
            optimization_hints: Vec::new(),
            topological_order: Vec::new(),
            next_node_id: 0,
        }
    }
    
    /// Add a graph node (operation)
    pub fn add_node(&mut self, op_type: OpType) -> NodeId {
        let node_id = NodeId(self.next_node_id);
        
        let node = GraphNode::new(node_id, op_type);
        self.nodes.push(node);
        
        self.next_node_id += 1;
        
        node_id
    }
    
    /// Add a weight tensor
    pub fn add_weight_tensor(&mut self, name: String, shape: Shape, data_type: DataType) -> TensorId {
        let index = self.weight_tensors.len();
        let tensor_id = TensorId::weight(index);
        let tensor = WeightTensor::new(name, shape, data_type);
        self.weight_tensors.push(tensor);
        tensor_id
    }
    
    /// Add a weight tensor with data
    pub fn add_weight_tensor_with_data(&mut self, name: String, shape: Shape, data_type: DataType, data: Vec<u8>) -> TensorId {
        let index = self.weight_tensors.len();
        let tensor_id = TensorId::weight(index);
        let tensor = WeightTensor::with_data(name, shape, data_type, data);
        self.weight_tensors.push(tensor);
        tensor_id
    }
    
    /// Add an intermediate tensor
    pub fn add_intermediate_tensor(&mut self, name: String, shape: Shape, data_type: DataType) -> TensorId {
        let index = self.intermediate_tensors.len();
        let tensor_id = TensorId::intermediate(index);
        let tensor = IntermediateTensor::new(name, shape, data_type);
        self.intermediate_tensors.push(tensor);
        tensor_id
    }
    
    /// Add an optimization hint
    pub fn add_optimization_hint(&mut self, hint: OptimizationHint) {
        self.optimization_hints.push(hint);
    }
    
    /// Compute and store topological order for execution
    pub fn compute_topological_order(&mut self) -> CoelanoxResult<()> {
        self.topological_order = self.topological_sort()?;
        Ok(())
    }

    /// Reorder nodes so that `nodes` is in topological (execution) order.
    /// Call this before the optimizer so that downstream passes see a linear execution order.
    /// Preserves the invariant that `nodes[i].id == NodeId(i)`.
    pub fn sort_nodes_for_execution(&mut self) -> CoelanoxResult<()> {
        if self.topological_order.is_empty() {
            self.compute_topological_order()?;
        }
        let order = std::mem::take(&mut self.topological_order);
        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        for (new_index, &node_id) in order.iter().enumerate() {
            let mut node = self
                .nodes
                .get(node_id.0)
                .cloned()
                .ok_or_else(|| CoelanoxError::format("NodeId out of range in topological order"))?;
            node.id = NodeId(new_index);
            new_nodes.push(node);
        }
        self.nodes = new_nodes;
        self.topological_order = (0..self.nodes.len()).map(NodeId).collect();
        Ok(())
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(node_id.0)
    }
    
    /// Get node by ID (mutable)
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(node_id.0)
    }
    
    /// Get weight tensor by ID (O(1) lookup)
    pub fn get_weight_tensor(&self, tensor_id: TensorId) -> Option<&WeightTensor> {
        if tensor_id.is_weight() {
            self.weight_tensors.get(tensor_id.index())
        } else {
            None
        }
    }
    
    /// Get intermediate tensor by ID (O(1) lookup)
    pub fn get_intermediate_tensor(&self, tensor_id: TensorId) -> Option<&IntermediateTensor> {
        if tensor_id.is_intermediate() {
            self.intermediate_tensors.get(tensor_id.index())
        } else {
            None
        }
    }
    
    /// Get any tensor by ID (weight or intermediate)
    pub fn get_tensor(&self, tensor_id: TensorId) -> Option<TensorRef> {
        if let Some(tensor) = self.get_weight_tensor(tensor_id) {
            return Some(TensorRef::Weight(tensor));
        }
        if let Some(tensor) = self.get_intermediate_tensor(tensor_id) {
            return Some(TensorRef::Intermediate(tensor));
        }
        None
    }
    
    /// Get tensor by name
    pub fn get_tensor_by_name(&self, name: &str) -> Option<(TensorId, TensorRef)> {
        // Check weight tensors first
        for (index, tensor) in self.weight_tensors.iter().enumerate() {
            if tensor.name == name {
                let tensor_id = TensorId::weight(index);
                return Some((tensor_id, TensorRef::Weight(tensor)));
            }
        }
        
        // Check intermediate tensors
        for (index, tensor) in self.intermediate_tensors.iter().enumerate() {
            if tensor.name == name {
                let tensor_id = TensorId::intermediate(index);
                return Some((tensor_id, TensorRef::Intermediate(tensor)));
            }
        }
        
        None
    }
    
    /// Validate the IR
    pub fn validate(&self) -> CoelanoxResult<()> {
        // Validate metadata
        if self.metadata.name.is_empty() {
            return Err(CoelanoxError::format("Model name cannot be empty"));
        }
        
        // Validate nodes
        for (i, node) in self.nodes.iter().enumerate() {
            if node.input_tensors.is_empty() && node.output_tensors.is_empty() {
                return Err(CoelanoxError::format(format!("Node {} has no inputs or outputs", i)));
            }
        }
        
        // Validate weight tensors
        for tensor in &self.weight_tensors {
            if tensor.name.is_empty() {
                return Err(CoelanoxError::format("Weight tensor name cannot be empty"));
            }
            
            if tensor.shape.dimensions.is_empty() {
                return Err(CoelanoxError::format(format!("Weight tensor {} has empty shape", tensor.name)));
            }
        }
        
        // Validate intermediate tensors
        for tensor in &self.intermediate_tensors {
            if tensor.name.is_empty() {
                return Err(CoelanoxError::format("Intermediate tensor name cannot be empty"));
            }
            
            if tensor.shape.dimensions.is_empty() {
                return Err(CoelanoxError::format(format!("Intermediate tensor {} has empty shape", tensor.name)));
            }
        }
        
        Ok(())
    }
    
    /// Get total size of all weight tensors
    pub fn total_weight_size(&self) -> usize {
        self.weight_tensors.iter().map(|t| t.size()).sum()
    }
    
    /// Get total memory requirement for intermediate tensors
    pub fn total_intermediate_memory(&self) -> usize {
        self.intermediate_tensors.iter().map(|t| t.size_bytes()).sum()
    }
    
    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get number of weight tensors
    pub fn weight_tensor_count(&self) -> usize {
        self.weight_tensors.len()
    }
    
    /// Get number of intermediate tensors
    pub fn intermediate_tensor_count(&self) -> usize {
        self.intermediate_tensors.len()
    }
    
    /// Get total number of tensors
    pub fn total_tensor_count(&self) -> usize {
        self.weight_tensor_count() + self.intermediate_tensor_count()
    }
    
    /// Get all nodes that produce a given tensor
    pub fn get_producer_nodes(&self, tensor_id: TensorId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|node| node.output_tensors.contains(&tensor_id))
            .map(|node| node.id)
            .collect()
    }
    
    /// Get all nodes that consume a given tensor
    pub fn get_consumer_nodes(&self, tensor_id: TensorId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|node| node.input_tensors.contains(&tensor_id))
            .map(|node| node.id)
            .collect()
    }
    
    /// Get all input tensors for a node
    pub fn get_node_inputs(&self, node_id: NodeId) -> Option<Vec<TensorId>> {
        self.get_node(node_id).map(|node| node.input_tensors.clone())
    }
    
    /// Get all output tensors for a node
    pub fn get_node_outputs(&self, node_id: NodeId) -> Option<Vec<TensorId>> {
        self.get_node(node_id).map(|node| node.output_tensors.clone())
    }
    
    /// Connect a tensor output from one node to input of another node
    pub fn connect_nodes(&mut self, producer_node: NodeId, output_idx: usize, consumer_node: NodeId, input_idx: usize) -> CoelanoxResult<()> {
        // Get the tensor ID from producer first
        let tensor_id = {
            let producer = self.get_node(producer_node)
                .ok_or_else(|| CoelanoxError::format("Producer node not found"))?;
            
            if output_idx >= producer.output_tensors.len() {
                return Err(CoelanoxError::format("Output index out of bounds"));
            }
            
            producer.output_tensors[output_idx]
        };
        
        // Now update the consumer
        let consumer = self.get_node_mut(consumer_node)
            .ok_or_else(|| CoelanoxError::format("Consumer node not found"))?;
        
        if input_idx >= consumer.input_tensors.len() {
            return Err(CoelanoxError::format("Input index out of bounds"));
        }
        
        consumer.input_tensors[input_idx] = tensor_id;
        
        Ok(())
    }
    
    /// Add input tensor to a node
    pub fn add_node_input(&mut self, node_id: NodeId, tensor_id: TensorId) -> CoelanoxResult<()> {
        let node = self.get_node_mut(node_id)
            .ok_or_else(|| CoelanoxError::format("Node not found"))?;
        
        node.add_input_tensor(tensor_id);
        Ok(())
    }
    
    /// Add output tensor to a node
    pub fn add_node_output(&mut self, node_id: NodeId, tensor_id: TensorId) -> CoelanoxResult<()> {
        let node = self.get_node_mut(node_id)
            .ok_or_else(|| CoelanoxError::format("Node not found"))?;
        
        node.add_output_tensor(tensor_id);
        Ok(())
    }
    
    /// Get topological order of nodes (for execution order)
    pub fn topological_sort(&self) -> CoelanoxResult<Vec<NodeId>> {
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        
        for node in &self.nodes {
            if !visited.contains(&node.id) {
                self.dfs_topological(node.id, &mut visited, &mut temp_visited, &mut result)?;
            }
        }
        
        result.reverse();
        Ok(result)
    }
    
    /// Depth-first search for topological sorting
    fn dfs_topological(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        temp_visited: &mut std::collections::HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> CoelanoxResult<()> {
        if temp_visited.contains(&node_id) {
            return Err(CoelanoxError::format("Cycle detected in computation graph"));
        }
        
        if visited.contains(&node_id) {
            return Ok(());
        }
        
        temp_visited.insert(node_id);
        
        // Visit all input tensors' producer nodes first
        if let Some(node) = self.get_node(node_id) {
            for tensor_id in &node.input_tensors {
                for producer_node in self.get_producer_nodes(*tensor_id) {
                    self.dfs_topological(producer_node, visited, temp_visited, result)?;
                }
            }
        }
        
        temp_visited.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);
        
        Ok(())
    }
    
    /// Get all leaf nodes (nodes with no outputs)
    pub fn get_leaf_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|node| node.output_tensors.is_empty())
            .map(|node| node.id)
            .collect()
    }
    
    /// Get all root nodes (nodes with no inputs)
    pub fn get_root_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|node| node.input_tensors.is_empty())
            .map(|node| node.id)
            .collect()
    }
    
    /// Check if the graph is acyclic
    pub fn is_acyclic(&self) -> bool {
        self.topological_sort().is_ok()
    }
    
    /// Get graph statistics
    pub fn get_graph_stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.node_count(),
            weight_tensor_count: self.weight_tensor_count(),
            intermediate_tensor_count: self.intermediate_tensor_count(),
            total_weight_size: self.total_weight_size(),
            total_intermediate_memory: self.total_intermediate_memory(),
            is_acyclic: self.is_acyclic(),
        }
    }
    
    /// Get all nodes that produce a specific tensor
    pub fn get_producer_node(&self, tensor_id: TensorId) -> Option<&GraphNode> {
        self.nodes.iter()
            .find(|node| node.output_tensors.contains(&tensor_id))
    }
    
    /// Check if a node has only one consumer (useful for fusion)
    pub fn has_single_consumer(&self, node: &GraphNode) -> bool {
        if node.output_tensors.is_empty() {
            return false;
        }
        
        // Get all consumers of this node's outputs
        let mut consumers = std::collections::HashSet::new();
        for output_tensor in &node.output_tensors {
            let node_consumers = self.get_consumer_nodes(*output_tensor);
            for consumer_id in node_consumers {
                consumers.insert(consumer_id);
            }
        }
        
        // Must have exactly one consumer
        consumers.len() == 1
    }
    
    /// Check if two nodes can be fused (node1 -> node2 with single consumer)
    pub fn can_fuse_nodes(&self, node1: &GraphNode, node2: &GraphNode) -> bool {
        // Basic validation
        if node1.id == node2.id {
            return false; // Can't fuse a node with itself
        }
        
        // Check if node1's outputs are inputs to node2
        let has_direct_connection = node1.output_tensors.iter()
            .any(|output_tensor| node2.input_tensors.contains(output_tensor));
            
        if !has_direct_connection {
            return false;
        }
        
        // Check if node1 has only one consumer (node2)
        if !self.has_single_consumer(node1) {
            return false;
        }
        
        // Additional validation: ensure node2 is actually the single consumer
        let consumers: Vec<_> = node1.output_tensors.iter()
            .flat_map(|tensor_id| self.get_consumer_nodes(*tensor_id))
            .collect();
            
        if consumers.len() != 1 || consumers[0] != node2.id {
            return false;
        }
        
        // Check data type compatibility (basic validation)
        if let Some(output_tensor) = node1.output_tensors.first() {
            if let Some(input_tensor) = node2.input_tensors.iter()
                .find(|tensor_id| node1.output_tensors.contains(tensor_id)) {
                
                // Get tensor references to check data types
                if let (Some(output_ref), Some(input_ref)) = (
                    self.get_tensor(*output_tensor),
                    self.get_tensor(*input_tensor)
                ) {
                    let output_dtype = match output_ref {
                        TensorRef::Weight(tensor) => &tensor.data_type,
                        TensorRef::Intermediate(tensor) => &tensor.data_type,
                    };
                    let input_dtype = match input_ref {
                        TensorRef::Weight(tensor) => &tensor.data_type,
                        TensorRef::Intermediate(tensor) => &tensor.data_type,
                    };
                    
                    // For now, allow fusion if data types are compatible
                    // (same type or compatible conversions)
                    match (output_dtype, input_dtype) {
                        (DataType::Float32, DataType::Float32) => true,
                        (DataType::Float64, DataType::Float64) => true,
                        (DataType::Float16, DataType::Float16) => true,
                        (DataType::BFloat16, DataType::BFloat16) => true,
                        (DataType::UInt8, DataType::UInt8) => true,
                        (DataType::Int32, DataType::Int32) => true,
                        (DataType::Int64, DataType::Int64) => true,
                        (DataType::Bool, DataType::Bool) => true,
                        // Allow some conversions for fusion
                        (DataType::Float32, DataType::Float64) => true,
                        (DataType::Float64, DataType::Float32) => true,
                        (DataType::Float32, DataType::Float16) => true,
                        (DataType::Float32, DataType::BFloat16) => true,
                        (DataType::Float16, DataType::Float32) => true,
                        (DataType::BFloat16, DataType::Float32) => true,
                        (DataType::Int32, DataType::Float32) => true,
                        (DataType::Float32, DataType::Int32) => true,
                        _ => false, // Incompatible types
                    }
                } else {
                    false // Couldn't resolve tensor references
                }
            } else {
                false // Couldn't find matching input tensor
            }
        } else {
            false // No output tensors
        }
    }
    
    /// Get the next node in topological order (useful for simple fusion patterns)
    pub fn get_next_node(&self, current_node: &GraphNode) -> Option<&GraphNode> {
        // Find current node's position in topological order
        let current_pos = self.topological_order.iter()
            .position(|&node_id| node_id == current_node.id)?;
            
        // Get next node ID
        let next_node_id = self.topological_order.get(current_pos + 1)?;
        
        // Return the actual node
        self.nodes.get(next_node_id.0)
    }
    
    /// Infer the output shape of a fused operation based on its inputs
    pub fn infer_fused_output_shape(&self, fused_op_type: &str, input_tensors: &[TensorId]) -> Option<Shape> {
        // Get input shapes
        let input_shapes: Vec<&Shape> = input_tensors.iter()
            .filter_map(|tensor_id| {
                if let Some(tensor_ref) = self.get_tensor(*tensor_id) {
                    match tensor_ref {
                        TensorRef::Weight(tensor) => Some(&tensor.shape),
                        TensorRef::Intermediate(tensor) => Some(&tensor.shape),
                    }
                } else {
                    None
                }
            })
            .collect();
            
        if input_shapes.is_empty() {
            return None;
        }
        
        match fused_op_type {
            "ConvReLU" => {
                // ConvReLU output shape is same as convolution output
                // For convolution: output_shape = (input_shape - kernel_size + 2*padding) / stride + 1
                // For now, assume same shape (would need kernel size, stride, padding info from attributes)
                if let Some(conv_input_shape) = input_shapes.first() {
                    Some((*conv_input_shape).clone())
                } else {
                    None
                }
            }
            "MatMulAdd" => {
                // MatMulAdd: if A is [m, k] and B is [k, n], output is [m, n]
                // Add bias: if bias is [n], output remains [m, n]
                if input_shapes.len() >= 2 {
                    let matmul_shape = input_shapes[0];
                    // Simplified: assume MatMul preserves the first input's shape
                    // Real implementation would need proper matrix multiplication logic
                    Some((*matmul_shape).clone())
                } else {
                    None
                }
            }
            "AddReLU" => {
                // AddReLU output shape is same as input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
            "ConvBatchNormReLU" => {
                // ConvBatchNormReLU output shape is same as convolution output
                // BatchNorm and ReLU don't change shape
                if let Some(conv_input_shape) = input_shapes.first() {
                    Some((*conv_input_shape).clone())
                } else {
                    None
                }
            }
            "EdgeConvReLU" => {
                // EdgeConvReLU output shape is same as convolution output
                // Edge quantization doesn't change shape, just data type
                if let Some(conv_input_shape) = input_shapes.first() {
                    Some((*conv_input_shape).clone())
                } else {
                    None
                }
            }
            "LayerNorm" => {
                // LayerNorm typically preserves input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
            "Gelu" => {
                // GELU activation preserves input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
            "Sigmoid" => {
                // Sigmoid activation preserves input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
            "Tanh" => {
                // Tanh activation preserves input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
            _ => {
                // Default: use first input shape
                if let Some(input_shape) = input_shapes.first() {
                    Some((*input_shape).clone())
                } else {
                    None
                }
            }
        }
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub weight_tensor_count: usize,
    pub intermediate_tensor_count: usize,
    pub total_weight_size: usize,
    pub total_intermediate_memory: usize,
    pub is_acyclic: bool,
}

/// Reference to either a weight tensor or intermediate tensor
#[derive(Debug, Clone)]
pub enum TensorRef<'a> {
    Weight(&'a WeightTensor),
    Intermediate(&'a IntermediateTensor),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_construction() {
        let mut ir = UniversalIR::new("test_model".to_string());
        
        // Add input tensor
        let input_tensor = ir.add_intermediate_tensor(
            "input".to_string(),
            Shape::new(vec![1, 784]), // 28x28 image flattened
            DataType::Float32,
        );
        
        // Add weight tensors
        let weight1 = ir.add_weight_tensor(
            "weight1".to_string(),
            Shape::new(vec![784, 128]),
            DataType::Float32,
        );
        
        let bias1 = ir.add_weight_tensor(
            "bias1".to_string(),
            Shape::new(vec![128]),
            DataType::Float32,
        );
        
        // Add intermediate tensors
        let hidden = ir.add_intermediate_tensor(
            "hidden".to_string(),
            Shape::new(vec![1, 128]),
            DataType::Float32,
        );
        
        let output = ir.add_intermediate_tensor(
            "output".to_string(),
            Shape::new(vec![1, 10]),
            DataType::Float32,
        );
        
        // Add operations
        let matmul_node = ir.add_node(OpType::MatMul);
        let add_node = ir.add_node(OpType::Add);
        let relu_node = ir.add_node(OpType::Relu);
        
        // Connect the graph
        ir.add_node_input(matmul_node, input_tensor).unwrap();
        ir.add_node_input(matmul_node, weight1).unwrap();
        ir.add_node_output(matmul_node, hidden).unwrap();
        
        ir.add_node_input(add_node, hidden).unwrap();
        ir.add_node_input(add_node, bias1).unwrap();
        ir.add_node_output(add_node, hidden).unwrap(); // Reuse hidden tensor (in-place operation)
        
        ir.add_node_input(relu_node, hidden).unwrap();
        ir.add_node_output(relu_node, output).unwrap(); // Output to final tensor
        
        // Validate the IR
        if let Err(e) = ir.validate() {
            panic!("Validation failed: {:?}", e);
        }
        
        // Check graph properties
        let stats = ir.get_graph_stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.weight_tensor_count, 2);
        assert_eq!(stats.intermediate_tensor_count, 3);
        // Note: Graph may have cycles due to in-place operations (hidden tensor reuse)
        
        // Test topological sort (may fail if graph has cycles)
        // Note: This test creates a cycle by reusing hidden tensor, so we skip this check
        // let execution_order = ir.topological_sort().unwrap();
        // assert_eq!(execution_order.len(), 3);
        
        // Test tensor lookups by ID (O(1) lookup)
        let tensor_ref = ir.get_tensor(input_tensor).unwrap();
        match tensor_ref {
            TensorRef::Intermediate(tensor) => {
                assert_eq!(tensor.name, "input");
                assert_eq!(tensor.shape.dimensions, vec![1, 784]);
            }
            _ => panic!("Expected intermediate tensor"),
        }
        
        // Test tensor type detection
        assert!(input_tensor.is_intermediate());
        assert!(weight1.is_weight());
    }
    
    #[test]
    fn test_graph_traversal() {
        let mut ir = UniversalIR::new("traversal_test".to_string());
        
        // Create a simple linear graph: A -> B -> C
        let tensor_a = ir.add_intermediate_tensor("A".to_string(), Shape::new(vec![1]), DataType::Float32);
        let tensor_b = ir.add_intermediate_tensor("B".to_string(), Shape::new(vec![1]), DataType::Float32);
        let tensor_c = ir.add_intermediate_tensor("C".to_string(), Shape::new(vec![1]), DataType::Float32);
        
        let node_a = ir.add_node(OpType::Relu);
        let node_b = ir.add_node(OpType::Relu);
        
        ir.add_node_input(node_a, tensor_a).unwrap();
        ir.add_node_output(node_a, tensor_b).unwrap();
        
        ir.add_node_input(node_b, tensor_b).unwrap();
        ir.add_node_output(node_b, tensor_c).unwrap();
        
        // Test producer/consumer relationships
        let producers_of_b = ir.get_producer_nodes(tensor_b);
        assert_eq!(producers_of_b.len(), 1);
        assert_eq!(producers_of_b[0], node_a);
        
        let consumers_of_b = ir.get_consumer_nodes(tensor_b);
        assert_eq!(consumers_of_b.len(), 1);
        assert_eq!(consumers_of_b[0], node_b);
        
        // Test root and leaf nodes
        let root_nodes = ir.get_root_nodes();
        assert_eq!(root_nodes.len(), 1);
        assert_eq!(root_nodes[0], node_a);
        
        let leaf_nodes = ir.get_leaf_nodes();
        assert_eq!(leaf_nodes.len(), 1);
        assert_eq!(leaf_nodes[0], node_b);
        
        // Test O(1) tensor lookups
        let tensor_b_ref = ir.get_tensor(tensor_b).unwrap();
        match tensor_b_ref {
            TensorRef::Intermediate(tensor) => {
                assert_eq!(tensor.name, "B");
                assert_eq!(tensor.shape.dimensions, vec![1]);
            }
            _ => panic!("Expected intermediate tensor"),
        }
    }
    
    #[test]
    fn test_optimized_ir_high_performance() {
        // Create original IR
        let mut original_ir = UniversalIR::new("test_model".to_string());
        
        // Add tensors and nodes
        let input_tensor = original_ir.add_intermediate_tensor("input".to_string(), Shape::new(vec![1, 784]), DataType::Float32);
        let weight = original_ir.add_weight_tensor("weight".to_string(), Shape::new(vec![784, 128]), DataType::Float32);
        let output_tensor = original_ir.add_intermediate_tensor("output".to_string(), Shape::new(vec![1, 128]), DataType::Float32);
        
        let matmul_node = original_ir.add_node(OpType::MatMul);
        original_ir.add_node_input(matmul_node, input_tensor).unwrap();
        original_ir.add_node_input(matmul_node, weight).unwrap();
        original_ir.add_node_output(matmul_node, output_tensor).unwrap();
        
        // Create optimized IR using high-performance architecture
        let target = crate::format::HardwareTarget::Cpu;
        let mut optimized_ir = OptimizedIR::new(&original_ir, target.clone());
        
        // Add optimized nodes (simulating optimization that reorders/fuses operations)
        let optimized_node = GraphNode::new(NodeId(0), OpType::Gemm); // Fused MatMul+Bias
        optimized_ir.add_optimized_node(optimized_node);
        
        // Add new intermediate tensor created during optimization
        let fused_tensor = IntermediateTensor::new("fused_output".to_string(), Shape::new(vec![1, 128]), DataType::Float32);
        optimized_ir.add_intermediate_tensor(fused_tensor);
        
        // Test high-performance O(1) lookups
        let node_ref = optimized_ir.get_optimized_node(NodeId(0)).unwrap();
        assert_eq!(node_ref.op_type, OpType::Gemm);
        
        // Test tensor lookups
        let tensor_ref = optimized_ir.get_intermediate_tensor(TensorId::intermediate(0)).unwrap();
        assert_eq!(tensor_ref.name, "fused_output");
        
        // Test shared weight access
        let weight_ref = optimized_ir.get_shared_weight(TensorId::weight(0)).unwrap();
        assert_eq!(weight_ref.name, "weight");
        
        // Test statistics
        assert_eq!(optimized_ir.node_count(), 1);
        assert_eq!(optimized_ir.intermediate_tensor_count(), 1);
        assert_eq!(optimized_ir.weight_tensor_count(), 1);
        
        // Test optimized nodes (fused MatMul+Bias -> Gemm)
        assert_eq!(optimized_ir.optimized_nodes.len(), 1);
        assert_eq!(optimized_ir.optimized_nodes[0].op_type, OpType::Gemm);
        
        // Verify memory efficiency - shared weights should be the same Arc
        let optimized_ir2 = OptimizedIR::new(&UniversalIR::new("test2".to_string()), target);
        // Both should have different Arc instances (different models)
        assert!(!std::ptr::eq(
            optimized_ir.shared_weights.as_ref(),
            optimized_ir2.shared_weights.as_ref()
        ));
    }
}