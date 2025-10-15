//! Universal IR (Intermediate Representation) for AI models
//! 
//! This module defines the universal intermediate representation that can hold
//! any AI model in a neutral, hardware-agnostic format.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{SaicError, SaicResult};

/// Data types supported by the Universal IR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
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

/// Operation types supported by the Universal IR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpType {
    // Basic arithmetic operations
    Add,
    Subtract,
    Multiply,
    Divide,
    
    // Neural network operations
    Convolution,
    Pooling,
    Activation,
    BatchNorm,
    Dropout,
    Reshape,
    Transpose,
    Concatenate,
    Split,
    
    // Matrix operations (for LLMs)
    MatMul,
    
    // Advanced operations
    Attention,
    MultiHeadAttention,
    SelfAttention,
    Transformer,
    Encoder,
    Decoder,
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

/// Operation in the Universal IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Operation type
    pub op_type: OpType,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Operation attributes
    pub attributes: HashMap<String, Attribute>,
}

impl Operation {
    /// Create a new operation
    pub fn new(op_type: OpType) -> Self {
        Self {
            op_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }
    
    /// Add input tensor
    pub fn add_input(&mut self, input: String) {
        self.inputs.push(input);
    }
    
    /// Add output tensor
    pub fn add_output(&mut self, output: String) {
        self.outputs.push(output);
    }
    
    /// Add attribute
    pub fn add_attribute(&mut self, key: String, value: Attribute) {
        self.attributes.insert(key, value);
    }
}

/// Tensor in the Universal IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Shape,
    /// Data type
    pub data_type: DataType,
    /// Raw tensor data
    pub data: Vec<u8>,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(name: String, shape: Shape, data_type: DataType) -> Self {
        let data_size = shape.total_elements() * data_type.size();
        Self {
            name,
            shape,
            data_type,
            data: vec![0; data_size],
        }
    }
    
    /// Create a tensor with data
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedIR {
    /// Original IR
    pub original_ir: UniversalIR,
    /// Hardware target
    pub target: crate::format::HardwareTarget,
    /// Optimized operations
    pub optimized_operations: Vec<Operation>,
    /// Optimization metadata
    pub optimization_metadata: HashMap<String, String>,
}

impl OptimizedIR {
    /// Create new optimized IR
    pub fn new(original_ir: UniversalIR, target: crate::format::HardwareTarget) -> Self {
        Self {
            original_ir,
            target: target.clone(),
            optimized_operations: Vec::new(),
            optimization_metadata: HashMap::new(),
        }
    }
    
    /// Add optimized operation
    pub fn add_optimized_operation(&mut self, operation: Operation) {
        self.optimized_operations.push(operation);
    }
    
    /// Add optimization metadata
    pub fn add_optimization_metadata(&mut self, key: String, value: String) {
        self.optimization_metadata.insert(key, value);
    }
}

/// Universal IR - the main intermediate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalIR {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Operations
    pub operations: Vec<Operation>,
    /// Tensors
    pub tensors: Vec<Tensor>,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
}

impl UniversalIR {
    /// Create a new Universal IR
    pub fn new(name: String) -> Self {
        Self {
            metadata: ModelMetadata::new(name),
            operations: Vec::new(),
            tensors: Vec::new(),
            optimization_hints: Vec::new(),
        }
    }
    
    /// Add an operation
    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }
    
    /// Add a tensor
    pub fn add_tensor(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }
    
    /// Add an optimization hint
    pub fn add_optimization_hint(&mut self, hint: OptimizationHint) {
        self.optimization_hints.push(hint);
    }
    
    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
    
    /// Get operation by index
    pub fn get_operation(&self, index: usize) -> Option<&Operation> {
        self.operations.get(index)
    }
    
    /// Validate the IR
    pub fn validate(&self) -> SaicResult<()> {
        // Validate metadata
        if self.metadata.name.is_empty() {
            return Err(SaicError::format("Model name cannot be empty"));
        }
        
        // Validate operations
        for (i, op) in self.operations.iter().enumerate() {
            if op.inputs.is_empty() && op.outputs.is_empty() {
                return Err(SaicError::format(format!("Operation {} has no inputs or outputs", i)));
            }
        }
        
        // Validate tensors
        for tensor in &self.tensors {
            if tensor.name.is_empty() {
                return Err(SaicError::format("Tensor name cannot be empty"));
            }
            
            if tensor.shape.dimensions.is_empty() {
                return Err(SaicError::format(format!("Tensor {} has empty shape", tensor.name)));
            }
        }
        
        Ok(())
    }
    
    /// Get total size of all tensors
    pub fn total_tensor_size(&self) -> usize {
        self.tensors.iter().map(|t| t.size()).sum()
    }
    
    /// Get number of operations
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
    
    /// Get number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}
