//! SAIC Model Format Specification
//! 
//! Defines the universal model format that SAIC can execute natively

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SAIC Model Payload Structure
/// 
/// This is the universal format that SAIC can execute natively.
/// Models are converted to this format when packaged into SAIC containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaicModel {
    /// Model metadata and configuration
    pub metadata: ModelMetadata,
    
    /// Computational graph definition
    pub graph: ComputationalGraph,
    
    /// Model weights and parameters
    pub weights: ModelWeights,
    
    /// Optimization hints for execution
    pub optimization_hints: ModelOptimizationHints,
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name and version
    pub name: String,
    pub version: String,
    
    /// Input and output specifications
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    
    /// Model architecture information
    pub architecture: String,
    pub framework: String,
    
    /// Model statistics
    pub parameter_count: usize,
    pub model_size_bytes: usize,
}

/// Tensor specification for inputs and outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    
    /// Data type (f32, f16, i8, etc.)
    pub data_type: DataType,
    
    /// Tensor shape (dynamic dimensions supported)
    pub shape: Vec<Dimension>,
    
    /// Optional quantization info
    pub quantization: Option<QuantizationInfo>,
}

/// Data types supported by SAIC
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    F32,
    F16,
    I8,
    I16,
    I32,
    U8,
    Bool,
}

/// Dimension specification (supports dynamic dimensions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Dimension {
    /// Fixed dimension size
    Fixed(usize),
    
    /// Dynamic dimension (will be determined at runtime)
    Dynamic(String),
    
    /// Unknown dimension (will be inferred)
    Unknown,
}

/// Quantization information for optimized inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization type
    pub quantization_type: QuantizationType,
    
    /// Scale factor for quantization
    pub scale: f32,
    
    /// Zero point for quantization
    pub zero_point: i32,
    
    /// Minimum and maximum values
    pub min_value: f32,
    pub max_value: f32,
}

/// Types of quantization supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    Symmetric,
    Asymmetric,
    Dynamic,
}

/// Computational graph definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalGraph {
    /// Graph nodes (operations)
    pub nodes: Vec<GraphNode>,
    
    /// Graph edges (data flow)
    pub edges: Vec<GraphEdge>,
    
    /// Execution order
    pub execution_order: Vec<usize>,
}

/// Graph node representing an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node identifier
    pub id: usize,
    
    /// Node name
    pub name: String,
    
    /// Operation type
    pub operation: Operation,
    
    /// Input tensor names
    pub inputs: Vec<String>,
    
    /// Output tensor names
    pub outputs: Vec<String>,
    
    /// Operation-specific attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Neural network operations supported by SAIC
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Operation {
    /// Input tensor operation
    Input,
    
    /// Output tensor operation
    Output,
    
    /// Matrix multiplication
    MatMul,
    
    /// Rectified Linear Unit activation
    Relu,
    
    /// 2D Convolution
    Conv2D,
    
    /// 2D Max Pooling
    MaxPool2D,
    
    /// Global Average Pooling
    GlobalAvgPool2D,
    
    /// Softmax activation
    Softmax,
    
    /// Custom operation (for universal compatibility)
    Custom(String),
}

/// Attribute values for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Bool(bool),
    IntArray(Vec<i64>),
    FloatArray(Vec<f32>),
    StringArray(Vec<String>),
}

/// Graph edge representing data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node and output
    pub source: (usize, String),
    
    /// Destination node and input
    pub destination: (usize, String),
    
    /// Edge metadata
    pub metadata: EdgeMetadata,
}

/// Edge metadata for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Data type of the tensor flowing through this edge
    pub data_type: DataType,
    
    /// Shape information
    pub shape: Vec<Dimension>,
    
    /// Memory layout hints
    pub memory_layout: MemoryLayout,
    
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Memory layout specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Row-major layout (C-style)
    RowMajor,
    
    /// Column-major layout (Fortran-style)
    ColumnMajor,
    
    /// Channel-first layout (NCHW)
    ChannelFirst,
    
    /// Channel-last layout (NHWC)
    ChannelLast,
}

/// Optimization hints for edge-level optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationHint {
    /// Use specific SIMD instructions
    UseSimd(String),
    
    /// Memory alignment hint
    AlignTo(usize),
    
    /// Cache optimization hint
    CacheOptimized,
    
    /// Zero-copy optimization
    ZeroCopy,
    
    /// Quantization hint
    Quantized(DataType),
    
    /// Custom optimization hint
    Custom(String),
}

/// Model weights and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Weight tensors by name
    pub tensors: HashMap<String, WeightTensor>,
    
    /// Weight metadata
    pub metadata: WeightsMetadata,
}

/// Individual weight tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTensor {
    /// Tensor name
    pub name: String,
    
    /// Data type
    pub data_type: DataType,
    
    /// Tensor shape
    pub shape: Vec<usize>,
    
    /// Raw weight data (encoded)
    pub data: Vec<u8>,
    
    /// Quantization info if applicable
    pub quantization: Option<QuantizationInfo>,
}

/// Weights metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsMetadata {
    /// Total number of parameters
    pub total_parameters: usize,
    
    /// Total size in bytes
    pub total_size_bytes: usize,
    
    /// Compression ratio if compressed
    pub compression_ratio: Option<f32>,
    
    /// Weight initialization method
    pub initialization_method: String,
}

/// Model optimization hints for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptimizationHints {
    /// Hardware-specific optimizations
    pub hardware_hints: HashMap<String, HardwareHints>,
    
    /// Memory optimization hints
    pub memory_hints: MemoryHints,
    
    /// Execution optimization hints
    pub execution_hints: ExecutionHints,
}

/// Hardware-specific optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareHints {
    /// Target hardware type
    pub hardware_type: String,
    
    /// Recommended optimizations
    pub optimizations: Vec<String>,
    
    /// Performance targets
    pub performance_targets: PerformanceTargets,
}

/// Performance targets for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency in microseconds
    pub target_latency_us: Option<u64>,
    
    /// Target throughput (inferences per second)
    pub target_throughput: Option<u64>,
    
    /// Target memory usage in MB
    pub target_memory_mb: Option<usize>,
    
    /// Target power consumption in watts
    pub target_power_w: Option<f32>,
}

/// Memory optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHints {
    /// Preferred memory layout
    pub preferred_layout: MemoryLayout,
    
    /// Memory alignment requirements
    pub alignment_bytes: usize,
    
    /// Cache optimization hints
    pub cache_hints: Vec<String>,
    
    /// Memory pooling recommendations
    pub memory_pooling: bool,
}

/// Execution optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHints {
    /// Parallelization strategy
    pub parallelization: ParallelizationStrategy,
    
    /// Batching recommendations
    pub batching: BatchingHints,
    
    /// Precision requirements
    pub precision: PrecisionHints,
}

/// Parallelization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// Single-threaded execution
    SingleThreaded,
    
    /// Multi-threaded execution
    MultiThreaded(usize),
    
    /// SIMD vectorization
    Simd,
    
    /// GPU acceleration
    Gpu,
    
    /// Hybrid approach
    Hybrid,
}

/// Batching optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingHints {
    /// Optimal batch size
    pub optimal_batch_size: usize,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Dynamic batching support
    pub dynamic_batching: bool,
}

/// Precision optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionHints {
    /// Minimum precision required
    pub min_precision: DataType,
    
    /// Optimal precision for performance
    pub optimal_precision: DataType,
    
    /// Mixed precision support
    pub mixed_precision: bool,
}

/// SAIC Model Builder for creating models programmatically
pub struct SaicModelBuilder {
    model: SaicModel,
}

impl SaicModelBuilder {
    /// Create a new model builder
    pub fn new(name: String, version: String) -> Self {
        Self {
            model: SaicModel {
                metadata: ModelMetadata {
                    name,
                    version,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    architecture: String::new(),
                    framework: String::new(),
                    parameter_count: 0,
                    model_size_bytes: 0,
                },
                graph: ComputationalGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                    execution_order: Vec::new(),
                },
                weights: ModelWeights {
                    tensors: HashMap::new(),
                    metadata: WeightsMetadata {
                        total_parameters: 0,
                        total_size_bytes: 0,
                        compression_ratio: None,
                        initialization_method: String::new(),
                    },
                },
                optimization_hints: ModelOptimizationHints {
                    hardware_hints: HashMap::new(),
                    memory_hints: MemoryHints {
                        preferred_layout: MemoryLayout::RowMajor,
                        alignment_bytes: 32,
                        cache_hints: Vec::new(),
                        memory_pooling: false,
                    },
                    execution_hints: ExecutionHints {
                        parallelization: ParallelizationStrategy::SingleThreaded,
                        batching: BatchingHints {
                            optimal_batch_size: 1,
                            max_batch_size: 1,
                            dynamic_batching: false,
                        },
                        precision: PrecisionHints {
                            min_precision: DataType::F32,
                            optimal_precision: DataType::F32,
                            mixed_precision: false,
                        },
                    },
                },
            },
        }
    }
    
    /// Add input tensor specification
    pub fn add_input(mut self, input: TensorSpec) -> Self {
        self.model.metadata.inputs.push(input);
        self
    }
    
    /// Add output tensor specification
    pub fn add_output(mut self, output: TensorSpec) -> Self {
        self.model.metadata.outputs.push(output);
        self
    }
    
    /// Set model architecture
    pub fn set_architecture(mut self, architecture: String) -> Self {
        self.model.metadata.architecture = architecture;
        self
    }
    
    /// Set framework
    pub fn set_framework(mut self, framework: String) -> Self {
        self.model.metadata.framework = framework;
        self
    }
    
    /// Add graph node
    pub fn add_node(mut self, node: GraphNode) -> Self {
        self.model.graph.nodes.push(node);
        self
    }
    
    /// Add graph edge
    pub fn add_edge(mut self, edge: GraphEdge) -> Self {
        self.model.graph.edges.push(edge);
        self
    }
    
    /// Set execution order
    pub fn set_execution_order(mut self, order: Vec<usize>) -> Self {
        self.model.graph.execution_order = order;
        self
    }
    
    /// Add weight tensor
    pub fn add_weight(mut self, name: String, weight: WeightTensor) -> Self {
        self.model.weights.tensors.insert(name, weight);
        self
    }
    
    /// Build the final model
    pub fn build(self) -> SaicModel {
        self.model
    }
}

impl SaicModel {
    /// Create a new model using the builder pattern
    pub fn builder(name: String, version: String) -> SaicModelBuilder {
        SaicModelBuilder::new(name, version)
    }
    
    /// Get the total number of parameters in the model
    pub fn parameter_count(&self) -> usize {
        self.weights.metadata.total_parameters
    }
    
    /// Get the total model size in bytes
    pub fn model_size_bytes(&self) -> usize {
        self.weights.metadata.total_size_bytes
    }
    
    /// Check if the model supports a specific operation
    pub fn supports_operation(&self, operation: &Operation) -> bool {
        self.graph.nodes.iter().any(|node| node.operation == *operation)
    }
    
    /// Get all input tensor names
    pub fn input_names(&self) -> Vec<&String> {
        self.metadata.inputs.iter().map(|input| &input.name).collect()
    }
    
    /// Get all output tensor names
    pub fn output_names(&self) -> Vec<&String> {
        self.metadata.outputs.iter().map(|output| &output.name).collect()
    }
}
