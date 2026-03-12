//! Backend Translator Parameters and Hints for COELANOX
//! 
//! This module defines the parameters and optimization hints used by backend translators.
//! Operations are now defined in coelanox-core::ir::OpType.

use std::collections::HashMap;

/// COELANOX AI Operation Parameters
/// 
/// Contains all the information needed to instantiate an AI operation
#[derive(Debug, Clone)]
pub struct CoelanoxAiOperationParams {
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Whether input tensors are contiguous in memory
    /// False indicates non-contiguous (e.g., from Slice, Transpose) - vendor must handle
    pub input_is_contiguous: Vec<bool>,
    /// Whether output tensors should be contiguous in memory
    pub output_is_contiguous: Vec<bool>,
    /// Operation-specific attributes
    pub attributes: HashMap<String, CoelanoxAttribute>,
    /// Optimization hints for the operation
    pub optimization_hints: Vec<CoelanoxOptimizationHint>,
}

/// COELANOX Attribute Types
#[derive(Debug, Clone)]
pub enum CoelanoxAttribute {
    Int(i32),
    Ints(Vec<i32>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
    Bool(bool),
}

/// COELANOX Optimization Hints
/// 
/// These hints help backend translators optimize operations for specific hardware
#[derive(Debug, Clone, PartialEq)]
pub enum CoelanoxOptimizationHint {
    /// Operation is memory efficient
    MemoryEfficient,
    /// Operation is compute intensive
    ComputeIntensive,
    /// Operation can be parallelized
    Parallelizable,
    /// Operation is cache friendly
    CacheFriendly,
    /// Operation can be vectorized
    Vectorizable,
    /// Operation benefits from tensor cores
    TensorCoreOptimized,
    /// Operation benefits from neural engines
    NeuralEngineOptimized,
    /// Operation benefits from NPU acceleration
    NpuOptimized,
}

/// COELANOX AI Operation Metadata
/// 
/// Contains metadata about an AI operation for optimization and debugging
#[derive(Debug, Clone)]
pub struct CoelanoxAiOperationMetadata {
    /// Operation name
    pub name: String,
    /// Operation description
    pub description: String,
    /// Expected performance characteristics
    pub performance_profile: CoelanoxPerformanceProfile,
    /// Memory requirements
    pub memory_requirements: CoelanoxMemoryRequirements,
    /// Hardware compatibility
    pub hardware_compatibility: Vec<CoelanoxHardwareType>,
}

/// Performance profile for an operation
#[derive(Debug, Clone)]
pub struct CoelanoxPerformanceProfile {
    /// Expected execution time (microseconds)
    pub expected_execution_time: u64,
    /// Expected throughput (operations per second)
    pub expected_throughput: u64,
    /// Scalability factor (how well it scales with hardware)
    pub scalability_factor: f32,
}

/// Memory requirements for an operation
#[derive(Debug, Clone)]
pub struct CoelanoxMemoryRequirements {
    /// Minimum memory required (bytes)
    pub min_memory: usize,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Memory bandwidth requirements (bytes/second)
    pub memory_bandwidth: u64,
}

/// Hardware types supported by COELANOX
#[derive(Debug, Clone, PartialEq)]
pub enum CoelanoxHardwareType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    NeuralEngine,
    Edge,
    Custom(String),
}

// Helper function to get optimization hints for OpType
// This can be used by backend translators to get hints for operations
pub fn get_optimization_hints_for_op_type(op_type: &coelanox_core::ir::OpType) -> Vec<CoelanoxOptimizationHint> {
    use coelanox_core::ir::OpType;
    match op_type {
        OpType::Add | OpType::Subtract | OpType::Multiply | OpType::Divide => vec![
            CoelanoxOptimizationHint::Vectorizable,
            CoelanoxOptimizationHint::Parallelizable,
            CoelanoxOptimizationHint::MemoryEfficient,
        ],
        OpType::Convolution => vec![
            CoelanoxOptimizationHint::ComputeIntensive,
            CoelanoxOptimizationHint::Parallelizable,
            CoelanoxOptimizationHint::TensorCoreOptimized,
            CoelanoxOptimizationHint::NeuralEngineOptimized,
        ],
        OpType::MatMul | OpType::Gemm => vec![
            CoelanoxOptimizationHint::ComputeIntensive,
            CoelanoxOptimizationHint::TensorCoreOptimized,
            CoelanoxOptimizationHint::NpuOptimized,
            CoelanoxOptimizationHint::CacheFriendly,
        ],
        OpType::Relu | OpType::Sigmoid | OpType::Tanh | OpType::Softmax | OpType::Gelu | OpType::Swish => vec![
            CoelanoxOptimizationHint::Vectorizable,
            CoelanoxOptimizationHint::MemoryEfficient,
            CoelanoxOptimizationHint::NeuralEngineOptimized,
        ],
        OpType::MaxPool | OpType::AvgPool | OpType::GlobalMaxPool | OpType::GlobalAvgPool => vec![
            CoelanoxOptimizationHint::MemoryEfficient,
            CoelanoxOptimizationHint::Parallelizable,
            CoelanoxOptimizationHint::NeuralEngineOptimized,
        ],
        _ => vec![
            CoelanoxOptimizationHint::Parallelizable,
        ],
    }
}
