//! Standardized AI Operations for COELANOX
//! 
//! This module defines the universal AI operations that COELANOX supports.
//! These operations form the foundation of COELANOX's "De Beers of AI" strategy.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// COELANOX Standardized AI Operations
/// 
/// These are the universal operations that COELANOX defines and controls.
/// Hardware vendors must implement translators for these operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoelanoxAiOperation {
    // Basic Arithmetic Operations
    Add,
    Subtract,
    Multiply,
    Divide,
    
    // Neural Network Operations
    Convolution,
    Pooling,
    Activation,
    BatchNorm,
    Dropout,
    
    // Matrix Operations
    MatMul,
    Transpose,
    Reshape,
    
    // Attention Operations
    Attention,
    MultiHeadAttention,
    SelfAttention,
    
    // Transformer Operations
    Transformer,
    Encoder,
    Decoder,
    
    // Custom Operations
    Custom(String),
}

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

impl CoelanoxAiOperation {
    /// Get metadata for an operation
    pub fn metadata(&self) -> CoelanoxAiOperationMetadata {
        match self {
            CoelanoxAiOperation::Add => CoelanoxAiOperationMetadata {
                name: "Add".to_string(),
                description: "Element-wise addition of two tensors".to_string(),
                performance_profile: CoelanoxPerformanceProfile {
                    expected_execution_time: 1,
                    expected_throughput: 1_000_000,
                    scalability_factor: 0.95,
                },
                memory_requirements: CoelanoxMemoryRequirements {
                    min_memory: 1024,
                    peak_memory: 4096,
                    memory_bandwidth: 1_000_000,
                },
                hardware_compatibility: vec![
                    CoelanoxHardwareType::Cpu,
                    CoelanoxHardwareType::Gpu,
                    CoelanoxHardwareType::Npu,
                    CoelanoxHardwareType::Tpu,
                    CoelanoxHardwareType::NeuralEngine,
                ],
            },
            CoelanoxAiOperation::Convolution => CoelanoxAiOperationMetadata {
                name: "Convolution".to_string(),
                description: "2D convolution operation".to_string(),
                performance_profile: CoelanoxPerformanceProfile {
                    expected_execution_time: 100,
                    expected_throughput: 10_000,
                    scalability_factor: 0.90,
                },
                memory_requirements: CoelanoxMemoryRequirements {
                    min_memory: 1024 * 1024,
                    peak_memory: 16 * 1024 * 1024,
                    memory_bandwidth: 100_000_000,
                },
                hardware_compatibility: vec![
                    CoelanoxHardwareType::Cpu,
                    CoelanoxHardwareType::Gpu,
                    CoelanoxHardwareType::Npu,
                    CoelanoxHardwareType::Tpu,
                    CoelanoxHardwareType::NeuralEngine,
                ],
            },
            CoelanoxAiOperation::MatMul => CoelanoxAiOperationMetadata {
                name: "MatMul".to_string(),
                description: "Matrix multiplication operation".to_string(),
                performance_profile: CoelanoxPerformanceProfile {
                    expected_execution_time: 50,
                    expected_throughput: 20_000,
                    scalability_factor: 0.85,
                },
                memory_requirements: CoelanoxMemoryRequirements {
                    min_memory: 512 * 1024,
                    peak_memory: 8 * 1024 * 1024,
                    memory_bandwidth: 50_000_000,
                },
                hardware_compatibility: vec![
                    CoelanoxHardwareType::Cpu,
                    CoelanoxHardwareType::Gpu,
                    CoelanoxHardwareType::Npu,
                    CoelanoxHardwareType::Tpu,
                    CoelanoxHardwareType::NeuralEngine,
                ],
            },
            _ => CoelanoxAiOperationMetadata {
                name: format!("{:?}", self),
                description: "Generic AI operation".to_string(),
                performance_profile: CoelanoxPerformanceProfile {
                    expected_execution_time: 10,
                    expected_throughput: 100_000,
                    scalability_factor: 0.80,
                },
                memory_requirements: CoelanoxMemoryRequirements {
                    min_memory: 2048,
                    peak_memory: 8192,
                    memory_bandwidth: 10_000_000,
                },
                hardware_compatibility: vec![
                    CoelanoxHardwareType::Cpu,
                    CoelanoxHardwareType::Gpu,
                    CoelanoxHardwareType::Npu,
                    CoelanoxHardwareType::Tpu,
                    CoelanoxHardwareType::NeuralEngine,
                ],
            },
        }
    }
    
    /// Get optimization hints for an operation
    pub fn optimization_hints(&self) -> Vec<CoelanoxOptimizationHint> {
        match self {
            CoelanoxAiOperation::Add => vec![
                CoelanoxOptimizationHint::Vectorizable,
                CoelanoxOptimizationHint::Parallelizable,
                CoelanoxOptimizationHint::MemoryEfficient,
            ],
            CoelanoxAiOperation::Convolution => vec![
                CoelanoxOptimizationHint::ComputeIntensive,
                CoelanoxOptimizationHint::Parallelizable,
                CoelanoxOptimizationHint::TensorCoreOptimized,
                CoelanoxOptimizationHint::NeuralEngineOptimized,
            ],
            CoelanoxAiOperation::MatMul => vec![
                CoelanoxOptimizationHint::ComputeIntensive,
                CoelanoxOptimizationHint::TensorCoreOptimized,
                CoelanoxOptimizationHint::NpuOptimized,
                CoelanoxOptimizationHint::CacheFriendly,
            ],
            CoelanoxAiOperation::Activation => vec![
                CoelanoxOptimizationHint::Vectorizable,
                CoelanoxOptimizationHint::MemoryEfficient,
                CoelanoxOptimizationHint::NeuralEngineOptimized,
            ],
            CoelanoxAiOperation::Pooling => vec![
                CoelanoxOptimizationHint::MemoryEfficient,
                CoelanoxOptimizationHint::Parallelizable,
                CoelanoxOptimizationHint::NeuralEngineOptimized,
            ],
            _ => vec![
                CoelanoxOptimizationHint::Parallelizable,
            ],
        }
    }
    
    /// Check if operation is supported on hardware type
    pub fn is_supported_on(&self, hardware_type: &CoelanoxHardwareType) -> bool {
        self.metadata().hardware_compatibility.contains(hardware_type)
    }
    
    /// Get all supported COELANOX AI operations
    pub fn all_operations() -> Vec<CoelanoxAiOperation> {
        vec![
            CoelanoxAiOperation::Add,
            CoelanoxAiOperation::Subtract,
            CoelanoxAiOperation::Multiply,
            CoelanoxAiOperation::Divide,
            CoelanoxAiOperation::Convolution,
            CoelanoxAiOperation::Pooling,
            CoelanoxAiOperation::Activation,
            CoelanoxAiOperation::BatchNorm,
            CoelanoxAiOperation::Dropout,
            CoelanoxAiOperation::MatMul,
            CoelanoxAiOperation::Transpose,
            CoelanoxAiOperation::Reshape,
            CoelanoxAiOperation::Attention,
            CoelanoxAiOperation::MultiHeadAttention,
            CoelanoxAiOperation::SelfAttention,
            CoelanoxAiOperation::Transformer,
            CoelanoxAiOperation::Encoder,
            CoelanoxAiOperation::Decoder,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coelanox_operation_metadata() {
        let add_op = CoelanoxAiOperation::Add;
        let metadata = add_op.metadata();
        
        assert_eq!(metadata.name, "Add");
        assert_eq!(metadata.description, "Element-wise addition of two tensors");
        assert!(metadata.hardware_compatibility.contains(&CoelanoxHardwareType::Cpu));
        assert!(metadata.hardware_compatibility.contains(&CoelanoxHardwareType::Gpu));
    }

    #[test]
    fn test_coelanox_operation_optimization_hints() {
        let conv_op = CoelanoxAiOperation::Convolution;
        let hints = conv_op.optimization_hints();
        
        assert!(hints.contains(&CoelanoxOptimizationHint::ComputeIntensive));
        assert!(hints.contains(&CoelanoxOptimizationHint::TensorCoreOptimized));
        assert!(hints.contains(&CoelanoxOptimizationHint::NeuralEngineOptimized));
    }

    #[test]
    fn test_coelanox_operation_hardware_support() {
        let add_op = CoelanoxAiOperation::Add;
        
        assert!(add_op.is_supported_on(&CoelanoxHardwareType::Cpu));
        assert!(add_op.is_supported_on(&CoelanoxHardwareType::Gpu));
        assert!(add_op.is_supported_on(&CoelanoxHardwareType::Npu));
        assert!(add_op.is_supported_on(&CoelanoxHardwareType::Tpu));
        assert!(add_op.is_supported_on(&CoelanoxHardwareType::NeuralEngine));
    }

    #[test]
    fn test_all_operations() {
        let operations = CoelanoxAiOperation::all_operations();
        
        assert!(operations.contains(&CoelanoxAiOperation::Add));
        assert!(operations.contains(&CoelanoxAiOperation::Convolution));
        assert!(operations.contains(&CoelanoxAiOperation::MatMul));
        assert!(operations.contains(&CoelanoxAiOperation::Attention));
        assert!(operations.contains(&CoelanoxAiOperation::Transformer));
    }
}
