//! Standardized AI Operations for SAIC
//! 
//! This module defines the universal AI operations that SAIC supports.
//! These operations form the foundation of SAIC's "De Beers of AI" strategy.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// SAIC Standardized AI Operations
/// 
/// These are the universal operations that SAIC defines and controls.
/// Hardware vendors must implement translators for these operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SaicAiOperation {
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

/// SAIC AI Operation Parameters
/// 
/// Contains all the information needed to instantiate an AI operation
#[derive(Debug, Clone)]
pub struct SaicAiOperationParams {
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Operation-specific attributes
    pub attributes: HashMap<String, SaicAttribute>,
    /// Optimization hints for the operation
    pub optimization_hints: Vec<SaicOptimizationHint>,
}

/// SAIC Attribute Types
#[derive(Debug, Clone)]
pub enum SaicAttribute {
    Int(i32),
    Ints(Vec<i32>),
    Float(f32),
    Floats(Vec<f32>),
    String(String),
    Strings(Vec<String>),
    Bool(bool),
}

/// SAIC Optimization Hints
/// 
/// These hints help backend translators optimize operations for specific hardware
#[derive(Debug, Clone, PartialEq)]
pub enum SaicOptimizationHint {
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

/// SAIC AI Operation Metadata
/// 
/// Contains metadata about an AI operation for optimization and debugging
#[derive(Debug, Clone)]
pub struct SaicAiOperationMetadata {
    /// Operation name
    pub name: String,
    /// Operation description
    pub description: String,
    /// Expected performance characteristics
    pub performance_profile: SaicPerformanceProfile,
    /// Memory requirements
    pub memory_requirements: SaicMemoryRequirements,
    /// Hardware compatibility
    pub hardware_compatibility: Vec<SaicHardwareType>,
}

/// Performance profile for an operation
#[derive(Debug, Clone)]
pub struct SaicPerformanceProfile {
    /// Expected execution time (microseconds)
    pub expected_execution_time: u64,
    /// Expected throughput (operations per second)
    pub expected_throughput: u64,
    /// Scalability factor (how well it scales with hardware)
    pub scalability_factor: f32,
}

/// Memory requirements for an operation
#[derive(Debug, Clone)]
pub struct SaicMemoryRequirements {
    /// Minimum memory required (bytes)
    pub min_memory: usize,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Memory bandwidth requirements (bytes/second)
    pub memory_bandwidth: u64,
}

/// Hardware types supported by SAIC
#[derive(Debug, Clone, PartialEq)]
pub enum SaicHardwareType {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    NeuralEngine,
    Edge,
    Custom(String),
}

impl SaicAiOperation {
    /// Get metadata for an operation
    pub fn metadata(&self) -> SaicAiOperationMetadata {
        match self {
            SaicAiOperation::Add => SaicAiOperationMetadata {
                name: "Add".to_string(),
                description: "Element-wise addition of two tensors".to_string(),
                performance_profile: SaicPerformanceProfile {
                    expected_execution_time: 1,
                    expected_throughput: 1_000_000,
                    scalability_factor: 0.95,
                },
                memory_requirements: SaicMemoryRequirements {
                    min_memory: 1024,
                    peak_memory: 4096,
                    memory_bandwidth: 1_000_000,
                },
                hardware_compatibility: vec![
                    SaicHardwareType::Cpu,
                    SaicHardwareType::Gpu,
                    SaicHardwareType::Npu,
                    SaicHardwareType::Tpu,
                    SaicHardwareType::NeuralEngine,
                ],
            },
            SaicAiOperation::Convolution => SaicAiOperationMetadata {
                name: "Convolution".to_string(),
                description: "2D convolution operation".to_string(),
                performance_profile: SaicPerformanceProfile {
                    expected_execution_time: 100,
                    expected_throughput: 10_000,
                    scalability_factor: 0.90,
                },
                memory_requirements: SaicMemoryRequirements {
                    min_memory: 1024 * 1024,
                    peak_memory: 16 * 1024 * 1024,
                    memory_bandwidth: 100_000_000,
                },
                hardware_compatibility: vec![
                    SaicHardwareType::Cpu,
                    SaicHardwareType::Gpu,
                    SaicHardwareType::Npu,
                    SaicHardwareType::Tpu,
                    SaicHardwareType::NeuralEngine,
                ],
            },
            SaicAiOperation::MatMul => SaicAiOperationMetadata {
                name: "MatMul".to_string(),
                description: "Matrix multiplication operation".to_string(),
                performance_profile: SaicPerformanceProfile {
                    expected_execution_time: 50,
                    expected_throughput: 20_000,
                    scalability_factor: 0.85,
                },
                memory_requirements: SaicMemoryRequirements {
                    min_memory: 512 * 1024,
                    peak_memory: 8 * 1024 * 1024,
                    memory_bandwidth: 50_000_000,
                },
                hardware_compatibility: vec![
                    SaicHardwareType::Cpu,
                    SaicHardwareType::Gpu,
                    SaicHardwareType::Npu,
                    SaicHardwareType::Tpu,
                    SaicHardwareType::NeuralEngine,
                ],
            },
            _ => SaicAiOperationMetadata {
                name: format!("{:?}", self),
                description: "Generic AI operation".to_string(),
                performance_profile: SaicPerformanceProfile {
                    expected_execution_time: 10,
                    expected_throughput: 100_000,
                    scalability_factor: 0.80,
                },
                memory_requirements: SaicMemoryRequirements {
                    min_memory: 2048,
                    peak_memory: 8192,
                    memory_bandwidth: 10_000_000,
                },
                hardware_compatibility: vec![
                    SaicHardwareType::Cpu,
                    SaicHardwareType::Gpu,
                    SaicHardwareType::Npu,
                    SaicHardwareType::Tpu,
                    SaicHardwareType::NeuralEngine,
                ],
            },
        }
    }
    
    /// Get optimization hints for an operation
    pub fn optimization_hints(&self) -> Vec<SaicOptimizationHint> {
        match self {
            SaicAiOperation::Add => vec![
                SaicOptimizationHint::Vectorizable,
                SaicOptimizationHint::Parallelizable,
                SaicOptimizationHint::MemoryEfficient,
            ],
            SaicAiOperation::Convolution => vec![
                SaicOptimizationHint::ComputeIntensive,
                SaicOptimizationHint::Parallelizable,
                SaicOptimizationHint::TensorCoreOptimized,
                SaicOptimizationHint::NeuralEngineOptimized,
            ],
            SaicAiOperation::MatMul => vec![
                SaicOptimizationHint::ComputeIntensive,
                SaicOptimizationHint::TensorCoreOptimized,
                SaicOptimizationHint::NpuOptimized,
                SaicOptimizationHint::CacheFriendly,
            ],
            SaicAiOperation::Activation => vec![
                SaicOptimizationHint::Vectorizable,
                SaicOptimizationHint::MemoryEfficient,
                SaicOptimizationHint::NeuralEngineOptimized,
            ],
            SaicAiOperation::Pooling => vec![
                SaicOptimizationHint::MemoryEfficient,
                SaicOptimizationHint::Parallelizable,
                SaicOptimizationHint::NeuralEngineOptimized,
            ],
            _ => vec![
                SaicOptimizationHint::Parallelizable,
            ],
        }
    }
    
    /// Check if operation is supported on hardware type
    pub fn is_supported_on(&self, hardware_type: &SaicHardwareType) -> bool {
        self.metadata().hardware_compatibility.contains(hardware_type)
    }
    
    /// Get all supported SAIC AI operations
    pub fn all_operations() -> Vec<SaicAiOperation> {
        vec![
            SaicAiOperation::Add,
            SaicAiOperation::Subtract,
            SaicAiOperation::Multiply,
            SaicAiOperation::Divide,
            SaicAiOperation::Convolution,
            SaicAiOperation::Pooling,
            SaicAiOperation::Activation,
            SaicAiOperation::BatchNorm,
            SaicAiOperation::Dropout,
            SaicAiOperation::MatMul,
            SaicAiOperation::Transpose,
            SaicAiOperation::Reshape,
            SaicAiOperation::Attention,
            SaicAiOperation::MultiHeadAttention,
            SaicAiOperation::SelfAttention,
            SaicAiOperation::Transformer,
            SaicAiOperation::Encoder,
            SaicAiOperation::Decoder,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saic_operation_metadata() {
        let add_op = SaicAiOperation::Add;
        let metadata = add_op.metadata();
        
        assert_eq!(metadata.name, "Add");
        assert_eq!(metadata.description, "Element-wise addition of two tensors");
        assert!(metadata.hardware_compatibility.contains(&SaicHardwareType::Cpu));
        assert!(metadata.hardware_compatibility.contains(&SaicHardwareType::Gpu));
    }

    #[test]
    fn test_saic_operation_optimization_hints() {
        let conv_op = SaicAiOperation::Convolution;
        let hints = conv_op.optimization_hints();
        
        assert!(hints.contains(&SaicOptimizationHint::ComputeIntensive));
        assert!(hints.contains(&SaicOptimizationHint::TensorCoreOptimized));
        assert!(hints.contains(&SaicOptimizationHint::NeuralEngineOptimized));
    }

    #[test]
    fn test_saic_operation_hardware_support() {
        let add_op = SaicAiOperation::Add;
        
        assert!(add_op.is_supported_on(&SaicHardwareType::Cpu));
        assert!(add_op.is_supported_on(&SaicHardwareType::Gpu));
        assert!(add_op.is_supported_on(&SaicHardwareType::Npu));
        assert!(add_op.is_supported_on(&SaicHardwareType::Tpu));
        assert!(add_op.is_supported_on(&SaicHardwareType::NeuralEngine));
    }

    #[test]
    fn test_all_operations() {
        let operations = SaicAiOperation::all_operations();
        
        assert!(operations.contains(&SaicAiOperation::Add));
        assert!(operations.contains(&SaicAiOperation::Convolution));
        assert!(operations.contains(&SaicAiOperation::MatMul));
        assert!(operations.contains(&SaicAiOperation::Attention));
        assert!(operations.contains(&SaicAiOperation::Transformer));
    }
}
