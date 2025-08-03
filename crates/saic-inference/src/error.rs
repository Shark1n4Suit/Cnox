//! Error types for SAIC inference engine

use thiserror::Error;

/// Inference engine error types
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Backend error: {0}")]
    BackendError(String),
    
    #[error("Model format not supported: {0}")]
    UnsupportedFormat(String),
    
    #[error("Hardware not compatible: {0}")]
    HardwareIncompatible(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    
    #[error("Input data error: {0}")]
    InputDataError(String),
    
    #[error("Output data error: {0}")]
    OutputDataError(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    #[error("Runtime library not found: {0}")]
    RuntimeNotFound(String),
    
    #[error("Initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<String> for InferenceError {
    fn from(err: String) -> Self {
        InferenceError::Unknown(err)
    }
}

impl From<&str> for InferenceError {
    fn from(err: &str) -> Self {
        InferenceError::Unknown(err.to_string())
    }
} 