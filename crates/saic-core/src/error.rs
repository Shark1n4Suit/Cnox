//! Error types for SAIC operations

use thiserror::Error;

/// Custom error types for SAIC operations
/// 
/// This enum provides specific, descriptive error types for different failure modes
/// during SAIC file processing, validation, and execution.
#[derive(Debug, Error)]
pub enum SaicError {
    /// I/O operation failed (file reading, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Invalid file path provided
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    
    /// SAIC file format error
    #[error("Invalid SAIC format: {0}")]
    InvalidFormat(String),
    
    /// Invalid magic number in SAIC header
    #[error("Invalid magic number: {0}")]
    InvalidMagic(String),
    
    /// Unsupported or invalid version
    #[error("Invalid version: {0}")]
    InvalidVersion(String),
    
    /// JSON manifest parsing error
    #[error("Manifest parsing error: {0}")]
    ManifestError(#[from] serde_json::Error),
    
    /// File is not a valid SAIC file
    #[error("Not a SAIC file: {0}")]
    NotSaicFile(String),
    
    /// Memory mapping failed
    #[error("Memory mapping failed: {0}")]
    MemoryMappingFailed(String),
    
    /// SIMD operation failed due to invalid data
    #[error("SIMD operation failed: {0}")]
    SimdOperationFailed(String),
    
    /// File size validation failed
    #[error("File size validation failed: {0}")]
    FileSizeValidationFailed(String),
    
    /// Data alignment validation failed
    #[error("Data alignment failed: {0}")]
    DataAlignmentFailed(String),
    
    /// Security validation failed
    #[error("Security validation failed: {0}")]
    SecurityError(String),
    
    /// Inference operation failed
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    /// Execution timeout exceeded
    #[error("Execution timeout: {0}")]
    ExecutionTimeout(String),
}

/// Result type for SAIC operations
pub type SaicResult<T> = Result<T, SaicError>; 