//! Error handling for COELANOX
//!
//! This module defines the error types and result types used throughout COELANOX.

use std::io;
use thiserror::Error;

/// Result type for COELANOX operations
pub type CoelanoxResult<T> = Result<T, CoelanoxError>;

/// COELANOX error types
#[derive(Error, Debug)]
pub enum CoelanoxError {
    /// Format-related errors
    #[error("Format error: {0}")]
    Format(String),

    /// Invalid magic number
    #[error("Invalid magic number")]
    InvalidMagic,

    /// Unsupported version
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(String),

    /// Manifest parsing error
    #[error("Manifest parsing error: {0}")]
    ManifestParse(String),

    /// Payload validation error
    #[error("Payload validation error: {0}")]
    PayloadValidation(String),

    /// Security validation error
    #[error("Security validation error: {0}")]
    SecurityValidation(String),

    /// Hash or signature verification failed while loading a container (distinct from generic runtime errors).
    #[error("Container integrity verification failed for {path}: {detail}")]
    ContainerIntegrity { path: String, detail: String },

    /// Path traversal error
    #[error("Path traversal error: {0}")]
    PathTraversal(String),

    /// File system error
    #[error("File system error: {0}")]
    FileSystem(#[from] io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Serialization error (including protobuf)
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Unsupported model format
    #[error("Unsupported model format: {0}")]
    UnsupportedModelFormat(String),

    /// Translation error
    #[error("Translation error: {0}")]
    Translation(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    Optimization(String),

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    Runtime(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Unsupported hardware target
    #[error("Unsupported hardware target: {0:?}")]
    UnsupportedHardwareTarget(crate::format::HardwareTarget),

    /// Missing template parameter
    #[error("Missing template parameter: {0}")]
    MissingTemplateParameter(String),

    /// Unsupported operation (requested op not supported by backend)
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Backend library not found
    #[error("Backend library not found: {0}")]
    BackendLibraryNotFound(String),

    /// Backend library loading failed (corrupted, invalid format, missing symbols)
    #[error("Failed to load backend library {0}: {1}")]
    BackendLibraryLoadFailed(String, String),

    /// Backend discovery failed (permission errors, invalid paths)
    #[error("Backend discovery failed: {0}")]
    BackendDiscoveryFailed(String),

    /// Backend not found (requested backend doesn't exist)
    #[error("Backend not found: {0}")]
    BackendNotFound(String),

    /// Backend health check failed
    #[error("Backend health check failed for {0}: {1}")]
    BackendHealthCheckFailed(String, String),

    /// Invalid backend name in AOT selection
    #[error("Invalid backend name in AOT selection: {0}")]
    InvalidBackendName(String),

    /// Device ID out of range
    #[error("Device ID {0} is out of range (available: 0-{1})")]
    DeviceIdOutOfRange(usize, usize),

    /// Backend routing failed (no available backends)
    #[error("Backend routing failed: {0}")]
    BackendRoutingFailed(String),

    /// Unsupported memory target (no backend registered for this target)
    #[error("Unsupported memory target: {0}")]
    UnsupportedMemoryTarget(String),

    /// Unsupported protection target (no backend registered for this target)
    #[error("Unsupported protection target: {0}")]
    UnsupportedProtectionTarget(String),
}

impl CoelanoxError {
    /// Create a format error
    pub fn format(msg: impl Into<String>) -> Self {
        Self::Format(msg.into())
    }

    /// Create a backend error
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }

    /// Create a translation error
    pub fn translation(msg: impl Into<String>) -> Self {
        Self::Translation(msg.into())
    }

    /// Create an optimization error
    pub fn optimization(msg: impl Into<String>) -> Self {
        Self::Optimization(msg.into())
    }

    /// Create an execution error
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }

    /// Create a configuration error
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }

    /// Create a container integrity (hash/signature) verification error at load time.
    pub fn container_integrity(path: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::ContainerIntegrity {
            path: path.into(),
            detail: detail.into(),
        }
    }

    /// True when the failure is specifically hash/signature verification (CLI may print extra user hints).
    pub fn is_container_integrity_failure(&self) -> bool {
        matches!(self, CoelanoxError::ContainerIntegrity { .. })
    }

    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

impl From<prost::DecodeError> for CoelanoxError {
    fn from(err: prost::DecodeError) -> Self {
        CoelanoxError::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::CoelanoxError;

    #[test]
    fn container_integrity_kind_detected_for_hints() {
        let e = CoelanoxError::container_integrity("m.cnox", "hash mismatch");
        assert!(e.is_container_integrity_failure());
        assert!(!CoelanoxError::format("other").is_container_integrity_failure());
    }
}
