//! SAIC Core Library
//! 
//! This library provides the core functionality for SAIC (Secure AI Container),
//! including file format specification, validation, security checks, and
//! performance optimization utilities.

pub mod format;
pub mod validation; 
pub mod security;
pub mod error;
pub mod memory;
pub mod optimization;

// Re-export main types for easy access
pub use format::*;
pub use validation::*;
pub use security::*;
pub use error::*;
pub use memory::*;
pub use optimization::*;

/// SAIC Core Library Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// SAIC File Format Magic Number
pub const MAGIC_NUMBER: &[u8; 4] = b"SAIC";

/// SAIC File Format Version
pub const FORMAT_VERSION: [u8; 4] = [0, 1, 0, 0];

/// Maximum manifest size (1MB)
pub const MAX_MANIFEST_SIZE: usize = 1024 * 1024;

/// Minimum file size for SAIC header
pub const MIN_FILE_SIZE: usize = 12; 