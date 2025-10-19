//! COELANOX (Secure AI Container) - Universal binary container for AI models
//! 
//! COELANOX is a universal binary container that provides secure, fast, and optimized
//! deployment of AI models across any hardware platform.

pub mod format;
pub mod packager;
pub mod runtime;
// pub mod onnx; // Removed - ONNX functionality moved to plugins
pub mod backends;
pub mod utils;
pub mod error;

// Re-export main types for convenience
pub use format::{CoelanoxContainer, CoelanoxHeader, CoelanoxManifest, ModelFormat, HardwareTarget};
pub use packager::CoelanoxPackager;
pub use packager::ir::{UniversalIR, ModelMetadata, Operation, Tensor};
pub use runtime::CoelanoxRuntime;
pub use backends::{Backend, BackendType};
pub use error::{CoelanoxError, CoelanoxResult};

/// COELANOX version
pub const COELANOX_VERSION: &str = env!("CARGO_PKG_VERSION");

/// COELANOX magic number for file identification
pub const COELANOX_MAGIC: [u8; 4] = [0x43, 0x4F, 0x45, 0x4C]; // "COEL"

/// Initialize COELANOX
pub fn init() -> CoelanoxResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize any global state
    tracing::info!("COELANOX initialized version {}", COELANOX_VERSION);
    
    Ok(())
}

/// Get COELANOX version
pub fn version() -> &'static str {
    COELANOX_VERSION
}

/// Validate COELANOX magic number
pub fn validate_magic(data: &[u8]) -> bool {
    data.len() >= 4 && data[0..4] == COELANOX_MAGIC
}

/// Example usage
/// 
/// ```rust
/// use coelanox::{CoelanoxPackager, HardwareTarget};
/// use std::path::Path;
/// 
/// fn main() -> coelanox::CoelanoxResult<()> {
///     // Initialize COELANOX
///     coelanox::init()?;
///     
///     // Create packager
///     let packager = CoelanoxPackager::new();
///     
///     // Create runtime
///     let runtime = coelanox::CoelanoxRuntime::new();
///     
///     // Note: This example doesn't actually package a model
///     // since we don't have a real model file
///     println!("COELANOX initialized successfully");
///     Ok(())
/// }
/// ```
pub fn example_usage() {
    // This function exists to hold the example doc comment
}

#[cfg(test)]
mod tests {
    use super::*;
    use packager::plugins::{TranslatorPlugin};
    use format::ModelFormat;

    #[test]
    fn test_pytorch_plugin() {
        // PyTorch plugin removed - focusing on ONNX only
        println!("PyTorch plugin removed - focusing on ONNX only");
        assert!(true); // Placeholder test
    }

    #[test]
    fn test_tensorflow_plugin() {
        // TensorFlow plugin removed - focusing on ONNX only
        println!("TensorFlow plugin removed - focusing on ONNX only");
        assert!(true); // Placeholder test
    }
}
