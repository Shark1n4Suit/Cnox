//! SAIC (Secure AI Container) - Universal binary container for AI models
//! 
//! SAIC is a universal binary container that provides secure, fast, and optimized
//! deployment of AI models across any hardware platform.

pub mod format;
pub mod packager;
pub mod runtime;
pub mod onnx;
pub mod backends;
pub mod utils;
pub mod error;

// Re-export main types for convenience
pub use format::{SaicContainer, SaicHeader, SaicManifest, ModelFormat, HardwareTarget};
pub use packager::SaicPackager;
pub use packager::ir::{UniversalIR, ModelMetadata, Operation, Tensor};
pub use runtime::SaicRuntime;
pub use backends::{Backend, BackendType};
pub use error::{SaicError, SaicResult};

/// SAIC version
pub const SAIC_VERSION: &str = env!("CARGO_PKG_VERSION");

/// SAIC magic number for file identification
pub const SAIC_MAGIC: [u8; 4] = [0x53, 0x41, 0x49, 0x43]; // "SAIC"

/// Initialize SAIC
pub fn init() -> SaicResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Initialize any global state
    tracing::info!("SAIC initialized version {}", SAIC_VERSION);
    
    Ok(())
}

/// Get SAIC version
pub fn version() -> &'static str {
    SAIC_VERSION
}

/// Validate SAIC magic number
pub fn validate_magic(data: &[u8]) -> bool {
    data.len() >= 4 && data[0..4] == SAIC_MAGIC
}

/// Example usage
/// 
/// ```rust
/// use saic::{SaicPackager, HardwareTarget};
/// use std::path::Path;
/// 
/// fn main() -> saic::SaicResult<()> {
///     // Initialize SAIC
///     saic::init()?;
///     
///     // Create packager
///     let packager = SaicPackager::new();
///     
///     // Create runtime
///     let runtime = saic::SaicRuntime::new();
///     
///     // Note: This example doesn't actually package a model
///     // since we don't have a real model file
///     println!("SAIC initialized successfully");
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
