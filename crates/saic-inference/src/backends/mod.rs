//! Backend implementations for SAIC inference engine

pub mod fallback;
pub mod native;

/// Initialize Native SAIC backend (primary)
pub fn initialize_native_saic() -> Option<Box<dyn crate::InferenceBackend>> {
    // Handle the Result return type from new()
    match native::NativeSAICBackend::new() {
        Ok(backend) => Some(Box::new(backend) as Box<dyn crate::InferenceBackend>),
        Err(e) => {
            eprintln!("⚠️  Native SAIC backend initialization failed: {}", e);
            None
        }
    }
}

/// Initialize Fallback backend (secondary)
pub fn initialize_fallback() -> Option<Box<dyn crate::InferenceBackend>> {
    Some(Box::new(fallback::FallbackBackend::new()) as Box<dyn crate::InferenceBackend>)
} 