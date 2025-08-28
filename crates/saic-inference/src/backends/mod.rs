//! Backend implementations for SAIC inference engine

pub mod fallback;
pub mod native;

use std::collections::HashMap;
use crate::InferenceBackend;
use crate::hardware::HardwareInfo;

/// Registry for managing available inference backends
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        let mut registry = Self {
            backends: HashMap::new(),
        };
        
        // Register available backends
        if let Some(native) = initialize_native_saic() {
            registry.backends.insert("native".to_string(), native);
        }
        
        if let Some(fallback) = initialize_fallback() {
            registry.backends.insert("fallback".to_string(), fallback);
        }
        
        registry
    }
    
    /// Get available backend names
    pub fn available_backends(&self) -> Vec<&String> {
        self.backends.keys().collect()
    }
    
    /// Check if registry is healthy
    pub fn is_healthy(&self) -> bool {
        !self.backends.is_empty()
    }
    
    /// Get backend by name
    pub fn get_backend(&mut self, name: &str) -> Option<&mut Box<dyn InferenceBackend>> {
        self.backends.get_mut(name)
    }
    
    /// Get recommended backend based on hardware
    pub fn get_recommended_backend(&self, hardware: &HardwareInfo) -> Option<&str> {
        // Prefer native backend if available
        if self.backends.contains_key("native") {
            Some("native")
        } else if self.backends.contains_key("fallback") {
            Some("fallback")
        } else {
            None
        }
    }
}

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