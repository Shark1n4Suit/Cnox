//! SAIC Inference Engine Library
//! 
//! Provides AI model inference capabilities with hardware optimization

pub mod hardware;
pub mod backends;
pub mod error;
pub mod optimization;

use std::collections::HashMap;
use error::InferenceError;

pub use hardware::HardwareInfo;
pub use backends::BackendRegistry;

/// Hardware requirements for inference
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    pub cpu: bool,
    pub gpu: bool,
    pub intel: bool,
    pub nvidia: bool,
    pub memory_mb: usize,
}

/// Compiled model for caching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledModel {
    pub model_hash: u64,
    pub compiled_kernels: HashMap<String, Vec<u8>>,
    pub optimization_flags: Vec<String>,
}

/// Trait for inference backends
pub trait InferenceBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Check if backend supports the given model format
    fn supports_format(&self, format: &str) -> bool;
    
    /// Check hardware compatibility
    fn check_hardware(&self, requirements: &HardwareRequirements) -> Result<(), InferenceError>;
    
    /// Load and validate model
    fn load_model(&mut self, model_path: &str) -> Result<(), InferenceError>;
    
    /// Execute inference
    fn execute(&mut self, inputs: HashMap<String, Vec<f32>>) -> Result<HashMap<String, Vec<f32>>, InferenceError>;
    
    /// Get backend statistics
    fn get_stats(&self) -> BackendStats;
}

/// Backend execution statistics
#[derive(Debug, Clone)]
pub struct BackendStats {
    pub total_inferences: usize,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    pub memory_usage_mb: usize,
}
