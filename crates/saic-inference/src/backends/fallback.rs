//! Fallback Backend for SAIC Inference Engine
//! 
//! Provides basic inference capabilities when optimized backends are unavailable

use crate::{InferenceBackend, InferenceError, HardwareRequirements, BackendStats};
use std::collections::HashMap;

/// Fallback backend for basic inference
pub struct FallbackBackend {
    initialized: bool,
    model_loaded: bool,
    stats: BackendStats,
}

impl FallbackBackend {
    /// Create a new fallback backend
    pub fn new() -> Self {
        Self {
            initialized: false,
            model_loaded: false,
            stats: BackendStats {
                total_inferences: 0,
                total_time_ms: 0.0,
                avg_time_ms: 0.0,
                memory_usage_mb: 0,
            },
        }
    }
}

impl InferenceBackend for FallbackBackend {
    fn name(&self) -> &str {
        "fallback"
    }
    
    fn supports_format(&self, format: &str) -> bool {
        format == "SAIC"
    }
    
    fn check_hardware(&self, requirements: &HardwareRequirements) -> Result<(), InferenceError> {
        // Fallback backend has minimal requirements
        if requirements.memory_mb > 256 {
            return Err(InferenceError::HardwareIncompatible(
                "Insufficient memory for fallback backend".to_string()
            ));
        }
        Ok(())
    }
    
    fn load_model(&mut self, model_path: &str) -> Result<(), InferenceError> {
        // Basic model loading validation
        if model_path.is_empty() {
            return Err(InferenceError::ModelLoadingFailed(
                "Model path cannot be empty".to_string()
            ));
        }
        self.model_loaded = true;
        Ok(())
    }
    
    fn execute(&mut self, inputs: HashMap<String, Vec<f32>>) -> Result<HashMap<String, Vec<f32>>, InferenceError> {
        if !self.model_loaded {
            return Err(InferenceError::ExecutionFailed(
                "Model not loaded".to_string()
            ));
        }
        
        // Update statistics
        self.stats.total_inferences += 1;
        self.stats.total_time_ms += 1.0; // Simulated execution time
        self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.total_inferences as f64;
        
        // Return dummy output for now
        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), vec![0.0; 1000]); // 1000-class output
        
        Ok(outputs)
    }
    
    fn get_stats(&self) -> BackendStats {
        self.stats.clone()
    }
} 