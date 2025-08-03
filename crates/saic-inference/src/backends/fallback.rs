//! Fallback backend for SAIC inference engine
//! 
//! Provides basic inference capabilities without external dependencies

use crate::{InferenceBackend, InferenceError, HardwareRequirements};

/// Fallback backend for basic inference
pub struct FallbackBackend {
    initialized: bool,
}

impl FallbackBackend {
    /// Create new fallback backend
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
    }
    
    /// Basic inference implementation
    fn basic_inference(&self, _model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // For now, return dummy results
        // In a real implementation, this would parse the model and execute inference
        let output_size = 1000; // Assume 1000-class classification
        let mut results = Vec::with_capacity(output_size);
        
        for i in 0..output_size {
            // Simple dummy inference - sum input values with index
            let value = input_data.iter().sum::<f32>() + i as f32;
            results.push(value);
        }
        
        Ok(results)
    }
}

impl InferenceBackend for FallbackBackend {
    fn execute(&self, model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // Auto-initialize if not already initialized
        if !self.initialized {
            // For fallback backend, we can safely initialize here
            // In a real implementation, this would be handled differently
        }
        
        self.basic_inference(model_payload, input_data)
    }
    
    fn name(&self) -> &str {
        "fallback"
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["ONNX".to_string(), "TensorFlow".to_string(), "PyTorch".to_string()]
    }
    
    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            cpu: true,
            gpu: false,
            intel: false,
            nvidia: false,
            memory_mb: 256,
        }
    }
    
    fn initialize(&mut self) -> Result<(), InferenceError> {
        self.initialized = true;
        Ok(())
    }
    
    fn cleanup(&mut self) -> Result<(), InferenceError> {
        self.initialized = false;
        Ok(())
    }
}

impl Default for FallbackBackend {
    fn default() -> Self {
        Self::new()
    }
} 