//! Fallback Backend for SAIC Inference Engine
//! 
//! Provides basic inference capabilities when optimized backends are unavailable

use crate::{InferenceBackend, InferenceError, HardwareRequirements, BackendStats};
use saic_core::{parser::parse_saic_model, execution::execute_saic_model};
use std::collections::HashMap;
use std::time::Instant;

/// Fallback backend for basic inference
pub struct FallbackBackend {
    model_loaded: bool,
    model_data: Option<Vec<u8>>,
    stats: BackendStats,
}

impl FallbackBackend {
    /// Create a new fallback backend
    pub fn new() -> Self {
        Self {
            model_loaded: false,
            model_data: None,
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
        "fallback-saic"
    }
    
    fn supports_format(&self, format: &str) -> bool {
        format == "SAIC" || format == "Universal"
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
        // Load the model file
        let model_data = std::fs::read(model_path)
            .map_err(|e| InferenceError::ModelLoadingFailed(format!("Failed to read model: {}", e)))?;
        
        // Validate it's a SAIC model by attempting to parse
        let _model = parse_saic_model(&model_data)
            .map_err(|e| InferenceError::ModelLoadingFailed(format!("Failed to parse SAIC model: {}", e)))?;
        
        self.model_data = Some(model_data);
        self.model_loaded = true;
        
        Ok(())
    }
    
    fn execute(&mut self, inputs: HashMap<String, Vec<f32>>) -> Result<HashMap<String, Vec<f32>>, InferenceError> {
        if !self.model_loaded {
            return Err(InferenceError::ExecutionFailed(
                "Model not loaded".to_string()
            ));
        }
        
        let start_time = Instant::now();
        
        // Get the model data
        let model_data = self.model_data.as_ref()
            .ok_or_else(|| InferenceError::ExecutionFailed("Model data not available".to_string()))?;
        
        // Parse the model
        let model = parse_saic_model(model_data)
            .map_err(|e| InferenceError::ExecutionFailed(format!("Failed to parse model: {}", e)))?;
        
        // Get the actual input tensor name from the model
        let input_names = model.input_names();
        let input_name = input_names.first()
            .ok_or_else(|| InferenceError::ExecutionFailed("Model has no input tensors".to_string()))?;
        
        // Create inputs with the correct tensor name
        let mut model_inputs = HashMap::new();
        let input_data = inputs.values().next()
            .ok_or_else(|| InferenceError::ExecutionFailed("No input data provided".to_string()))?;
        model_inputs.insert(input_name.to_string(), input_data.clone());
        
        // Debug: Print what we're actually sending
        println!("🔍 Fallback Backend Debug:");
        println!("   Model input names: {:?}", input_names);
        println!("   Using input name: {}", input_name);
        println!("   Input data size: {} floats", input_data.len());
        
        // Execute inference using SAIC core
        let outputs = execute_saic_model(&model, model_inputs)
            .map_err(|e| InferenceError::ExecutionFailed(format!("SAIC execution failed: {}", e)))?;
        
        // Update statistics
        let execution_time = start_time.elapsed();
        self.stats.total_inferences += 1;
        self.stats.total_time_ms += execution_time.as_millis() as f64;
        self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.total_inferences as f64;
        self.stats.memory_usage_mb = (model_data.len() / (1024 * 1024)) as usize;
        
        Ok(outputs)
    }
    
    fn get_stats(&self) -> BackendStats {
        self.stats.clone()
    }
} 