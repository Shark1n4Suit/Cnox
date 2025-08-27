//! Fallback backend for SAIC inference engine
//! 
//! Provides real inference capabilities using SAIC execution engine

use crate::{InferenceBackend, InferenceError, HardwareRequirements};
use saic_core::{parser::parse_saic_model, execution::execute_saic_model};
use std::collections::HashMap;
use std::time::Instant;

/// Fallback backend for real inference using SAIC execution engine
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
    
    /// Real inference implementation using SAIC execution engine
    fn real_inference(&self, model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        let total_start = Instant::now();
        
        // Stage 1: Model Parsing
        let parse_start = Instant::now();
        let model = parse_saic_model(model_payload)
            .map_err(|e| InferenceError::ModelLoadingFailed(format!("Failed to parse SAIC model: {}", e)))?;
        let parse_time = parse_start.elapsed();
        println!("⏱️  Stage 1 - Model Parsing: {:.3}ms", parse_time.as_micros() as f64 / 1000.0);
        println!("DEBUG: Model parsed successfully");
        println!("DEBUG: Input names: {:?}", model.input_names());
        println!("DEBUG: Output names: {:?}", model.output_names());

        // Stage 2: Input Preparation
        let input_start = Instant::now();
        let mut inputs = HashMap::new();
        let input_names = model.input_names();
        let input_name = input_names.first()
            .ok_or_else(|| InferenceError::InputDataError("Model has no input tensors".to_string()))?;
        inputs.insert(input_name.to_string(), input_data.to_vec());
        let input_time = input_start.elapsed();
        println!("⏱️  Stage 2 - Input Preparation: {:.3}ms", input_time.as_micros() as f64 / 1000.0);
        println!("DEBUG: Inputs prepared: {:?}", inputs);

        // Stage 3: Model Execution
        let execution_start = Instant::now();
        let outputs = execute_saic_model(&model, inputs)
            .map_err(|e| InferenceError::ModelLoadingFailed(format!("Execution failed: {}", e)))?;
        let execution_time = execution_start.elapsed();
        println!("⏱️  Stage 3 - Model Execution: {:.3}ms", execution_time.as_micros() as f64 / 1000.0);
        println!("DEBUG: Model executed, outputs: {:?}", outputs);

        // Stage 4: Output Extraction
        let output_start = Instant::now();
        let output_names = model.output_names();
        let output_name = output_names.first()
            .ok_or_else(|| InferenceError::InputDataError("Model has no output tensors".to_string()))?;
        println!("DEBUG: Looking for output tensor: {}", output_name);

        let output_data = outputs.get(output_name.as_str())
            .ok_or_else(|| InferenceError::InputDataError("Output tensor not found".to_string()))?;
        let output_time = output_start.elapsed();
        println!("⏱️  Stage 4 - Output Extraction: {:.3}ms", output_time.as_micros() as f64 / 1000.0);

        let total_time = total_start.elapsed();
        println!("🚀 Total End-to-End Latency: {:.3}ms", total_time.as_micros() as f64 / 1000.0);
        println!("📊 Latency Breakdown:");
        println!("   • Model Parsing: {:.3}ms ({:.1}%)", parse_time.as_micros() as f64 / 1000.0, (parse_time.as_micros() as f64 / total_time.as_micros() as f64) * 100.0);
        println!("   • Input Preparation: {:.3}ms ({:.1}%)", input_time.as_micros() as f64 / 1000.0, (input_time.as_micros() as f64 / total_time.as_micros() as f64) * 100.0);
        println!("   • Model Execution: {:.3}ms ({:.1}%)", execution_time.as_micros() as f64 / 1000.0, (execution_time.as_micros() as f64 / total_time.as_micros() as f64) * 100.0);
        println!("   • Output Extraction: {:.3}ms ({:.1}%)", output_time.as_micros() as f64 / 1000.0, (output_time.as_micros() as f64 / total_time.as_micros() as f64) * 100.0);

        Ok(output_data.clone())
    }
}

impl InferenceBackend for FallbackBackend {
    fn execute(&self, model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // Auto-initialize if not already initialized
        if !self.initialized {
            // For fallback backend, we can safely initialize here
            // In a real implementation, this would be handled differently
        }
        
        self.real_inference(model_payload, input_data)
    }
    
    fn name(&self) -> &str {
        "fallback-saic"
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["SAIC".to_string()]
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