//! COELANOX backends
//! 
//! This module defines the backend interface for COELANOX execution.

use crate::{CoelanoxError, CoelanoxResult};
use crate::format::{CoelanoxContainer, HardwareTarget};

/// Backend trait for COELANOX execution
pub trait Backend: Send + Sync {
    /// Execute a container
    fn execute(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>>;
    
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get supported hardware targets
    fn supported_targets(&self) -> Vec<HardwareTarget>;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
}

/// Backend type
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// OpenVINO backend
    OpenVINO,
    /// TensorRT backend
    TensorRT,
    /// Fallback backend
    Fallback,
}

/// Fallback backend (always available)
pub struct FallbackBackend {
    /// Backend name
    name: String,
}

impl FallbackBackend {
    /// Create a new fallback backend
    pub fn new() -> Self {
        Self {
            name: "fallback_backend".to_string(),
        }
    }
}

impl Backend for FallbackBackend {
    fn execute(&self, container: &CoelanoxContainer, _input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // TODO: Implement fallback execution
        // For now, return dummy output
        let output_size = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        Ok(vec![0.0f32; output_size])
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Cpu]
    }
    
    fn is_available(&self) -> bool {
        true // Always available
    }
}

/// OpenVINO backend
pub struct OpenVINOBackend {
    /// Backend name
    name: String,
}

impl OpenVINOBackend {
    /// Create a new OpenVINO backend
    pub fn new() -> Self {
        Self {
            name: "openvino_backend".to_string(),
        }
    }
}

impl Backend for OpenVINOBackend {
    fn execute(&self, _container: &CoelanoxContainer, _input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // TODO: Implement OpenVINO execution
        Err(CoelanoxError::backend("OpenVINO backend not yet implemented"))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Cpu, HardwareTarget::Gpu]
    }
    
    fn is_available(&self) -> bool {
        // TODO: Check if OpenVINO is available
        false
    }
}

/// TensorRT backend
pub struct TensorRTBackend {
    /// Backend name
    name: String,
}

impl TensorRTBackend {
    /// Create a new TensorRT backend
    pub fn new() -> Self {
        Self {
            name: "tensorrt_backend".to_string(),
        }
    }
}

impl Backend for TensorRTBackend {
    fn execute(&self, _container: &CoelanoxContainer, _input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // TODO: Implement TensorRT execution
        Err(CoelanoxError::backend("TensorRT backend not yet implemented"))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn supported_targets(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Gpu]
    }
    
    fn is_available(&self) -> bool {
        // TODO: Check if TensorRT is available
        false
    }
}


