//! SAIC Inference Engine
//! 
//! Universal inference engine with dynamic runtime loading

pub mod backends;
pub mod error;
pub mod hardware;
pub mod ffi;

use std::collections::HashMap;
use crate::error::InferenceError;
use crate::hardware::HardwareInfo;

/// Universal inference backend trait
pub trait InferenceBackend: Send + Sync {
    /// Execute inference on model payload with input data
    fn execute(&self, model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError>;
    
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get supported model formats
    fn supported_formats(&self) -> Vec<String>;
    
    /// Get hardware requirements
    fn hardware_requirements(&self) -> HardwareRequirements;
    
    /// Initialize backend (optional)
    fn initialize(&mut self) -> Result<(), InferenceError> {
        Ok(()) // Default implementation does nothing
    }
    
    /// Cleanup backend (optional)
    fn cleanup(&mut self) -> Result<(), InferenceError> {
        Ok(()) // Default implementation does nothing
    }
}

/// Hardware requirements for backend
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    pub cpu: bool,
    pub gpu: bool,
    pub intel: bool,
    pub nvidia: bool,
    pub memory_mb: usize,
}

impl Default for HardwareRequirements {
    fn default() -> Self {
        Self {
            cpu: true,
            gpu: false,
            intel: false,
            nvidia: false,
            memory_mb: 512,
        }
    }
}

impl HardwareRequirements {
    /// Check if hardware requirements are satisfied
    pub fn is_satisfied_by(&self, hardware: &HardwareInfo) -> bool {
        // Check CPU requirement
        if self.cpu && !hardware.has_cpu {
            return false;
        }
        
        // Check GPU requirement
        if self.gpu && !hardware.has_gpu {
            return false;
        }
        
        // Check Intel requirement
        if self.intel && !hardware.is_intel {
            return false;
        }
        
        // Check NVIDIA requirement
        if self.nvidia && !hardware.is_nvidia {
            return false;
        }
        
        // Check memory requirement
        if hardware.available_memory_mb < self.memory_mb {
            return false;
        }
        
        true
    }
}

/// Backend registry for managing available inference backends
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    /// Create new backend registry with available backends
    pub fn new() -> Self {
        let mut registry = Self { backends: HashMap::new() };
        
        // Always add fallback backend
        registry.backends.insert("fallback".to_string(), Box::new(backends::fallback::FallbackBackend::new()));
        
        // Dynamically initialize optional backends
        #[cfg(feature = "openvino")]
        if let Some(openvino) = backends::initialize_openvino() {
            registry.backends.insert("openvino".to_string(), openvino);
        }
        
        #[cfg(feature = "tensorrt")]
        if let Some(tensorrt) = backends::initialize_tensorrt() {
            registry.backends.insert("tensorrt".to_string(), tensorrt);
        }
        
        registry
    }
    
    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> Option<&dyn InferenceBackend> {
        self.backends.get(name).map(|b| b.as_ref())
    }
    
    /// Get all available backend names
    pub fn available_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }
    
    /// Select optimal backend for model format and hardware
    pub fn select_backend(&self, model_format: &str, hardware: &HardwareInfo) -> &dyn InferenceBackend {
        // 1. Try optimal backend for model format
        let optimal_backend = match model_format {
            "ONNX" => self.backends.get("tensorrt").or_else(|| self.backends.get("openvino")),
            "TensorFlow" => self.backends.get("openvino"),
            _ => None,
        };
        
        // 2. Check hardware compatibility
        if let Some(backend) = optimal_backend {
            if backend.hardware_requirements().is_satisfied_by(hardware) {
                return backend.as_ref();
            }
        }
        
        // 3. Try any compatible backend
        for backend in self.backends.values() {
            if backend.hardware_requirements().is_satisfied_by(hardware) {
                return backend.as_ref();
            }
        }
        
        // 4. Final fallback
        self.backends.get("fallback").unwrap().as_ref()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
} 