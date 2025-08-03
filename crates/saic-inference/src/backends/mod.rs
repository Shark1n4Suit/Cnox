//! Backend implementations for SAIC inference engine

pub mod fallback;
pub mod openvino;
pub mod tensorrt;

/// Initialize OpenVINO backend if available
#[cfg(feature = "openvino")]
pub fn initialize_openvino() -> Option<Box<dyn crate::InferenceBackend>> {
    openvino::OpenVINOBackend::new().map(|b| Box::new(b) as Box<dyn crate::InferenceBackend>)
}

/// Initialize TensorRT backend if available
#[cfg(feature = "tensorrt")]
pub fn initialize_tensorrt() -> Option<Box<dyn crate::InferenceBackend>> {
    tensorrt::TensorRTBackend::new().map(|b| Box::new(b) as Box<dyn crate::InferenceBackend>)
}

/// Stub implementations when features are disabled
#[cfg(not(feature = "openvino"))]
pub fn initialize_openvino() -> Option<Box<dyn crate::InferenceBackend>> {
    None
}

#[cfg(not(feature = "tensorrt"))]
pub fn initialize_tensorrt() -> Option<Box<dyn crate::InferenceBackend>> {
    None
} 