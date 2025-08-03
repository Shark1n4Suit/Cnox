//! FFI bindings for external runtimes

#[cfg(feature = "openvino")]
pub mod openvino;

#[cfg(feature = "tensorrt")]
pub mod tensorrt; 