//! Backend implementations for different hardware targets
//! 
//! This module provides backend implementations for CPU, GPU, NPU, and TPU targets.

pub mod cpu_backend;
pub mod gpu_backend;
pub mod npu_backend;
pub mod tpu_backend;

pub use cpu_backend::*;
pub use gpu_backend::*;
pub use npu_backend::*;
pub use tpu_backend::*;
