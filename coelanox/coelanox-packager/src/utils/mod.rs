//! Utility functions for the COELANOX packager
//! 
//! This module provides utility functions used by the packager components.

use std::path::Path;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;

/// Format detection utilities
pub mod format_detection {
    use super::*;
    
    /// Detect model format from file extension
    pub fn detect_from_extension(path: &Path) -> Option<ModelFormat> {
        let extension = path.extension()?.to_string_lossy().to_lowercase();
        
        match extension.as_str() {
            "onnx" => Some(ModelFormat::Onnx),
            "cnox" => Some(ModelFormat::Coelanox),
            // ONNX only; PyTorch (.pt/.pth) and TensorFlow (.pb) not supported
            _ => None,
        }
    }
    
    /// Detect model format from file content
    pub fn detect_from_content(data: &[u8]) -> Option<ModelFormat> {
        // Check for COELANOX magic number
        if data.len() >= 4 && &data[0..4] == b"COEL" {
            return Some(ModelFormat::Coelanox);
        }
        
        // Check for ONNX magic number
        if data.len() >= 8 && &data[0..8] == b"\x08\x00\x00\x00\x00\x00\x00\x00" {
            return Some(ModelFormat::Onnx);
        }

        None
    }
    
    /// Detect model format from file
    pub fn detect_format(path: &Path) -> CoelanoxResult<ModelFormat> {
        // Try extension first
        if let Some(format) = detect_from_extension(path) {
            return Ok(format);
        }
        
        // Try content detection
        let data = std::fs::read(path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        if let Some(format) = detect_from_content(&data) {
            return Ok(format);
        }
        
        // Fallback to custom
        Ok(ModelFormat::Custom("unknown".to_string()))
    }
}

/// Performance utilities
pub mod performance {
    use std::time::{Instant, Duration};
    
    /// Performance timer
    pub struct Timer {
        start: Instant,
    }
    
    impl Timer {
        /// Start a new timer
        pub fn start() -> Self {
            Self {
                start: Instant::now(),
            }
        }
        
        /// Get elapsed time
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        /// Get elapsed time in microseconds
        pub fn elapsed_micros(&self) -> u64 {
            self.elapsed().as_micros() as u64
        }
        
        /// Get elapsed time in milliseconds
        pub fn elapsed_millis(&self) -> u64 {
            self.elapsed().as_millis() as u64
        }
    }
    
    /// Benchmark a function
    pub fn benchmark<F, R>(func: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let timer = Timer::start();
        let result = func();
        let duration = timer.elapsed();
        (result, duration)
    }
    
    /// Benchmark a function multiple times
    pub fn benchmark_multiple<F, R>(func: F, iterations: usize) -> (R, Duration, Duration)
    where
        F: Fn() -> R,
    {
        let mut times = Vec::with_capacity(iterations);
        let mut last_result = None;
        
        for _ in 0..iterations {
            let timer = Timer::start();
            let result = func();
            let duration = timer.elapsed();
            
            times.push(duration);
            last_result = Some(result);
        }
        
        let total_time: Duration = times.iter().sum();
        let avg_time = total_time / iterations as u32;
        
        (last_result.unwrap(), total_time, avg_time)
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate input shape
    pub fn validate_input_shape(shape: &[usize]) -> CoelanoxResult<()> {
        if shape.is_empty() {
            return Err(CoelanoxError::format("Input shape cannot be empty"));
        }
        
        for &dim in shape {
            if dim == 0 {
                return Err(CoelanoxError::format("Input shape cannot contain zero dimensions"));
            }
        }
        
        Ok(())
    }
    
    /// Validate output shape
    pub fn validate_output_shape(shape: &[usize]) -> CoelanoxResult<()> {
        if shape.is_empty() {
            return Err(CoelanoxError::format("Output shape cannot be empty"));
        }
        
        for &dim in shape {
            if dim == 0 {
                return Err(CoelanoxError::format("Output shape cannot contain zero dimensions"));
            }
        }
        
        Ok(())
    }
    
    /// Calculate total elements in shape
    pub fn calculate_elements(shape: &[usize]) -> usize {
        shape.iter().product()
    }
    
    /// Validate file size
    pub fn validate_file_size(path: &Path, max_size: u64) -> CoelanoxResult<()> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        let size = metadata.len();
        
        if size > max_size {
            return Err(CoelanoxError::format(format!(
                "File too large: {} bytes (max: {} bytes)",
                size, max_size
            )));
        }
        
        Ok(())
    }
}

/// Memory utilities
pub mod memory {
    use std::alloc::{alloc, dealloc, Layout};
    
    /// Allocate aligned memory
    pub unsafe fn allocate_aligned(size: usize, alignment: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, alignment)
            .expect("Invalid layout");
        alloc(layout)
    }
    
    /// Deallocate aligned memory
    pub unsafe fn deallocate_aligned(ptr: *mut u8, size: usize, alignment: usize) {
        let layout = Layout::from_size_align(size, alignment)
            .expect("Invalid layout");
        dealloc(ptr, layout);
    }
    
    /// Zero-copy slice from memory. Requires correct alignment for T; ARM will crash
    /// on unaligned access, while x86 may tolerate it. Always validate alignment.
    pub fn slice_from_memory<T>(data: &[u8], offset: usize, len: usize) -> &[T] {
        let start = offset;
        let end = offset + len * std::mem::size_of::<T>();
        
        if end > data.len() {
            panic!("Slice out of bounds");
        }
        
        let slice = &data[start..end];
        let ptr = slice.as_ptr() as *const T;
        let align = std::mem::align_of::<T>();
        if (ptr as usize) % align != 0 {
            panic!(
                "slice_from_memory: misaligned pointer for {} (align {} required, ptr {:p})",
                std::any::type_name::<T>(),
                align,
                ptr
            );
        }
        unsafe {
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

/// Logging utilities
pub mod logging {
    use tracing::{info, warn, error, debug};
    
    /// Log packaging start
    pub fn log_packaging_start(model_path: &str, target: &str) {
        info!("Starting packaging: {} -> {}", model_path, target);
    }
    
    /// Log packaging step
    pub fn log_packaging_step(step: &str, duration_ms: u64) {
        info!("Packaging step '{}' completed in {}ms", step, duration_ms);
    }
    
    /// Log packaging completion
    pub fn log_packaging_complete(output_path: &str, total_duration_ms: u64) {
        info!("Packaging completed: {} ({}ms)", output_path, total_duration_ms);
    }
    
    /// Log optimization step
    pub fn log_optimization_step(optimizer: &str, target: &str, duration_ms: u64) {
        info!("Optimization '{}' for '{}' completed in {}ms", optimizer, target, duration_ms);
    }
    
    /// Log code generation step
    pub fn log_code_generation_step(generator: &str, target: &str, duration_ms: u64) {
        info!("Code generation '{}' for '{}' completed in {}ms", generator, target, duration_ms);
    }
    
    /// Log error
    pub fn log_error(error: &str) {
        error!("Packaging error: {}", error);
    }
    
    /// Log warning
    pub fn log_warning(warning: &str) {
        warn!("Packaging warning: {}", warning);
    }
    
    /// Log debug info
    pub fn log_debug(debug_info: &str) {
        debug!("Packaging debug: {}", debug_info);
    }
}
