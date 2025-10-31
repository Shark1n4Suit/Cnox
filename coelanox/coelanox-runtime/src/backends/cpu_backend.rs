//! CPU backend implementation with SIMD optimizations
//! 
//! This module provides optimized CPU execution for COELANOX containers,
//! including SIMD acceleration and multi-threading support.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::CoelanoxContainer;
use coelanox_core::ir::{Operation, OpType, Attribute};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// CPU backend for executing operations with SIMD optimizations
pub struct CpuBackend {
    /// Number of CPU cores available
    num_cores: usize,
    /// SIMD capabilities
    simd_capabilities: SimdCapabilities,
    /// Thread pool for parallel execution
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

/// SIMD capabilities detection
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// SSE2 support
    pub sse2: bool,
    /// SSE4.1 support
    pub sse41: bool,
    /// AVX support
    pub avx: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512 support
    pub avx512: bool,
    /// NEON support (ARM)
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse41: is_x86_feature_detected!("sse4.1"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512: is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse41: false,
                avx: false,
                avx2: false,
                avx512: false,
                neon: true, // ARM64 always has NEON
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse2: false,
                sse41: false,
                avx: false,
                avx2: false,
                avx512: false,
                neon: false,
            }
        }
    }
    
    /// Get the best SIMD instruction set available
    pub fn best_simd(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse41 {
            "SSE4.1"
        } else if self.sse2 {
            "SSE2"
        } else if self.neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        let num_cores = num_cpus::get();
        let simd_capabilities = SimdCapabilities::detect();
        
        // Create thread pool for parallel execution
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cores)
            .build()
            .ok()
            .map(Arc::new);
        
        info!("🚀 CPU Backend initialized:");
        info!("  Cores: {}", num_cores);
        info!("  SIMD: {}", simd_capabilities.best_simd());
        info!("  Thread pool: {}", if thread_pool.is_some() { "enabled" } else { "disabled" });
        
        Self {
            num_cores,
            simd_capabilities,
            thread_pool,
        }
    }
    
    /// Execute operations on CPU with SIMD optimizations
    pub fn execute(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        debug!("🖥️  CPU Backend executing with {} cores", self.num_cores);
        
        // Calculate output size
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        let mut output = vec![0.0f32; output_size];
        
        // Load weights from container
        let weights = container.decompress_weights()?;
        
        // Execute operations with CPU optimizations
        // Legacy operations field removed - using empty operations
        let empty_operations: Vec<serde_json::Value> = Vec::new();
        self.execute_operations_cpu(input_data, &mut output, &weights, &empty_operations)?;
        
        debug!("✅ CPU Backend execution completed: {} -> {} elements", input_data.len(), output.len());
        Ok(output)
    }
    
    /// Execute operations with CPU-specific optimizations
    fn execute_operations_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        for operation_json in operations {
            let operation: Operation = serde_json::from_value(operation_json.clone())
                .map_err(|e| CoelanoxError::Execution(format!("Failed to deserialize operation: {}", e)))?;
            
            match operation.op_type {
                OpType::Add => {
                    self.execute_add_cpu(input, output, weights)?;
                }
                OpType::Multiply => {
                    self.execute_multiply_cpu(input, output, weights)?;
                }
                OpType::Convolution => {
                    self.execute_convolution_cpu(input, output, weights, &operation)?;
                }
                OpType::MatMul => {
                    self.execute_matmul_cpu(input, output, weights, &operation)?;
                }
                OpType::Activation => {
                    self.execute_activation_cpu(input, output, &operation)?;
                }
                OpType::Relu => {
                    self.execute_relu_cpu(input, output)?;
                }
                OpType::Sigmoid => {
                    self.execute_sigmoid_cpu(input, output)?;
                }
                OpType::Tanh => {
                    self.execute_tanh_cpu(input, output)?;
                }
                OpType::Softmax => {
                    self.execute_softmax_cpu(input, output)?;
                }
                OpType::MaxPool => {
                    self.execute_maxpool_cpu(input, output, &operation)?;
                }
                OpType::AvgPool => {
                    self.execute_avgpool_cpu(input, output, &operation)?;
                }
                OpType::BatchNorm => {
                    self.execute_batchnorm_cpu(input, output, weights, &operation)?;
                }
                OpType::LayerNorm => {
                    self.execute_layernorm_cpu(input, output, weights, &operation)?;
                }
                _ => {
                    // Fallback to scalar implementation
                    self.execute_scalar_fallback(input, output, weights, &operation)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// SIMD-optimized addition
    fn execute_add_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        if self.simd_capabilities.avx2 {
            self.execute_add_avx2(input, output, weights)?;
        } else if self.simd_capabilities.sse41 {
            self.execute_add_sse41(input, output, weights)?;
        } else {
            self.execute_add_scalar(input, output, weights)?;
        }
        Ok(())
    }
    
    /// SIMD-optimized multiplication
    fn execute_multiply_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        if self.simd_capabilities.avx2 {
            self.execute_multiply_avx2(input, output, weights)?;
        } else if self.simd_capabilities.sse41 {
            self.execute_multiply_sse41(input, output, weights)?;
        } else {
            self.execute_multiply_scalar(input, output, weights)?;
        }
        Ok(())
    }
    
    /// SIMD-optimized ReLU activation
    fn execute_relu_cpu(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        if self.simd_capabilities.avx2 {
            self.execute_relu_avx2(input, output)?;
        } else if self.simd_capabilities.sse41 {
            self.execute_relu_sse41(input, output)?;
        } else {
            self.execute_relu_scalar(input, output)?;
        }
        Ok(())
    }
    
    /// AVX2-optimized addition
    fn execute_add_avx2(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len()).min(weights.len());
            let simd_len = len - (len % 8); // Process 8 elements at a time
            
            unsafe {
                for i in (0..simd_len).step_by(8) {
                    let input_vec = _mm256_loadu_ps(&input[i]);
                    let weight_vec = _mm256_loadu_ps(&weights[i]);
                    let result_vec = _mm256_add_ps(input_vec, weight_vec);
                    _mm256_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i] + weights[i];
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_add_scalar(input, output, weights)?;
        }
        
        Ok(())
    }
    
    /// AVX2-optimized multiplication
    fn execute_multiply_avx2(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len()).min(weights.len());
            let simd_len = len - (len % 8); // Process 8 elements at a time
            
            unsafe {
                for i in (0..simd_len).step_by(8) {
                    let input_vec = _mm256_loadu_ps(&input[i]);
                    let weight_vec = _mm256_loadu_ps(&weights[i]);
                    let result_vec = _mm256_mul_ps(input_vec, weight_vec);
                    _mm256_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i] * weights[i];
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_multiply_scalar(input, output, weights)?;
        }
        
        Ok(())
    }
    
    /// AVX2-optimized ReLU
    fn execute_relu_avx2(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len());
            let simd_len = len - (len % 8); // Process 8 elements at a time
            let zero_vec = unsafe { _mm256_setzero_ps() };
            
            unsafe {
                for i in (0..simd_len).step_by(8) {
                    let input_vec = _mm256_loadu_ps(&input[i]);
                    let result_vec = _mm256_max_ps(input_vec, zero_vec);
                    _mm256_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i].max(0.0);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_relu_scalar(input, output)?;
        }
        
        Ok(())
    }
    
    /// SSE4.1-optimized addition
    fn execute_add_sse41(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len()).min(weights.len());
            let simd_len = len - (len % 4); // Process 4 elements at a time
            
            unsafe {
                for i in (0..simd_len).step_by(4) {
                    let input_vec = _mm_loadu_ps(&input[i]);
                    let weight_vec = _mm_loadu_ps(&weights[i]);
                    let result_vec = _mm_add_ps(input_vec, weight_vec);
                    _mm_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i] + weights[i];
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_add_scalar(input, output, weights)?;
        }
        
        Ok(())
    }
    
    /// SSE4.1-optimized multiplication
    fn execute_multiply_sse41(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len()).min(weights.len());
            let simd_len = len - (len % 4); // Process 4 elements at a time
            
            unsafe {
                for i in (0..simd_len).step_by(4) {
                    let input_vec = _mm_loadu_ps(&input[i]);
                    let weight_vec = _mm_loadu_ps(&weights[i]);
                    let result_vec = _mm_mul_ps(input_vec, weight_vec);
                    _mm_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i] * weights[i];
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_multiply_scalar(input, output, weights)?;
        }
        
        Ok(())
    }
    
    /// SSE4.1-optimized ReLU
    fn execute_relu_sse41(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            let len = input.len().min(output.len());
            let simd_len = len - (len % 4); // Process 4 elements at a time
            let zero_vec = unsafe { _mm_setzero_ps() };
            
            unsafe {
                for i in (0..simd_len).step_by(4) {
                    let input_vec = _mm_loadu_ps(&input[i]);
                    let result_vec = _mm_max_ps(input_vec, zero_vec);
                    _mm_storeu_ps(&mut output[i], result_vec);
                }
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i].max(0.0);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.execute_relu_scalar(input, output)?;
        }
        
        Ok(())
    }
    
    /// Scalar fallback implementations
    fn execute_add_scalar(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len()).min(weights.len());
        for i in 0..len {
            output[i] = input[i] + weights[i];
        }
        Ok(())
    }
    
    fn execute_multiply_scalar(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len()).min(weights.len());
        for i in 0..len {
            output[i] = input[i] * weights[i];
        }
        Ok(())
    }
    
    fn execute_relu_scalar(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i].max(0.0);
        }
        Ok(())
    }
    
    /// Optimized convolution implementation
    fn execute_convolution_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        let kernel_size = operation.attributes.get("kernel_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(3);
        
        // Use parallel processing for large convolutions
        if input.len() > 1000 && self.thread_pool.is_some() {
            self.execute_convolution_parallel(input, output, weights, kernel_size)?;
        } else {
            self.execute_convolution_sequential(input, output, weights, kernel_size)?;
        }
        
        Ok(())
    }
    
    /// Sequential convolution
    fn execute_convolution_sequential(&self, input: &[f32], output: &mut [f32], weights: &[f32], kernel_size: usize) -> CoelanoxResult<()> {
        for i in 0..input.len().saturating_sub(kernel_size - 1) {
            let mut sum = 0.0;
            for j in 0..kernel_size {
                if i + j < input.len() && j < weights.len() {
                    sum += input[i + j] * weights[j];
                }
            }
            if i < output.len() {
                output[i] = sum;
            }
        }
        Ok(())
    }
    
    /// Parallel convolution using thread pool
    fn execute_convolution_parallel(&self, input: &[f32], output: &mut [f32], weights: &[f32], kernel_size: usize) -> CoelanoxResult<()> {
        // For now, use sequential execution to avoid thread safety issues with raw pointers
        // TODO: Implement proper thread-safe parallel execution
        self.execute_convolution_sequential(input, output, weights, kernel_size)
    }
    
    /// Optimized matrix multiplication
    fn execute_matmul_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        // Get matrix dimensions
        let input_rows = operation.attributes.get("input_rows")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(1);
        
        let input_cols = operation.attributes.get("input_cols")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(input.len());
        
        let weight_rows = operation.attributes.get("weight_rows")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(weights.len() / input_cols.max(1));
        
        let weight_cols = operation.attributes.get("weight_cols")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(weights.len() / weight_rows.max(1));
        
        // Use parallel processing for large matrices
        if input_rows * weight_cols > 1000 && self.thread_pool.is_some() {
            self.execute_matmul_parallel(input, output, weights, input_rows, input_cols, weight_rows, weight_cols)?;
        } else {
            self.execute_matmul_sequential(input, output, weights, input_rows, input_cols, weight_rows, weight_cols)?;
        }
        
        Ok(())
    }
    
    /// Sequential matrix multiplication
    fn execute_matmul_sequential(&self, input: &[f32], output: &mut [f32], weights: &[f32], input_rows: usize, input_cols: usize, weight_rows: usize, weight_cols: usize) -> CoelanoxResult<()> {
        for i in 0..input_rows {
            for j in 0..weight_cols {
                let mut sum = 0.0;
                for k in 0..input_cols {
                    let input_idx = i * input_cols + k;
                    let weight_idx = k * weight_cols + j;
                    if input_idx < input.len() && weight_idx < weights.len() {
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
                let output_idx = i * weight_cols + j;
                if output_idx < output.len() {
                    output[output_idx] = sum;
                }
            }
        }
        Ok(())
    }
    
    /// Parallel matrix multiplication
    fn execute_matmul_parallel(&self, input: &[f32], output: &mut [f32], weights: &[f32], input_rows: usize, input_cols: usize, weight_rows: usize, weight_cols: usize) -> CoelanoxResult<()> {
        // For now, use sequential execution to avoid thread safety issues with raw pointers
        // TODO: Implement proper thread-safe parallel execution
        self.execute_matmul_sequential(input, output, weights, input_rows, input_cols, weight_rows, weight_cols)
    }
    
    /// Activation functions
    fn execute_activation_cpu(&self, input: &[f32], output: &mut [f32], operation: &Operation) -> CoelanoxResult<()> {
        match operation.attributes.get("activation_type") {
            Some(Attribute::String(activation_type)) => {
                match activation_type.as_str() {
                    "relu" => self.execute_relu_cpu(input, output)?,
                    "sigmoid" => self.execute_sigmoid_cpu(input, output)?,
                    "tanh" => self.execute_tanh_cpu(input, output)?,
                    "softmax" => self.execute_softmax_cpu(input, output)?,
                    _ => {
                        warn!("Unknown activation type: {}", activation_type);
                        self.execute_relu_cpu(input, output)?; // Default to ReLU
                    }
                }
            }
            _ => {
                self.execute_relu_cpu(input, output)?; // Default to ReLU
            }
        }
        Ok(())
    }
    
    /// Sigmoid activation
    fn execute_sigmoid_cpu(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
        Ok(())
    }
    
    /// Tanh activation
    fn execute_tanh_cpu(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i].tanh();
        }
        Ok(())
    }
    
    /// Softmax activation
    fn execute_softmax_cpu(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        if input.is_empty() || output.is_empty() {
            return Ok(());
        }
        
        // Find max for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials and sum
        let mut sum = 0.0;
        for (i, &val) in input.iter().enumerate() {
            if i < output.len() {
                let exp_val = (val - max_val).exp();
                output[i] = exp_val;
                sum += exp_val;
            }
        }
        
        // Normalize
        if sum > 0.0 {
            for i in 0..output.len().min(input.len()) {
                output[i] /= sum;
            }
        }
        
        Ok(())
    }
    
    /// Pooling operations
    fn execute_maxpool_cpu(&self, input: &[f32], output: &mut [f32], operation: &Operation) -> CoelanoxResult<()> {
        let pool_size = operation.attributes.get("pool_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(2);
        
        for i in 0..input.len() / pool_size {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..pool_size {
                let idx = i * pool_size + j;
                if idx < input.len() {
                    max_val = max_val.max(input[idx]);
                }
            }
            if i < output.len() {
                output[i] = max_val;
            }
        }
        Ok(())
    }
    
    fn execute_avgpool_cpu(&self, input: &[f32], output: &mut [f32], operation: &Operation) -> CoelanoxResult<()> {
        let pool_size = operation.attributes.get("pool_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(2);
        
        for i in 0..input.len() / pool_size {
            let mut sum = 0.0;
            let mut count = 0;
            for j in 0..pool_size {
                let idx = i * pool_size + j;
                if idx < input.len() {
                    sum += input[idx];
                    count += 1;
                }
            }
            if i < output.len() && count > 0 {
                output[i] = sum / count as f32;
            }
        }
        Ok(())
    }
    
    /// Normalization operations
    fn execute_batchnorm_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        let epsilon = operation.attributes.get("epsilon")
            .and_then(|attr| match attr {
                Attribute::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);
        
        if weights.len() >= 4 {
            let gamma = weights[0];
            let beta = weights[1];
            let mean = weights[2];
            let var = weights[3];
            
            for (i, &val) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = gamma * (val - mean) / (var + epsilon).sqrt() + beta;
                }
            }
        } else {
            // Fallback: passthrough
            for (i, &val) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = val;
                }
            }
        }
        Ok(())
    }
    
    fn execute_layernorm_cpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        let epsilon = operation.attributes.get("epsilon")
            .and_then(|attr| match attr {
                Attribute::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);
        
        // Compute mean and variance
        let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let var: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        
        if weights.len() >= 2 {
            let gamma = weights[0];
            let beta = weights[1];
            
            for (i, &val) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = gamma * (val - mean) / (var + epsilon).sqrt() + beta;
                }
            }
        } else {
            // Fallback: normalize without learnable parameters
            for (i, &val) in input.iter().enumerate() {
                if i < output.len() {
                    output[i] = (val - mean) / (var + epsilon).sqrt();
                }
            }
        }
        Ok(())
    }
    
    /// Scalar fallback for unsupported operations
    fn execute_scalar_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        warn!("Using scalar fallback for operation: {:?}", operation.op_type);
        
        // Simple passthrough for unsupported operations
        for (i, &val) in input.iter().enumerate() {
            if i < output.len() {
                output[i] = val;
            }
        }
        Ok(())
    }
    
    /// Get CPU backend information
    pub fn get_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("backend_type".to_string(), "CPU".to_string());
        info.insert("num_cores".to_string(), self.num_cores.to_string());
        info.insert("simd_capabilities".to_string(), self.simd_capabilities.best_simd().to_string());
        info.insert("thread_pool_enabled".to_string(), self.thread_pool.is_some().to_string());
        info
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
