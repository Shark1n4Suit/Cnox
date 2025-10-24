//! COELANOX runtime for executing containers
//! 
//! This module provides the runtime functionality for executing COELANOX containers.
//! The runtime loads binary containers and executes the optimized machine code.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::CoelanoxContainer;
use coelanox_core::ir::{Operation, OpType, Attribute};
use std::collections::HashMap;

/// COELANOX runtime for executing containers
pub struct CoelanoxRuntime {
    /// Runtime name
    name: String,
    /// Runtime version
    version: String,
    /// Loaded containers cache
    containers: HashMap<String, CoelanoxContainer>,
    /// Execution statistics
    stats: RuntimeStats,
    /// Executable memory blocks for cleanup
    executable_memory: Vec<ExecutableMemoryBlock>,
}

/// Executable memory block for proper cleanup
struct ExecutableMemoryBlock {
    ptr: *mut u8,
    size: usize,
}

/// Runtime execution statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Total executions
    pub total_executions: u64,
    /// Total execution time (microseconds)
    pub total_execution_time: u64,
    /// Average execution time (microseconds)
    pub avg_execution_time: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
}

impl RuntimeStats {
    pub fn new() -> Self {
        Self {
            total_executions: 0,
            total_execution_time: 0,
            avg_execution_time: 0,
            peak_memory_usage: 0,
        }
    }
    
    pub fn update_execution(&mut self, execution_time: u64, memory_usage: usize) {
        self.total_executions += 1;
        self.total_execution_time += execution_time;
        self.avg_execution_time = self.total_execution_time / self.total_executions;
        
        if memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = memory_usage;
        }
    }
}

impl CoelanoxRuntime {
    /// Create a new COELANOX runtime
    pub fn new() -> Self {
        Self {
            name: "coelanox_runtime".to_string(),
            version: "1.0.0".to_string(),
            containers: HashMap::new(),
            stats: RuntimeStats::new(),
            executable_memory: Vec::new(),
        }
    }
    
    /// Load a COELANOX container from file
    pub fn load_container(&mut self, path: &str) -> CoelanoxResult<()> {
        let container = CoelanoxContainer::read_binary_file(std::path::Path::new(path))?;
        
        // Validate container
        container.validate()?;
        
        // Store in cache
        self.containers.insert(path.to_string(), container);
        
        Ok(())
    }
    
    /// Load a COELANOX container from binary data
    pub fn load_container_from_data(&mut self, name: &str, data: &[u8]) -> CoelanoxResult<()> {
        let container = CoelanoxContainer::from_binary(data)?;
        
        // Validate container
        container.validate()?;
        
        // Store in cache
        self.containers.insert(name.to_string(), container);
        
        Ok(())
    }
    
    /// Execute a COELANOX container by name
    pub fn execute(&mut self, container_name: &str, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        let start_time = std::time::Instant::now();
        
        // Get container from cache and clone it to avoid borrowing conflicts
        let container = self.containers.get(container_name)
            .ok_or_else(|| CoelanoxError::Runtime("Container not loaded".to_string()))?
            .clone();
        
        // Validate input size
        let expected_input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        if input_data.len() != expected_input_size {
            return Err(CoelanoxError::Runtime(format!(
                "Input size mismatch: expected {}, got {}",
                expected_input_size,
                input_data.len()
            )));
        }
        
        // Execute the real machine code (preferred) or fallback to Rust implementation
        let output = match self.execute_machine_code(&container, input_data) {
            Ok(result) => {
                println!("✅ Executed using real machine code");
                result
            }
            Err(e) => {
                println!("⚠️  Machine code execution failed: {:?}, falling back to Rust implementation", e);
                self.execute_machine_code_with_weights(&container, input_data)?
            }
        };
        
        // Update statistics
        let execution_time = start_time.elapsed().as_micros() as u64;
        let memory_usage = std::mem::size_of_val(input_data) + std::mem::size_of_val(&output);
        self.stats.update_execution(execution_time, memory_usage);
        
        Ok(output)
    }
    
    /// Execute machine code with weights
    fn execute_machine_code_with_weights(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Load weights from container
        let weights = container.decompress_weights()?;
        
        // Calculate output size
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        let mut output = vec![0.0f32; output_size];
        
        // Execute operations with real weights
        for operation_json in &container.manifest.operations {
            // Convert JSON to Operation struct
            let operation: Operation = serde_json::from_value(operation_json.clone())
                .map_err(|e| CoelanoxError::format(&format!("Failed to deserialize operation: {}", e)))?;
            
            let op_output = self.execute_operation_with_weights(&operation, input_data, &weights)?;
            
            // Copy operation output to final output
            for (i, &val) in op_output.iter().enumerate() {
                if i < output.len() {
                    output[i] = val;
                }
            }
        }
        
        Ok(output)
    }
    
    /// Execute a single operation with weights
    fn execute_operation_with_weights(&self, operation: &Operation, input_data: &[f32], weights: &[f32]) -> CoelanoxResult<Vec<f32>> {
        match operation.op_type {
            OpType::Add => {
                // Real addition with weights
                let mut output = Vec::new();
                for (i, &val) in input_data.iter().enumerate() {
                    if i < weights.len() {
                        output.push(val + weights[i]);
                    } else {
                        output.push(val);
                    }
                }
                Ok(output)
            }
            OpType::Multiply => {
                // Real multiplication with weights
                let mut output = Vec::new();
                for (i, &val) in input_data.iter().enumerate() {
                    if i < weights.len() {
                        output.push(val * weights[i]);
                    } else {
                        output.push(val);
                    }
                }
                Ok(output)
            }
            OpType::Convolution => {
                // Simple convolution implementation
                self.execute_convolution_with_weights(operation, input_data, weights)
            }
            OpType::MatMul => {
                // Simple matrix multiplication implementation
                self.execute_matmul_with_weights(operation, input_data, weights)
            }
            OpType::Activation => {
                // ReLU activation
                let output: Vec<f32> = input_data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
                Ok(output)
            }
            _ => {
                // Fallback: pass through input
                Ok(input_data.to_vec())
            }
        }
    }
    
    /// Execute convolution with weights
    fn execute_convolution_with_weights(&self, operation: &Operation, input_data: &[f32], weights: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Simple convolution implementation
        let mut output = Vec::new();
        
        // Get kernel size from operation attributes
        let kernel_size = operation.attributes.get("kernel_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(3);
        
        // Simple 1D convolution
        for i in 0..input_data.len().saturating_sub(kernel_size - 1) {
            let mut sum = 0.0;
            for j in 0..kernel_size {
                if i + j < input_data.len() && j < weights.len() {
                    sum += input_data[i + j] * weights[j];
                }
            }
            output.push(sum);
        }
        
        Ok(output)
    }
    
    /// Execute matrix multiplication with weights
    fn execute_matmul_with_weights(&self, operation: &Operation, input_data: &[f32], weights: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Simple matrix multiplication implementation
        let mut output = Vec::new();
        
        // Get matrix dimensions from operation attributes
        let dims = operation.attributes.get("dimensions")
            .and_then(|attr| match attr {
                Attribute::IntArray(dims) => Some(dims.iter().map(|&x| x as i32).collect()),
                _ => None,
            })
            .unwrap_or(vec![input_data.len() as i32, 1]);
        
        if dims.len() >= 2 {
            let rows = dims[0] as usize;
            let cols = dims[1] as usize;
            
            // Simple matrix multiplication: input * weights
            for i in 0..rows {
                let mut sum = 0.0;
                for j in 0..cols {
                    let input_idx = i * cols + j;
                    let weight_idx = i * cols + j;
                    if input_idx < input_data.len() && weight_idx < weights.len() {
                        sum += input_data[input_idx] * weights[weight_idx];
                    }
                }
                output.push(sum);
            }
        } else {
            // Fallback: element-wise multiplication
            for (i, &val) in input_data.iter().enumerate() {
                if i < weights.len() {
                    output.push(val * weights[i]);
                } else {
                    output.push(val);
                }
            }
        }
        
        Ok(output)
    }
    
    /// Execute machine code from container (legacy method)
    fn execute_machine_code(&mut self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Calculate output size
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        // Create executable memory and run real machine code
        let mut output = vec![0.0f32; output_size];
        
        // Generate real executable function
        let executable_func = self.create_executable_function(container, input_data)?;
        
        // Execute the real machine code
        executable_func(input_data, &mut output);
        
        Ok(output)
    }
    
    /// Create executable function from machine code
    fn create_executable_function(&mut self, container: &CoelanoxContainer, _input_data: &[f32]) -> CoelanoxResult<fn(&[f32], &mut [f32])> {
        // Try to allocate executable memory
        let code_size = container.optimized_code.len();
        let page_size = 4096; // Standard page size
        let aligned_size = ((code_size + page_size - 1) / page_size) * page_size;
        
        match self.allocate_executable_memory(aligned_size) {
            Ok((ptr, size)) => {
                println!("✅ Successfully allocated executable memory: {} bytes", size);
                
                // Copy machine code to executable memory
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        container.optimized_code.as_ptr(),
                        ptr,
                        code_size
                    );
                }
                
                // Create function pointer
                let func_ptr = ptr as usize;
                let func: fn(&[f32], &mut [f32]) = unsafe { std::mem::transmute(func_ptr) };
                
                // Store the memory block for cleanup
                self.executable_memory.push(ExecutableMemoryBlock { ptr, size });
                
                println!("✅ Created function pointer: 0x{:x}", func_ptr);
                Ok(func)
            }
            Err(e) => {
                println!("❌ Failed to allocate executable memory: {:?}", e);
                // Return a static function that doesn't capture anything
                // This will be handled by the fallback in execute_machine_code_with_weights
                Err(e)
            }
        }
    }
    
    /// Allocate executable memory with proper permissions
    fn allocate_executable_memory(&self, size: usize) -> CoelanoxResult<(*mut u8, usize)> {
        #[cfg(target_os = "linux")]
        {
            use libc::{mmap, MAP_PRIVATE, MAP_ANONYMOUS, PROT_READ, PROT_WRITE, PROT_EXEC};
            
            // Use mmap to allocate executable memory directly
            let result = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            
            if result == libc::MAP_FAILED {
                return Err(CoelanoxError::Runtime("mmap failed for executable memory".to_string()));
            }
            
            Ok((result as *mut u8, size))
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            #[cfg(target_os = "windows")]
            {
                use windows_sys::Win32::System::Memory::{
                    VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE
                };
                
                let result = unsafe {
                    VirtualAlloc(
                        std::ptr::null_mut(),
                        size,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_EXECUTE_READWRITE,
                    )
                };
                
                if result.is_null() {
                    return Err(CoelanoxError::Runtime("VirtualAlloc failed".to_string()));
                }
                
                Ok((result as *mut u8, size))
            }
            
            #[cfg(not(target_os = "windows"))]
            {
                // Generic fallback - allocate regular memory
                let mut memory = vec![0u8; size];
                let ptr = memory.as_mut_ptr();
                std::mem::forget(memory); // Prevent deallocation
                Ok((ptr, size))
            }
        }
    }
    
    /// Execute with specific backend
    pub fn execute_with_backend(
        &mut self,
        container_name: &str,
        input_data: &[f32],
        backend: &str,
    ) -> CoelanoxResult<Vec<f32>> {
        // For now, backend selection is not implemented
        // In the future, this would select different execution engines
        match backend {
            "cpu" => self.execute(container_name, input_data),
            "gpu" => self.execute_gpu(container_name, input_data),
            "edge" => self.execute_edge(container_name, input_data),
            _ => Err(CoelanoxError::Runtime(format!("Unknown backend: {}", backend))),
        }
    }
    
    /// Execute on GPU (placeholder)
    fn execute_gpu(&mut self, container_name: &str, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // TODO: Implement GPU execution
        // This would use CUDA/OpenCL/other GPU APIs
        self.execute(container_name, input_data)
    }
    
    /// Execute on Edge device (placeholder)
    fn execute_edge(&mut self, container_name: &str, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // TODO: Implement edge execution
        // This would use ARM NEON, DSP instructions, etc.
        self.execute(container_name, input_data)
    }
    
    /// Get container information
    pub fn get_container_info(&self, container_name: &str) -> CoelanoxResult<&CoelanoxContainer> {
        self.containers.get(container_name)
            .ok_or_else(|| CoelanoxError::Runtime("Container not found".to_string()))
    }
    
    /// List loaded containers
    pub fn list_containers(&self) -> Vec<&String> {
        self.containers.keys().collect()
    }
    
    /// Get runtime statistics
    pub fn get_stats(&self) -> &RuntimeStats {
        &self.stats
    }
    
    /// Get runtime name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get runtime version
    pub fn version(&self) -> &str {
        &self.version
    }
    
    /// Execute operations using fallback implementation
    fn execute_operations_fallback(&self, input: &[f32], output: &mut [f32], container: &CoelanoxContainer) {
        // Load weights from container
        let weights = match container.decompress_weights() {
            Ok(w) => w,
            Err(_) => {
                // If weights fail, use simple passthrough
                for (i, &val) in input.iter().enumerate() {
                    if i < output.len() {
                        output[i] = val;
                    }
                }
                return;
            }
        };
        
        // Process each operation in the container
        for operation_json in &container.manifest.operations {
            // Convert JSON to Operation struct
            let operation: Operation = match serde_json::from_value(operation_json.clone()) {
                Ok(op) => op,
                Err(_) => continue, // Skip invalid operations
            };
            
            match operation.op_type {
                OpType::Add => {
                    // Element-wise addition with weights
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() && i < weights.len() {
                            output[i] = val + weights[i];
                        } else if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
                OpType::Subtract => {
                    // Element-wise subtraction with weights
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() && i < weights.len() {
                            output[i] = val - weights[i];
                        } else if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
                OpType::Multiply => {
                    // Element-wise multiplication with weights
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() && i < weights.len() {
                            output[i] = val * weights[i];
                        } else if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
                OpType::Divide => {
                    // Element-wise division with weights
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() && i < weights.len() {
                            output[i] = if weights[i] != 0.0 { val / weights[i] } else { val };
                        } else if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
                OpType::Relu => {
                    // ReLU activation
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = if val > 0.0 { val } else { 0.0 };
                        }
                    }
                }
                OpType::Sigmoid => {
                    // Sigmoid activation
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = 1.0 / (1.0 + (-val).exp());
                        }
                    }
                }
                OpType::Tanh => {
                    // Tanh activation
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = val.tanh();
                        }
                    }
                }
                OpType::Softmax => {
                    // Softmax activation
                    self.execute_softmax_fallback(input, output);
                }
                OpType::Gelu => {
                    // GELU activation
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = 0.5 * val * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh());
                        }
                    }
                }
                OpType::Swish => {
                    // Swish activation (x * sigmoid(x))
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            let sigmoid = 1.0 / (1.0 + (-val).exp());
                            output[i] = val * sigmoid;
                        }
                    }
                }
                OpType::Convolution => {
                    // Simple convolution implementation
                    self.execute_convolution_fallback(input, output, &weights, &operation);
                }
                OpType::MatMul => {
                    // Matrix multiplication implementation
                    self.execute_matmul_fallback(input, output, &weights, &operation);
                }
                OpType::Gemm => {
                    // Generalized matrix multiplication (same as MatMul for now)
                    self.execute_matmul_fallback(input, output, &weights, &operation);
                }
                OpType::MaxPool => {
                    // Max pooling implementation
                    self.execute_maxpool_fallback(input, output, &operation);
                }
                OpType::AvgPool => {
                    // Average pooling implementation
                    self.execute_avgpool_fallback(input, output, &operation);
                }
                OpType::GlobalMaxPool => {
                    // Global max pooling
                    self.execute_global_maxpool_fallback(input, output);
                }
                OpType::GlobalAvgPool => {
                    // Global average pooling
                    self.execute_global_avgpool_fallback(input, output);
                }
                OpType::BatchNorm => {
                    // Batch normalization
                    self.execute_batchnorm_fallback(input, output, &weights, &operation);
                }
                OpType::LayerNorm => {
                    // Layer normalization
                    self.execute_layernorm_fallback(input, output, &weights, &operation);
                }
                OpType::ReduceSum => {
                    // Sum reduction
                    self.execute_reduce_sum_fallback(input, output, &operation);
                }
                OpType::ReduceMean => {
                    // Mean reduction
                    self.execute_reduce_mean_fallback(input, output, &operation);
                }
                OpType::ReduceMax => {
                    // Max reduction
                    self.execute_reduce_max_fallback(input, output, &operation);
                }
                OpType::ReduceMin => {
                    // Min reduction
                    self.execute_reduce_min_fallback(input, output, &operation);
                }
                _ => {
                    // Default: passthrough for unsupported operations
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
            }
        }
    }
    
    /// Execute convolution using fallback implementation
    fn execute_convolution_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) {
        let kernel_size = operation.attributes.get("kernel_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(3);
        
        // Simple 1D convolution
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
    }
    
    /// Execute matrix multiplication using fallback implementation
    fn execute_matmul_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) {
        // Get matrix dimensions from operation attributes or infer from sizes
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
        
        // Validate dimensions
        if input_cols != weight_rows || input.len() < input_rows * input_cols || weights.len() < weight_rows * weight_cols {
            // Fallback: element-wise multiplication if dimensions don't match
            for (i, &val) in input.iter().enumerate() {
                if i < output.len() && i < weights.len() {
                    output[i] = val * weights[i];
                } else if i < output.len() {
                    output[i] = val;
                }
            }
            return;
        }
        
        // Perform matrix multiplication: input @ weights
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
    }
    
    /// Execute max pooling using fallback implementation
    fn execute_maxpool_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let pool_size = operation.attributes.get("pool_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(2);
        
        // Simple 1D max pooling
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
    }
    
    /// Execute average pooling using fallback implementation
    fn execute_avgpool_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let pool_size = operation.attributes.get("pool_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(2);
        
        // Simple 1D average pooling
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
    }
    
    /// Execute softmax using fallback implementation
    fn execute_softmax_fallback(&self, input: &[f32], output: &mut [f32]) {
        if input.is_empty() || output.is_empty() {
            return;
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
    }
    
    /// Execute global max pooling using fallback implementation
    fn execute_global_maxpool_fallback(&self, input: &[f32], output: &mut [f32]) {
        if input.is_empty() || output.is_empty() {
            return;
        }
        
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for i in 0..output.len() {
            output[i] = max_val;
        }
    }
    
    /// Execute global average pooling using fallback implementation
    fn execute_global_avgpool_fallback(&self, input: &[f32], output: &mut [f32]) {
        if input.is_empty() || output.is_empty() {
            return;
        }
        
        let sum: f32 = input.iter().sum();
        let avg_val = sum / input.len() as f32;
        for i in 0..output.len() {
            output[i] = avg_val;
        }
    }
    
    /// Execute batch normalization using fallback implementation
    fn execute_batchnorm_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) {
        let epsilon = operation.attributes.get("epsilon")
            .and_then(|attr| match attr {
                Attribute::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);
        
        // Simple batch norm: (x - mean) / sqrt(var + epsilon) * gamma + beta
        // For now, assume weights contain [gamma, beta, mean, var] in that order
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
    }
    
    /// Execute layer normalization using fallback implementation
    fn execute_layernorm_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) {
        let epsilon = operation.attributes.get("epsilon")
            .and_then(|attr| match attr {
                Attribute::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);
        
        // Compute mean and variance
        let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        let var: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
        
        // Layer norm: (x - mean) / sqrt(var + epsilon) * gamma + beta
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
    }
    
    /// Execute sum reduction using fallback implementation
    fn execute_reduce_sum_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let axis = operation.attributes.get("axis")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            });
        
        match axis {
            Some(0) => {
                // Sum along first dimension
                let sum: f32 = input.iter().sum();
                for i in 0..output.len() {
                    output[i] = sum;
                }
            }
            _ => {
                // Default: sum all elements
                let sum: f32 = input.iter().sum();
                for i in 0..output.len() {
                    output[i] = sum;
                }
            }
        }
    }
    
    /// Execute mean reduction using fallback implementation
    fn execute_reduce_mean_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let axis = operation.attributes.get("axis")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            });
        
        match axis {
            Some(0) => {
                // Mean along first dimension
                let mean = input.iter().sum::<f32>() / input.len() as f32;
                for i in 0..output.len() {
                    output[i] = mean;
                }
            }
            _ => {
                // Default: mean of all elements
                let mean = input.iter().sum::<f32>() / input.len() as f32;
                for i in 0..output.len() {
                    output[i] = mean;
                }
            }
        }
    }
    
    /// Execute max reduction using fallback implementation
    fn execute_reduce_max_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let axis = operation.attributes.get("axis")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            });
        
        match axis {
            Some(0) => {
                // Max along first dimension
                let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                for i in 0..output.len() {
                    output[i] = max_val;
                }
            }
            _ => {
                // Default: max of all elements
                let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                for i in 0..output.len() {
                    output[i] = max_val;
                }
            }
        }
    }
    
    /// Execute min reduction using fallback implementation
    fn execute_reduce_min_fallback(&self, input: &[f32], output: &mut [f32], operation: &Operation) {
        let axis = operation.attributes.get("axis")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            });
        
        match axis {
            Some(0) => {
                // Min along first dimension
                let min_val = input.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                for i in 0..output.len() {
                    output[i] = min_val;
                }
            }
            _ => {
                // Default: min of all elements
                let min_val = input.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                for i in 0..output.len() {
                    output[i] = min_val;
                }
            }
        }
    }
    
    /// Cleanup executable memory
    pub fn cleanup(&mut self) {
        #[cfg(target_os = "linux")]
        {
            use libc::munmap;
            
            for block in &self.executable_memory {
                unsafe {
                    munmap(block.ptr as *mut std::ffi::c_void, block.size);
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            use windows_sys::Win32::System::Memory::VirtualFree;
            use windows_sys::Win32::System::Memory::MEM_RELEASE;
            
            for block in &self.executable_memory {
                unsafe {
                    VirtualFree(block.ptr as *mut std::ffi::c_void, 0, MEM_RELEASE);
                }
            }
        }
        
        self.executable_memory.clear();
    }
    
    /// Benchmark a container
    pub fn benchmark(&mut self, container_name: &str, iterations: usize) -> CoelanoxResult<BenchmarkResult> {
        let container = self.get_container_info(container_name)?;
        
        // Generate dummy input data
        let input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        let input_data = vec![1.0f32; input_size];
        
        let mut total_time = 0u64;
        let mut min_time = u64::MAX;
        let mut max_time = 0u64;
        
        // Warm up
        for _ in 0..10 {
            self.execute(container_name, &input_data)?;
        }
        
        // Benchmark
        for _ in 0..iterations {
            let start_time = std::time::Instant::now();
            self.execute(container_name, &input_data)?;
            let execution_time = start_time.elapsed().as_micros() as u64;
            
            total_time += execution_time;
            min_time = min_time.min(execution_time);
            max_time = max_time.max(execution_time);
        }
        
        let avg_time = total_time / iterations as u64;
        
        Ok(BenchmarkResult {
            container_name: container_name.to_string(),
            iterations,
            avg_time,
            min_time,
            max_time,
            total_time,
        })
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub container_name: String,
    pub iterations: usize,
    pub avg_time: u64,
    pub min_time: u64,
    pub max_time: u64,
    pub total_time: u64,
}

impl Drop for CoelanoxRuntime {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl Default for CoelanoxRuntime {
    fn default() -> Self {
        Self::new()
    }
}
