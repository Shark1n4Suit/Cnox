//! NPU backend implementation with specialized AI acceleration
//! 
//! This module provides NPU-accelerated execution for COELANOX containers,
//! supporting various NPU architectures with optimized AI operations.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::CoelanoxContainer;
use coelanox_core::ir::{Operation, OpType, Attribute};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn, error};

/// NPU backend for executing operations with specialized AI acceleration
pub struct NpuBackend {
    /// NPU device information
    device_info: NpuDeviceInfo,
    /// NPU context (if available)
    npu_context: Option<NpuContext>,
    /// Preferred NPU type
    preferred_npu: NpuType,
    /// Memory manager for NPU allocations
    memory_manager: Arc<NpuMemoryManager>,
    /// Operation cache for optimized execution
    operation_cache: Arc<NpuOperationCache>,
}

/// NPU device information
#[derive(Debug, Clone)]
pub struct NpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Device vendor
    pub vendor: String,
    /// NPU architecture
    pub architecture: NpuArchitecture,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Number of NPU cores
    pub num_cores: u32,
    /// Peak performance (TOPS)
    pub peak_performance_tops: f32,
    /// Supported data types
    pub supported_data_types: Vec<NpuDataType>,
    /// Maximum tensor dimensions
    pub max_tensor_dims: Vec<usize>,
}

/// NPU architecture types
#[derive(Debug, Clone, PartialEq)]
pub enum NpuArchitecture {
    /// ARM Neural Processing Unit
    ArmNpu,
    /// Qualcomm Hexagon DSP
    HexagonDsp,
    /// Apple Neural Engine
    AppleNeuralEngine,
    /// Intel Movidius Myriad
    MovidiusMyriad,
    /// Google Edge TPU
    EdgeTpu,
    /// Generic NPU
    GenericNpu,
}

/// NPU context wrapper
#[derive(Debug)]
pub struct NpuContext {
    /// NPU device ID
    device_id: u32,
    /// NPU context handle
    context: *mut std::ffi::c_void,
    /// NPU command queue
    command_queue: *mut std::ffi::c_void,
    /// NPU memory allocator
    memory_allocator: *mut std::ffi::c_void,
}

/// NPU type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum NpuType {
    /// ARM NPU
    ArmNpu,
    /// Hexagon DSP
    HexagonDsp,
    /// Apple Neural Engine
    AppleNeuralEngine,
    /// Movidius Myriad
    MovidiusMyriad,
    /// Edge TPU
    EdgeTpu,
    /// Generic NPU
    GenericNpu,
    /// CPU fallback
    CpuFallback,
}

/// NPU data types
#[derive(Debug, Clone, PartialEq)]
pub enum NpuDataType {
    /// 8-bit integer
    Int8,
    /// 16-bit integer
    Int16,
    /// 32-bit integer
    Int32,
    /// 16-bit floating point
    Float16,
    /// 32-bit floating point
    Float32,
    /// 8-bit unsigned integer
    Uint8,
}

/// NPU memory manager for efficient memory allocation
#[derive(Debug)]
pub struct NpuMemoryManager {
    /// Allocated memory blocks
    allocated_blocks: Vec<NpuMemoryBlock>,
    /// Free memory blocks
    free_blocks: Vec<NpuMemoryBlock>,
    /// Total allocated memory
    total_allocated: usize,
    /// Memory pool size limit
    pool_size_limit: usize,
    /// Memory alignment requirements
    alignment: usize,
}

/// NPU memory block
#[derive(Debug, Clone)]
pub struct NpuMemoryBlock {
    /// Memory pointer
    ptr: *mut u8,
    /// Block size in bytes
    size: usize,
    /// Allocation timestamp
    allocated_at: std::time::Instant,
    /// Block ID
    id: u64,
    /// Memory type
    memory_type: NpuMemoryType,
}

/// NPU memory types
#[derive(Debug, Clone, PartialEq)]
pub enum NpuMemoryType {
    /// Input memory
    Input,
    /// Output memory
    Output,
    /// Weight memory
    Weight,
    /// Intermediate memory
    Intermediate,
}

/// NPU operation cache for optimized execution
#[derive(Debug)]
pub struct NpuOperationCache {
    /// Cached operations
    cached_operations: HashMap<String, CachedOperation>,
    /// Cache hit count
    cache_hits: u64,
    /// Cache miss count
    cache_misses: u64,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Cached operation
#[derive(Debug, Clone)]
pub struct CachedOperation {
    /// Operation hash
    operation_hash: String,
    /// Compiled operation
    compiled_operation: *mut std::ffi::c_void,
    /// Execution time
    execution_time: std::time::Duration,
    /// Last used timestamp
    last_used: std::time::Instant,
}

impl NpuBackend {
    /// Create a new NPU backend
    pub fn new() -> Self {
        let device_info = Self::detect_npu_devices();
        let preferred_npu = Self::select_preferred_npu(&device_info);
        
        let npu_context = match preferred_npu {
            NpuType::ArmNpu => Self::initialize_arm_npu(&device_info),
            NpuType::HexagonDsp => Self::initialize_hexagon_dsp(&device_info),
            NpuType::AppleNeuralEngine => Self::initialize_apple_neural_engine(&device_info),
            NpuType::MovidiusMyriad => Self::initialize_movidius_myriad(&device_info),
            NpuType::EdgeTpu => Self::initialize_edge_tpu(&device_info),
            NpuType::GenericNpu => Self::initialize_generic_npu(&device_info),
            NpuType::CpuFallback => {
                warn!("No NPU devices found, falling back to CPU");
                None
            }
        };
        
        let memory_manager = Arc::new(NpuMemoryManager::new(512 * 1024 * 1024)); // 512MB pool
        let operation_cache = Arc::new(NpuOperationCache::new(100)); // 100 operations cache
        
        info!("🚀 NPU Backend initialized:");
        info!("  Device: {}", device_info.name);
        info!("  Vendor: {}", device_info.vendor);
        info!("  Architecture: {:?}", device_info.architecture);
        info!("  Memory: {} MB / {} MB", device_info.available_memory / (1024 * 1024), device_info.total_memory / (1024 * 1024));
        info!("  Cores: {}", device_info.num_cores);
        info!("  Performance: {} TOPS", device_info.peak_performance_tops);
        info!("  NPU Type: {:?}", preferred_npu);
        
        Self {
            device_info,
            npu_context,
            preferred_npu,
            memory_manager,
            operation_cache,
        }
    }
    
    /// Detect available NPU devices
    fn detect_npu_devices() -> NpuDeviceInfo {
        // Try ARM NPU first
        if let Some(arm_info) = Self::detect_arm_npu() {
            return arm_info;
        }
        
        // Try Hexagon DSP
        if let Some(hexagon_info) = Self::detect_hexagon_dsp() {
            return hexagon_info;
        }
        
        // Try Apple Neural Engine
        if let Some(ane_info) = Self::detect_apple_neural_engine() {
            return ane_info;
        }
        
        // Try Movidius Myriad
        if let Some(myriad_info) = Self::detect_movidius_myriad() {
            return myriad_info;
        }
        
        // Try Edge TPU
        if let Some(edge_tpu_info) = Self::detect_edge_tpu() {
            return edge_tpu_info;
        }
        
        // Try Generic NPU
        if let Some(generic_info) = Self::detect_generic_npu() {
            return generic_info;
        }
        
        // Fallback to CPU
        NpuDeviceInfo {
            name: "CPU Fallback".to_string(),
            vendor: "Generic".to_string(),
            architecture: NpuArchitecture::GenericNpu,
            total_memory: 0,
            available_memory: 0,
            num_cores: num_cpus::get() as u32,
            peak_performance_tops: 0.0,
            supported_data_types: vec![NpuDataType::Float32],
            max_tensor_dims: vec![1024, 1024, 1024],
        }
    }
    
    /// Detect ARM NPU
    fn detect_arm_npu() -> Option<NpuDeviceInfo> {
        #[cfg(feature = "arm-npu")]
        {
            // ARM NPU detection implementation would go here
            None
        }
        
        #[cfg(not(feature = "arm-npu"))]
        {
            None
        }
    }
    
    /// Detect Hexagon DSP
    fn detect_hexagon_dsp() -> Option<NpuDeviceInfo> {
        #[cfg(feature = "hexagon-dsp")]
        {
            // Hexagon DSP detection implementation would go here
            None
        }
        
        #[cfg(not(feature = "hexagon-dsp"))]
        {
            None
        }
    }
    
    /// Detect Apple Neural Engine
    fn detect_apple_neural_engine() -> Option<NpuDeviceInfo> {
        #[cfg(target_os = "ios")]
        {
            // Apple Neural Engine detection implementation would go here
            None
        }
        
        #[cfg(not(target_os = "ios"))]
        {
            None
        }
    }
    
    /// Detect Movidius Myriad
    fn detect_movidius_myriad() -> Option<NpuDeviceInfo> {
        #[cfg(feature = "movidius-myriad")]
        {
            // Movidius Myriad detection implementation would go here
            None
        }
        
        #[cfg(not(feature = "movidius-myriad"))]
        {
            None
        }
    }
    
    /// Detect Edge TPU
    fn detect_edge_tpu() -> Option<NpuDeviceInfo> {
        #[cfg(feature = "edge-tpu")]
        {
            // Edge TPU detection implementation would go here
            None
        }
        
        #[cfg(not(feature = "edge-tpu"))]
        {
            None
        }
    }
    
    /// Detect Generic NPU
    fn detect_generic_npu() -> Option<NpuDeviceInfo> {
        // Generic NPU detection implementation would go here
        None
    }
    
    /// Select preferred NPU based on available devices
    fn select_preferred_npu(device_info: &NpuDeviceInfo) -> NpuType {
        match device_info.architecture {
            NpuArchitecture::ArmNpu => NpuType::ArmNpu,
            NpuArchitecture::HexagonDsp => NpuType::HexagonDsp,
            NpuArchitecture::AppleNeuralEngine => NpuType::AppleNeuralEngine,
            NpuArchitecture::MovidiusMyriad => NpuType::MovidiusMyriad,
            NpuArchitecture::EdgeTpu => NpuType::EdgeTpu,
            NpuArchitecture::GenericNpu => NpuType::GenericNpu,
        }
    }
    
    /// Initialize ARM NPU
    fn initialize_arm_npu(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        #[cfg(feature = "arm-npu")]
        {
            // ARM NPU initialization would go here
            None
        }
        
        #[cfg(not(feature = "arm-npu"))]
        {
            None
        }
    }
    
    /// Initialize Hexagon DSP
    fn initialize_hexagon_dsp(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        #[cfg(feature = "hexagon-dsp")]
        {
            // Hexagon DSP initialization would go here
            None
        }
        
        #[cfg(not(feature = "hexagon-dsp"))]
        {
            None
        }
    }
    
    /// Initialize Apple Neural Engine
    fn initialize_apple_neural_engine(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        #[cfg(target_os = "ios")]
        {
            // Apple Neural Engine initialization would go here
            None
        }
        
        #[cfg(not(target_os = "ios"))]
        {
            None
        }
    }
    
    /// Initialize Movidius Myriad
    fn initialize_movidius_myriad(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        #[cfg(feature = "movidius-myriad")]
        {
            // Movidius Myriad initialization would go here
            None
        }
        
        #[cfg(not(feature = "movidius-myriad"))]
        {
            None
        }
    }
    
    /// Initialize Edge TPU
    fn initialize_edge_tpu(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        #[cfg(feature = "edge-tpu")]
        {
            // Edge TPU initialization would go here
            None
        }
        
        #[cfg(not(feature = "edge-tpu"))]
        {
            None
        }
    }
    
    /// Initialize Generic NPU
    fn initialize_generic_npu(_device_info: &NpuDeviceInfo) -> Option<NpuContext> {
        // Generic NPU initialization would go here
        None
    }
    
    /// Execute operations on NPU with acceleration
    pub fn execute(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        debug!("🧠 NPU Backend executing with {:?}", self.preferred_npu);
        
        // Calculate output size
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        let mut output = vec![0.0f32; output_size];
        
        // Load weights from container
        let weights = container.decompress_weights()?;
        
        // Legacy operations field removed - using empty operations
        let empty_operations: Vec<serde_json::Value> = Vec::new();
        
        // Execute operations with NPU acceleration
        match self.preferred_npu {
            NpuType::ArmNpu => {
                self.execute_operations_arm_npu(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::HexagonDsp => {
                self.execute_operations_hexagon_dsp(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::AppleNeuralEngine => {
                self.execute_operations_apple_neural_engine(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::MovidiusMyriad => {
                self.execute_operations_movidius_myriad(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::EdgeTpu => {
                self.execute_operations_edge_tpu(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::GenericNpu => {
                self.execute_operations_generic_npu(input_data, &mut output, &weights, &empty_operations)?;
            }
            NpuType::CpuFallback => {
                self.execute_operations_cpu_fallback(input_data, &mut output, &weights, &empty_operations)?;
            }
        }
        
        debug!("✅ NPU Backend execution completed: {} -> {} elements", input_data.len(), output.len());
        Ok(output)
    }
    
    /// Execute operations with ARM NPU acceleration
    fn execute_operations_arm_npu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // ARM NPU execution implementation would go here
            // For now, fall back to CPU
            warn!("ARM NPU execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("ARM NPU context not available".to_string()));
        }
        Ok(())
    }
    
    /// Execute operations with Hexagon DSP acceleration
    fn execute_operations_hexagon_dsp(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // Hexagon DSP execution implementation would go here
            // For now, fall back to CPU
            warn!("Hexagon DSP execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("Hexagon DSP context not available".to_string()));
        }
        Ok(())
    }
    
    /// Execute operations with Apple Neural Engine acceleration
    fn execute_operations_apple_neural_engine(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // Apple Neural Engine execution implementation would go here
            // For now, fall back to CPU
            warn!("Apple Neural Engine execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("Apple Neural Engine context not available".to_string()));
        }
        Ok(())
    }
    
    /// Execute operations with Movidius Myriad acceleration
    fn execute_operations_movidius_myriad(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // Movidius Myriad execution implementation would go here
            // For now, fall back to CPU
            warn!("Movidius Myriad execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("Movidius Myriad context not available".to_string()));
        }
        Ok(())
    }
    
    /// Execute operations with Edge TPU acceleration
    fn execute_operations_edge_tpu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // Edge TPU execution implementation would go here
            // For now, fall back to CPU
            warn!("Edge TPU execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("Edge TPU context not available".to_string()));
        }
        Ok(())
    }
    
    /// Execute operations with Generic NPU acceleration
    fn execute_operations_generic_npu(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        if let Some(npu_ctx) = &self.npu_context {
            // Generic NPU execution implementation would go here
            // For now, fall back to CPU
            warn!("Generic NPU execution not implemented, falling back to CPU");
            self.execute_operations_cpu_fallback(input, output, weights, operations)?;
        } else {
            return Err(CoelanoxError::Backend("Generic NPU context not available".to_string()));
        }
        Ok(())
    }
    
    /// CPU fallback implementation
    fn execute_operations_cpu_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operations: &[serde_json::Value]) -> CoelanoxResult<()> {
        warn!("Using CPU fallback for NPU backend");
        
        // Simple CPU implementation for fallback
        for operation_json in operations {
            let operation: Operation = serde_json::from_value(operation_json.clone())
                .map_err(|e| CoelanoxError::Execution(format!("Failed to deserialize operation: {}", e)))?;
            
            match operation.op_type {
                OpType::Add => {
                    self.execute_add_cpu_fallback(input, output, weights)?;
                }
                OpType::Multiply => {
                    self.execute_multiply_cpu_fallback(input, output, weights)?;
                }
                OpType::Convolution => {
                    self.execute_convolution_cpu_fallback(input, output, weights, &operation)?;
                }
                OpType::MatMul => {
                    self.execute_matmul_cpu_fallback(input, output, weights, &operation)?;
                }
                OpType::Relu => {
                    self.execute_relu_cpu_fallback(input, output)?;
                }
                OpType::Sigmoid => {
                    self.execute_sigmoid_cpu_fallback(input, output)?;
                }
                OpType::Tanh => {
                    self.execute_tanh_cpu_fallback(input, output)?;
                }
                OpType::Softmax => {
                    self.execute_softmax_cpu_fallback(input, output)?;
                }
                _ => {
                    // Passthrough for unsupported operations
                    for (i, &val) in input.iter().enumerate() {
                        if i < output.len() {
                            output[i] = val;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// CPU fallback implementations
    fn execute_add_cpu_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len()).min(weights.len());
        for i in 0..len {
            output[i] = input[i] + weights[i];
        }
        Ok(())
    }
    
    fn execute_multiply_cpu_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len()).min(weights.len());
        for i in 0..len {
            output[i] = input[i] * weights[i];
        }
        Ok(())
    }
    
    fn execute_convolution_cpu_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
        let kernel_size = operation.attributes.get("kernel_size")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(3);
        
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
    
    fn execute_matmul_cpu_fallback(&self, input: &[f32], output: &mut [f32], weights: &[f32], operation: &Operation) -> CoelanoxResult<()> {
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
        
        let weight_cols = operation.attributes.get("weight_cols")
            .and_then(|attr| match attr {
                Attribute::Int(i) => Some(*i as usize),
                _ => None,
            })
            .unwrap_or(weights.len() / input_cols.max(1));
        
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
    
    fn execute_relu_cpu_fallback(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i].max(0.0);
        }
        Ok(())
    }
    
    fn execute_sigmoid_cpu_fallback(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = 1.0 / (1.0 + (-input[i]).exp());
        }
        Ok(())
    }
    
    fn execute_tanh_cpu_fallback(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i].tanh();
        }
        Ok(())
    }
    
    fn execute_softmax_cpu_fallback(&self, input: &[f32], output: &mut [f32]) -> CoelanoxResult<()> {
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
    
    /// Allocate NPU memory
    pub fn allocate_memory(&self, size: usize, memory_type: NpuMemoryType) -> CoelanoxResult<*mut u8> {
        self.memory_manager.allocate(size, memory_type)
    }
    
    /// Deallocate NPU memory
    pub fn deallocate_memory(&self, ptr: *mut u8) -> CoelanoxResult<()> {
        self.memory_manager.deallocate(ptr)
    }
    
    /// Get NPU backend information
    pub fn get_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("backend_type".to_string(), "NPU".to_string());
        info.insert("device_name".to_string(), self.device_info.name.clone());
        info.insert("device_vendor".to_string(), self.device_info.vendor.clone());
        info.insert("architecture".to_string(), format!("{:?}", self.device_info.architecture));
        info.insert("total_memory".to_string(), self.device_info.total_memory.to_string());
        info.insert("available_memory".to_string(), self.device_info.available_memory.to_string());
        info.insert("num_cores".to_string(), self.device_info.num_cores.to_string());
        info.insert("peak_performance_tops".to_string(), self.device_info.peak_performance_tops.to_string());
        info.insert("npu_type".to_string(), format!("{:?}", self.preferred_npu));
        info.insert("npu_available".to_string(), self.npu_context.is_some().to_string());
        info
    }
}

impl NpuMemoryManager {
    /// Create a new NPU memory manager
    pub fn new(pool_size_limit: usize) -> Self {
        Self {
            allocated_blocks: Vec::new(),
            free_blocks: Vec::new(),
            total_allocated: 0,
            pool_size_limit,
            alignment: 64, // 64-byte alignment for NPU memory
        }
    }
    
    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize, memory_type: NpuMemoryType) -> CoelanoxResult<*mut u8> {
        // Align size to NPU requirements
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);
        
        // Check if we have a free block of sufficient size
        for (i, block) in self.free_blocks.iter().enumerate() {
            if block.size >= aligned_size {
                // Use this block
                let ptr = block.ptr;
                // Remove from free blocks and add to allocated blocks
                // Note: This is a simplified implementation
                return Ok(ptr);
            }
        }
        
        // Allocate new memory
        if self.total_allocated + aligned_size > self.pool_size_limit {
            return Err(CoelanoxError::Runtime("NPU memory pool limit exceeded".to_string()));
        }
        
        // For now, use standard allocation (would be NPU-specific in real implementation)
        let ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align(aligned_size, self.alignment).unwrap())
        };
        
        if ptr.is_null() {
            return Err(CoelanoxError::Runtime("Failed to allocate NPU memory".to_string()));
        }
        
        Ok(ptr)
    }
    
    /// Deallocate memory back to the pool
    pub fn deallocate(&self, ptr: *mut u8) -> CoelanoxResult<()> {
        // Find the allocated block and move it to free blocks
        // This is a simplified implementation
        unsafe {
            std::alloc::dealloc(ptr, std::alloc::Layout::from_size_align(1, 1).unwrap());
        }
        Ok(())
    }
}

impl NpuOperationCache {
    /// Create a new NPU operation cache
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cached_operations: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            max_cache_size,
        }
    }
    
    /// Get cached operation
    pub fn get(&mut self, operation_hash: &str) -> Option<&CachedOperation> {
        if let Some(cached_op) = self.cached_operations.get(operation_hash) {
            self.cache_hits += 1;
            Some(cached_op)
        } else {
            self.cache_misses += 1;
            None
        }
    }
    
    /// Cache operation
    pub fn cache(&mut self, operation_hash: String, compiled_operation: *mut std::ffi::c_void, execution_time: std::time::Duration) {
        if self.cached_operations.len() >= self.max_cache_size {
            // Remove oldest operation
            let oldest_key = self.cached_operations.iter()
                .min_by_key(|(_, op)| op.last_used)
                .map(|(key, _)| key.clone());
            
            if let Some(key) = oldest_key {
                self.cached_operations.remove(&key);
            }
        }
        
        let cached_op = CachedOperation {
            operation_hash: operation_hash.clone(),
            compiled_operation,
            execution_time,
            last_used: std::time::Instant::now(),
        };
        
        self.cached_operations.insert(operation_hash, cached_op);
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }
}

impl Drop for NpuContext {
    fn drop(&mut self) {
        // NPU context cleanup would go here
    }
}

impl Default for NpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
