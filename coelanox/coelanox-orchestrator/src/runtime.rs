//! COELANOX runtime for executing containers
//! 
//! This module provides the runtime functionality for executing COELANOX containers.
//! The runtime loads binary containers and executes the optimized machine code.
//! 
//! # Production Features
//! - Comprehensive error handling and logging
//! - Input validation and sanitization
//! - Memory safety and proper cleanup
//! - Performance monitoring and metrics
//! - Security hardening and validation
//! - Multi-backend execution support

use crate::{
    backend_manager::BackendManager,
    container_clfmm_backend::ContainerClfmmBackend,
    container_clfmp_backend::ContainerClfmpBackend,
    RuntimeBackendRegistry,
    ScalarExecutionBackend,
};
use coelanox_executor::{max_conv_scratch_elements, parse_plan, run_with_plan};
use coelanox_core::{
    format::{CoelanoxContainer, HardwareTarget},
    ir::{Attribute, OpType},
    memory_hal::{MemoryBackendRegistry, MemoryHAL, MemoryRegion, MemoryTarget, RegionOptions, MemoryTier},
    protection_hal::{ProtectionBackendRegistry, ProtectionHAL, RegionPermissions},
    CoelanoxError, CoelanoxResult,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn, instrument, Level};
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result as AnyhowResult};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// Percentile from sorted latencies (0.0..=100.0). Returns 0 if empty.
fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (pct / 100.0 * (sorted.len() as f64 - 1.0)).floor().max(0.0) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Returns current process RSS in bytes if available (Linux: /proc/self/status VmRSS; Windows: PROCESS_MEMORY_COUNTERS).
fn current_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let (_, rest) = line.split_once(':')?;
                let num: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some((num * 1024) as usize);
            }
        }
        None
    }
    #[cfg(windows)]
    {
        use std::mem::size_of;
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        unsafe {
            let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            counters.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
                Some(counters.WorkingSetSize)
            } else {
                None
            }
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = ();
        None
    }
}

/// Input descriptor for unified multi-input support
/// This structure allows passing any number of inputs to generated machine code
/// Using #[repr(C)] ensures C-compatible memory layout for generated code access
#[repr(C)]
pub struct InputDescriptor {
    /// Number of inputs
    pub num_inputs: usize,
    /// Array of fat pointers (ptr, len) for each input
    /// Each element is 16 bytes: 8 bytes ptr + 8 bytes len
    pub inputs: [FatPtr; 32], // Support up to 32 inputs (can be increased if needed)
}

/// Fat pointer representation matching Rust slice layout
/// ptr: *const f32, len: usize
#[repr(C)]
#[derive(Copy, Clone)]
pub struct FatPtr {
    /// Pointer to the input data
    pub ptr: *const f32,
    /// Length of the input data in elements
    pub len: usize,
}

impl InputDescriptor {
    /// Create a new input descriptor from input slices
    pub fn from_slices(input_slices: &[&[f32]]) -> Self {
        let mut descriptor = InputDescriptor {
            num_inputs: input_slices.len().min(32),
            inputs: [FatPtr { ptr: std::ptr::null(), len: 0 }; 32],
        };
        
        for (i, slice) in input_slices.iter().take(32).enumerate() {
            descriptor.inputs[i] = FatPtr {
                ptr: slice.as_ptr(),
                len: slice.len(),
            };
        }
        
        descriptor
    }
    
    /// Get the i-th input as a slice (unsafe, but necessary for generated code compatibility)
    pub unsafe fn get_input(&self, index: usize) -> Option<&[f32]> {
        if index >= self.num_inputs || index >= 32 {
            return None;
        }
        
        let fat_ptr = &self.inputs[index];
        if fat_ptr.ptr.is_null() || fat_ptr.len == 0 {
            return None;
        }
        
        Some(std::slice::from_raw_parts(fat_ptr.ptr, fat_ptr.len))
    }
}

/// Production-ready runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Maximum number of containers to cache (LRU eviction when exceeded)
    pub max_containers: usize,
    /// Optional TTL in seconds; containers not used within this period are evicted (None = no TTL)
    #[serde(default)]
    pub container_cache_ttl_secs: Option<u64>,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Execution timeout in milliseconds
    pub execution_timeout_ms: u64,
    /// Enable security validation
    pub enable_security_validation: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Log level for runtime operations
    pub log_level: String,
    /// Default backend to use
    pub default_backend: String,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Maximum input size in bytes
    pub max_input_size_bytes: usize,
    /// Enable input validation
    pub enable_input_validation: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_containers: 100,
            container_cache_ttl_secs: None,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            execution_timeout_ms: 30000, // 30 seconds
            enable_security_validation: true,
            enable_performance_monitoring: true,
            log_level: "info".to_string(),
            default_backend: "cpu".to_string(),
            enable_memory_pooling: true,
            max_input_size_bytes: 100 * 1024 * 1024, // 100MB
            enable_input_validation: true,
        }
    }
}

/// Input validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Execution status for tracking execution state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is pending
    Pending,
    /// Execution is running
    Running,
    /// Execution completed successfully
    Completed,
    /// Execution failed
    Failed,
    /// Execution timed out
    Timeout,
    /// Execution was cancelled
    Cancelled,
    /// Execution is retrying
    Retrying,
}

/// Execution error types for better error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionErrorType {
    /// Container-related errors
    ContainerError(String),
    /// Input validation errors
    InputError(String),
    /// Execution runtime errors
    RuntimeError(String),
    /// Timeout errors
    TimeoutError(String),
    /// Resource errors (memory, CPU, etc.)
    ResourceError(String),
    /// Security validation errors
    SecurityError(String),
    /// Backend-specific errors
    BackendError(String),
}

impl ExecutionErrorType {
    pub fn message(&self) -> String {
        match self {
            ExecutionErrorType::ContainerError(msg) => format!("Container error: {}", msg),
            ExecutionErrorType::InputError(msg) => format!("Input error: {}", msg),
            ExecutionErrorType::RuntimeError(msg) => format!("Runtime error: {}", msg),
            ExecutionErrorType::TimeoutError(msg) => format!("Timeout error: {}", msg),
            ExecutionErrorType::ResourceError(msg) => format!("Resource error: {}", msg),
            ExecutionErrorType::SecurityError(msg) => format!("Security error: {}", msg),
            ExecutionErrorType::BackendError(msg) => format!("Backend error: {}", msg),
        }
    }
    
    pub fn is_retryable(&self) -> bool {
        match self {
            ExecutionErrorType::TimeoutError(_) => true,
            ExecutionErrorType::ResourceError(_) => true,
            ExecutionErrorType::BackendError(_) => true,
            _ => false,
        }
    }
}

/// Error context for preserving debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique execution ID
    pub execution_id: String,
    /// Container name
    pub container_name: String,
    /// Input data size
    pub input_size: usize,
    /// Execution duration
    pub execution_time: Duration,
    /// Memory usage at time of error
    pub memory_usage: usize,
    /// Error chain for debugging
    pub error_chain: Vec<String>,
    /// Execution stage where error occurred
    pub error_stage: String,
    /// Additional context information
    pub context_info: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(execution_id: String, container_name: String) -> Self {
        Self {
            execution_id,
            container_name,
            input_size: 0,
            execution_time: Duration::ZERO,
            memory_usage: 0,
            error_chain: Vec::new(),
            error_stage: "unknown".to_string(),
            context_info: HashMap::new(),
        }
    }
    
    pub fn add_error(&mut self, error: String) {
        self.error_chain.push(error);
    }
    
    pub fn set_stage(&mut self, stage: &str) {
        self.error_stage = stage.to_string();
    }
    
    pub fn add_context(&mut self, key: String, value: String) {
        self.context_info.insert(key, value);
    }
}

/// Execution state for tracking and monitoring
#[derive(Debug, Clone)]
pub struct ExecutionState {
    /// Unique execution ID
    pub id: String,
    /// Current execution status
    pub status: ExecutionStatus,
    /// Execution progress (0.0 to 1.0)
    pub progress: f32,
    /// Execution start time
    pub start_time: Instant,
    /// Last update time
    pub last_update: Instant,
    /// Container name being executed
    pub container_name: String,
    /// Input data size
    pub input_size: usize,
    /// Error context if execution failed
    pub error_context: Option<ErrorContext>,
    /// Retry count
    pub retry_count: usize,
    /// Maximum retries allowed
    pub max_retries: usize,
    /// Execution timeout
    pub timeout: Duration,
    /// Output data from execution (stored here for retrieval)
    pub output: Option<Vec<f32>>,
}

impl ExecutionState {
    pub fn new(container_name: String, input_size: usize, timeout: Duration, max_retries: usize) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4().to_string(),
            status: ExecutionStatus::Pending,
            progress: 0.0,
            start_time: now,
            last_update: now,
            container_name,
            input_size,
            error_context: None,
            retry_count: 0,
            max_retries,
            timeout,
            output: None,
        }
    }
    
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
        self.last_update = Instant::now();
    }
    
    pub fn set_status(&mut self, status: ExecutionStatus) {
        self.status = status;
        self.last_update = Instant::now();
    }
    
    pub fn set_error(&mut self, error_type: ExecutionErrorType, stage: &str) {
        let mut error_context = ErrorContext::new(self.id.clone(), self.container_name.clone());
        error_context.input_size = self.input_size;
        error_context.execution_time = self.start_time.elapsed();
        error_context.set_stage(stage);
        error_context.add_error(error_type.message());
        
        self.error_context = Some(error_context);
        self.status = ExecutionStatus::Failed;
        self.last_update = Instant::now();
    }
    
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries && 
        self.error_context.as_ref()
            .map(|ctx| ctx.error_chain.last()
                .map(|err| err.contains("Timeout") || err.contains("Resource") || err.contains("Backend"))
                .unwrap_or(false))
            .unwrap_or(false)
    }
    
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
        self.status = ExecutionStatus::Retrying;
        self.progress = 0.0;
        self.last_update = Instant::now();
    }
    
    pub fn is_expired(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }
    
    pub fn set_output(&mut self, output: Vec<f32>) {
        self.output = Some(output);
    }
    
    pub fn get_output(&self) -> Option<Vec<f32>> {
        self.output.clone()
    }
}

/// Execution pipeline for monitoring execution stages
#[derive(Debug, Clone)]
pub struct ExecutionPipeline {
    /// Pipeline stages
    pub stages: Vec<ExecutionStage>,
    /// Current stage index
    pub current_stage: usize,
    /// Stage results
    pub stage_results: Vec<StageResult>,
    /// Pipeline start time
    pub start_time: Instant,
}

/// Execution stage information
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    /// Stage name
    pub name: String,
    /// Stage description
    pub description: String,
    /// Expected duration
    pub expected_duration: Duration,
    /// Stage dependencies
    pub dependencies: Vec<String>,
}

/// Stage execution result
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage name
    pub stage_name: String,
    /// Execution duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl ExecutionPipeline {
    pub fn new() -> Self {
        Self {
            stages: vec![
                ExecutionStage {
                    name: "validation".to_string(),
                    description: "Input and container validation".to_string(),
                    expected_duration: Duration::from_millis(1),
                    dependencies: vec![],
                },
                ExecutionStage {
                    name: "preparation".to_string(),
                    description: "Container preparation and warming".to_string(),
                    expected_duration: Duration::from_millis(5),
                    dependencies: vec!["validation".to_string()],
                },
                ExecutionStage {
                    name: "execution".to_string(),
                    description: "Machine code execution".to_string(),
                    expected_duration: Duration::from_millis(100),
                    dependencies: vec!["preparation".to_string()],
                },
                ExecutionStage {
                    name: "postprocessing".to_string(),
                    description: "Result processing and validation".to_string(),
                    expected_duration: Duration::from_millis(1),
                    dependencies: vec!["execution".to_string()],
                },
            ],
            current_stage: 0,
            stage_results: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    pub fn advance_stage(&mut self, result: StageResult) {
        self.stage_results.push(result);
        self.current_stage += 1;
    }
    
    pub fn get_progress(&self) -> f32 {
        if self.stages.is_empty() {
            return 1.0;
        }
        self.current_stage as f32 / self.stages.len() as f32
    }
    
    pub fn get_current_stage(&self) -> Option<&ExecutionStage> {
        self.stages.get(self.current_stage)
    }
}

/// COELANOX runtime for executing containers
pub struct CoelanoxRuntime {
    /// Runtime name
    name: String,
    /// Runtime version
    version: String,
    /// Runtime configuration
    config: Arc<RwLock<RuntimeConfig>>,
    /// Loaded containers cache (thread-safe). Arc avoids cloning; Instant used for TTL eviction.
    containers: Arc<RwLock<HashMap<String, (Arc<CoelanoxContainer>, Instant)>>>,
    /// Execution statistics (thread-safe)
    stats: Arc<Mutex<RuntimeStats>>,
    /// Executable memory blocks for cleanup (thread-safe)
    executable_memory: Arc<Mutex<Vec<ExecutableMemoryBlock>>>,
    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Security validator
    security_validator: Arc<SecurityValidator>,
    /// Active execution states (thread-safe)
    active_executions: Arc<RwLock<HashMap<String, ExecutionState>>>,
    /// Execution counter for unique IDs
    execution_counter: AtomicU64,
    /// Container warming cache (thread-safe)
    warmed_containers: Arc<RwLock<HashMap<String, Instant>>>,
    /// Memory HAL registry for executable/storage backends. When a backend is registered
    /// for executable, allocation uses HAL; otherwise legacy mmap/VirtualAlloc is used.
    memory_registry: Arc<RwLock<MemoryBackendRegistry>>,
    /// Protection HAL registry for MMU/MPU configuration. When a backend is registered,
    /// allocate_and_copy_code calls configure_region + enable_protection after loading code.
    protection_registry: Arc<RwLock<ProtectionBackendRegistry>>,
    /// Runtime execution backend registry. Holds fallback backends (e.g. scalar-fallback).
    /// The runtime tries CPU native machine code first, then falls back via this registry.
    runtime_backend_registry: Arc<RwLock<RuntimeBackendRegistry>>,
    /// Backend manager for dynamic resolution of "gpu"/"npu"/"tpu"/"edge" to discovered backends
    /// (e.g. CLF name "resnet_tiny_mnist.spv"). Same discovery as packager (COELANOX_CLF_PATH etc.).
    backend_manager: Arc<RwLock<BackendManager>>,
}

/// Simple memory region wrapper for legacy allocations. Implements MemoryRegion for
/// Protection HAL integration when no Memory HAL backend is used.
struct SimpleRegion {
    ptr: *mut u8,
    size: usize,
}

impl MemoryRegion for SimpleRegion {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn size(&self) -> usize {
        self.size
    }
}

unsafe impl Send for SimpleRegion {}
unsafe impl Sync for SimpleRegion {}

/// Bootstrapped executable state for a container. The runtime tracks this via `executable_memory`;
/// the pointer returned from `allocate_and_copy_code` is used to invoke the loaded code.
/// When the container has CLFMM, allocation is done by the container (self-bootstrap).

/// Executable memory block: legacy (mmap/VirtualAlloc), HAL-backed, or container CLFMM.
enum ExecutableMemoryBlock {
    Legacy {
        region: SimpleRegion,
        allocated_at: Instant,
    },
    Hal {
        target: MemoryTarget,
        region: Box<dyn MemoryRegion>,
        allocated_at: Instant,
    },
    /// Container-embedded CLFMM; backend owns deallocation
    Clfmm {
        region: Box<dyn MemoryRegion>,
        backend: std::sync::Arc<ContainerClfmmBackend>,
        allocated_at: Instant,
    },
}

impl ExecutableMemoryBlock {
    fn new_legacy(ptr: *mut u8, size: usize) -> Self {
        Self::Legacy {
            region: SimpleRegion { ptr, size },
            allocated_at: Instant::now(),
        }
    }

    /// Get region as &dyn MemoryRegion for Protection HAL.
    fn as_region_ref(&self) -> &dyn MemoryRegion {
        match self {
            Self::Legacy { region, .. } => region,
            Self::Hal { region, .. } | Self::Clfmm { region, .. } => region.as_ref(),
        }
    }

    fn new_hal(target: MemoryTarget, region: Box<dyn MemoryRegion>) -> Self {
        Self::Hal {
            target,
            region,
            allocated_at: Instant::now(),
        }
    }

    fn new_clfmm(region: Box<dyn MemoryRegion>, backend: std::sync::Arc<ContainerClfmmBackend>) -> Self {
        Self::Clfmm {
            region,
            backend,
            allocated_at: Instant::now(),
        }
    }

    /// Pointer to executable memory (for calling loaded code).
    fn as_ptr(&self) -> *mut u8 {
        match self {
            Self::Legacy { region, .. } => region.as_ptr(),
            Self::Hal { region, .. } | Self::Clfmm { region, .. } => region.as_ptr(),
        }
    }

    /// Size in bytes.
    fn size(&self) -> usize {
        match self {
            Self::Legacy { region, .. } => region.size(),
            Self::Hal { region, .. } | Self::Clfmm { region, .. } => region.size(),
        }
    }

    /// Get allocation age in milliseconds.
    fn age_ms(&self) -> u64 {
        let at = match self {
            Self::Legacy { allocated_at, .. } | Self::Hal { allocated_at, .. } | Self::Clfmm { allocated_at, .. } => allocated_at,
        };
        at.elapsed().as_millis() as u64
    }
}

/// Performance metrics for runtime monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total memory allocated (bytes)
    pub total_memory_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Current memory usage (bytes)
    pub current_memory_usage: usize,
    /// Total executable memory blocks
    pub executable_blocks_count: usize,
    /// Average execution time (microseconds)
    pub avg_execution_time_us: u64,
    /// Total executions completed
    pub total_executions: u64,
    /// Failed executions count
    pub failed_executions: u64,
    /// Last execution timestamp
    pub last_execution_time: Option<Instant>,
    /// Throughput (executions per second)
    pub throughput_ops_per_sec: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_memory_allocated: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            executable_blocks_count: 0,
            avg_execution_time_us: 0,
            total_executions: 0,
            failed_executions: 0,
            last_execution_time: None,
            throughput_ops_per_sec: 0.0,
        }
    }
    
    pub fn update_execution(&mut self, execution_time_us: u64, memory_usage: usize, success: bool) {
        self.total_executions += 1;
        if !success {
            self.failed_executions += 1;
        }
        
        // Update average execution time
        self.avg_execution_time_us = 
            (self.avg_execution_time_us * (self.total_executions - 1) + execution_time_us) / self.total_executions;
        
        // Update memory metrics
        self.current_memory_usage = memory_usage;
        if memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = memory_usage;
        }
        
        // Update throughput
        if self.avg_execution_time_us > 0 {
            self.throughput_ops_per_sec = 1_000_000.0 / self.avg_execution_time_us as f64;
        }
        
        self.last_execution_time = Some(Instant::now());
    }
    
    pub fn update_memory_allocation(&mut self, size: usize) {
        self.total_memory_allocated += size;
        self.current_memory_usage += size;
        if self.current_memory_usage > self.peak_memory_usage {
            self.peak_memory_usage = self.current_memory_usage;
        }
    }
    
    pub fn update_memory_deallocation(&mut self, size: usize) {
        self.current_memory_usage = self.current_memory_usage.saturating_sub(size);
    }
}

/// Security validator for input validation and sanitization
#[derive(Debug)]
pub struct SecurityValidator {
    /// Maximum input size in bytes
    max_input_size_bytes: usize,
    /// Maximum container size in bytes
    max_container_size_bytes: usize,
    /// Enable path traversal protection
    enable_path_traversal_protection: bool,
    /// Enable memory bounds checking
    enable_memory_bounds_checking: bool,
}

impl SecurityValidator {
    pub fn new(config: &RuntimeConfig) -> Self {
        Self {
            max_input_size_bytes: config.max_input_size_bytes,
            max_container_size_bytes: config.max_memory_bytes,
            enable_path_traversal_protection: config.enable_security_validation,
            enable_memory_bounds_checking: config.enable_security_validation,
        }
    }
    
    /// Validate input data for security and size constraints
    pub fn validate_input(&self, input_data: &[f32], expected_size: usize) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Check input size
        if input_data.len() != expected_size {
            result.add_error(format!(
                "Input size mismatch: expected {}, got {}",
                expected_size,
                input_data.len()
            ));
        }
        
        // Check maximum input size
        let input_size_bytes = input_data.len() * std::mem::size_of::<f32>();
        if input_size_bytes > self.max_input_size_bytes {
            result.add_error(format!(
                "Input size exceeds maximum allowed: {} bytes > {} bytes",
                input_size_bytes,
                self.max_input_size_bytes
            ));
        }
        
        // Check for NaN or infinite values
        for (i, &value) in input_data.iter().enumerate() {
            if !value.is_finite() {
                result.add_error(format!(
                    "Invalid input value at index {}: {} (NaN or infinite)",
                    i, value
                ));
            }
        }
        
        result
    }
    
    /// Validate container path for security
    pub fn validate_container_path(&self, path: &str) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        if self.enable_path_traversal_protection {
            // Check for path traversal attempts
            if path.contains("..") || path.contains("~") {
                result.add_error(format!(
                    "Path traversal detected in container path: {}",
                    path
                ));
            }
            
            // Check for absolute paths (security risk)
            if path.starts_with('/') || path.starts_with('\\') {
                result.add_warning(format!(
                    "Absolute path detected: {} (consider using relative paths)",
                    path
                ));
            }
        }
        
        result
    }
    
    /// Validate container size
    pub fn validate_container_size(&self, size: usize) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        if size > self.max_container_size_bytes {
            result.add_error(format!(
                "Container size exceeds maximum allowed: {} bytes > {} bytes",
                size,
                self.max_container_size_bytes
            ));
        }
        
        result
    }
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
    /// Create a new COELANOX runtime with default configuration
    pub fn new() -> Self {
        Self::with_config(RuntimeConfig::default())
    }
    
    /// Create a new COELANOX runtime with custom configuration
    pub fn with_config(config: RuntimeConfig) -> Self {
        let security_validator = Arc::new(SecurityValidator::new(&config));
        
        Self {
            name: "coelanox_orchestrator".to_string(),
            version: "1.0.0".to_string(),
            config: Arc::new(RwLock::new(config)),
            containers: Arc::new(RwLock::new(HashMap::<String, (Arc<CoelanoxContainer>, Instant)>::new())),
            stats: Arc::new(Mutex::new(RuntimeStats::new())),
            executable_memory: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            security_validator,
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            execution_counter: AtomicU64::new(0),
            warmed_containers: Arc::new(RwLock::new(HashMap::new())),
            memory_registry: Arc::new(RwLock::new(MemoryBackendRegistry::new())),
            protection_registry: Arc::new(RwLock::new(ProtectionBackendRegistry::new())),
            runtime_backend_registry: {
                let mut reg = RuntimeBackendRegistry::new();
                reg.register("scalar-fallback", Box::new(ScalarExecutionBackend::new()));
                Arc::new(RwLock::new(reg))
            },
            backend_manager: {
                let mut mgr = BackendManager::new();
                let paths = BackendManager::default_search_paths();
                let _ = mgr.discover_backends(&paths);
                Arc::new(RwLock::new(mgr))
            },
        }
    }

    /// Access the runtime backend registry (fallback execution backends).
    pub fn runtime_backend_registry(&self) -> std::sync::RwLockReadGuard<'_, RuntimeBackendRegistry> {
        self.runtime_backend_registry.read().unwrap()
    }

    /// Mutable access to the runtime backend registry.
    pub fn runtime_backend_registry_mut(&self) -> std::sync::RwLockWriteGuard<'_, RuntimeBackendRegistry> {
        self.runtime_backend_registry.write().unwrap()
    }

    /// Access the protection HAL registry. When a backend is registered, it is invoked
    /// after allocate_and_copy_code to configure region permissions (Load & Lock).
    pub fn protection_registry(&self) -> std::sync::RwLockReadGuard<'_, ProtectionBackendRegistry> {
        self.protection_registry.read().unwrap()
    }

    /// Mutable access to the protection HAL registry.
    pub fn protection_registry_mut(&self) -> std::sync::RwLockWriteGuard<'_, ProtectionBackendRegistry> {
        self.protection_registry.write().unwrap()
    }

    /// Access the memory HAL registry to register executable/storage backends.
    /// When a backend is registered for executable, allocation uses HAL.
    pub fn memory_registry(&self) -> std::sync::RwLockReadGuard<'_, MemoryBackendRegistry> {
        self.memory_registry.read().unwrap()
    }

    /// Mutable access to the memory HAL registry (e.g. to register backends).
    pub fn memory_registry_mut(&self) -> std::sync::RwLockWriteGuard<'_, MemoryBackendRegistry> {
        self.memory_registry.write().unwrap()
    }
    
    /// Get runtime configuration
    pub fn get_config(&self) -> RuntimeConfig {
        self.config.read().unwrap().clone()
    }
    
    /// Update runtime configuration
    pub fn update_config(&self, config: RuntimeConfig) -> CoelanoxResult<()> {
        // Validate configuration
        self.validate_config(&config)?;
        
        // Update configuration
        *self.config.write().unwrap() = config;
        
        // Recreate security validator with new config
        let new_security_validator = Arc::new(SecurityValidator::new(&self.get_config()));
        // Note: We can't directly replace Arc<SecurityValidator> in the struct,
        // but we can update the internal configuration
        
        Ok(())
    }
    
    /// Validate runtime configuration
    fn validate_config(&self, config: &RuntimeConfig) -> CoelanoxResult<()> {
        if config.max_containers == 0 {
            return Err(CoelanoxError::configuration("max_containers cannot be zero"));
        }
        
        if config.max_memory_bytes == 0 {
            return Err(CoelanoxError::configuration("max_memory_bytes cannot be zero"));
        }
        
        if config.execution_timeout_ms == 0 {
            return Err(CoelanoxError::configuration("execution_timeout_ms cannot be zero"));
        }
        
        if config.max_input_size_bytes == 0 {
            return Err(CoelanoxError::configuration("max_input_size_bytes cannot be zero"));
        }
        
        Ok(())
    }
    
    /// Load a COELANOX container from file with security validation
    pub fn load_container(&self, path: &str) -> CoelanoxResult<()> {
        // Security validation
        let path_validation = self.security_validator.validate_container_path(path);
        if !path_validation.is_valid {
            return Err(CoelanoxError::SecurityValidation(
                path_validation.errors.join("; ")
            ));
        }
        
        // Log warnings if any
        for warning in &path_validation.warnings {
            warn!("Security warning: {}", warning);
        }
        
        // Load container
        let container = CoelanoxContainer::read_binary_file(std::path::Path::new(path))
            .map_err(|e| CoelanoxError::Runtime(format!("Failed to load container from path: {}: {}", path, e)))?;
        
        // Validate container size
        let size_validation = self.security_validator.validate_container_size(container.total_size());
        if !size_validation.is_valid {
            return Err(CoelanoxError::SecurityValidation(
                size_validation.errors.join("; ")
            ));
        }
        
        // Validate container integrity
        container.validate()
            .map_err(|e| CoelanoxError::Runtime(format!("Container validation failed for: {}: {}", path, e)))?;
        
        // Check container cache limits
        let config = self.get_config();
        let mut containers = self.containers.write().unwrap();
        
        // Evict expired containers when TTL is set
        if config.container_cache_ttl_secs.is_some() {
            self.evict_expired_containers_inner(&mut containers, &config);
        }
        if containers.len() >= config.max_containers {
            // Remove one arbitrary (first key) when at max
            if let Some(oldest_key) = containers.keys().next().cloned() {
                containers.remove(&oldest_key);
                info!("Removed container from cache (max_containers): {}", oldest_key);
            }
        }
        
        // Store in cache with load time for TTL eviction
        let container = Arc::new(container);
        let total_size = container.total_size();
        containers.insert(path.to_string(), (container, Instant::now()));
        
        info!("Successfully loaded container: {} ({} bytes)", path, total_size);
        
        Ok(())
    }
    
    /// Load a COELANOX container from binary data with security validation
    pub fn load_container_from_data(&self, name: &str, data: &[u8]) -> CoelanoxResult<()> {
        // Security validation
        let path_validation = self.security_validator.validate_container_path(name);
        if !path_validation.is_valid {
            return Err(CoelanoxError::SecurityValidation(
                path_validation.errors.join("; ")
            ));
        }
        
        // Load container
        let container = CoelanoxContainer::from_binary(data)
            .map_err(|e| CoelanoxError::Runtime(format!("Failed to load container from binary data: {}: {}", name, e)))?;
        
        // Validate container size
        let size_validation = self.security_validator.validate_container_size(container.total_size());
        if !size_validation.is_valid {
            return Err(CoelanoxError::SecurityValidation(
                size_validation.errors.join("; ")
            ));
        }
        
        // Validate container integrity
        container.validate()
            .map_err(|e| CoelanoxError::Runtime(format!("Container validation failed for: {}: {}", name, e)))?;
        
        // Check container cache limits
        let config = self.get_config();
        let mut containers = self.containers.write().unwrap();
        
        if config.container_cache_ttl_secs.is_some() {
            self.evict_expired_containers_inner(&mut containers, &config);
        }
        if containers.len() >= config.max_containers {
            if let Some(oldest_key) = containers.keys().next().cloned() {
                containers.remove(&oldest_key);
                info!("Removed container from cache (max_containers): {}", oldest_key);
            }
        }
        
        let container = Arc::new(container);
        let total_size = container.total_size();
        containers.insert(name.to_string(), (container, Instant::now()));
        
        info!("Successfully loaded container from data: {} ({} bytes)", name, total_size);
        
        Ok(())
    }
    
    /// Execute a COELANOX container by name with comprehensive validation and monitoring.
    /// Uses "auto" backend selection (backend manager chooses based on detected hardware).
    pub fn execute(&self, container_name: &str, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        self.execute_with_backend(container_name, input_data, "auto")
    }

    /// Execute a COELANOX container with retry logic and timeout enforcement
    pub fn execute_with_retry(&self, container_name: &str, input_data: &[f32], max_retries: usize, timeout: Duration) -> CoelanoxResult<Vec<f32>> {
        let mut execution_state = ExecutionState::new(
            container_name.to_string(),
            input_data.len(),
            timeout,
            max_retries,
        );
        
        // Register execution state
        {
            let mut active_executions = self.active_executions.write().unwrap();
            active_executions.insert(execution_state.id.clone(), execution_state.clone());
        }
        
        let result = self.execute_with_state(&mut execution_state, input_data);
        
        // Clean up execution state
        {
            let mut active_executions = self.active_executions.write().unwrap();
            active_executions.remove(&execution_state.id);
        }
        
        result
    }
    
    /// Execute with progress tracking and state management
    pub fn execute_with_progress<F>(&self, container_name: &str, input_data: &[f32], progress_callback: F) -> CoelanoxResult<Vec<f32>>
    where
        F: Fn(f32) + Send + Sync,
    {
        let config = self.get_config();
        let timeout = Duration::from_millis(config.execution_timeout_ms);
        let max_retries = 3;
        
        let mut execution_state = ExecutionState::new(
            container_name.to_string(),
            input_data.len(),
            timeout,
            max_retries,
        );
        
        // Register execution state
        {
            let mut active_executions = self.active_executions.write().unwrap();
            active_executions.insert(execution_state.id.clone(), execution_state.clone());
        }
        
        let result = self.execute_with_state_and_progress(&mut execution_state, input_data, progress_callback);
        
        // Clean up execution state
        {
            let mut active_executions = self.active_executions.write().unwrap();
            active_executions.remove(&execution_state.id);
        }
        
        result
    }
    
    /// Execute with execution state tracking
    fn execute_with_state(&self, execution_state: &mut ExecutionState, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        let mut pipeline = ExecutionPipeline::new();
        
        loop {
            // Check for timeout
            if execution_state.is_expired() {
                execution_state.set_error(
                    ExecutionErrorType::TimeoutError("Execution timeout exceeded".to_string()),
                    "timeout_check"
                );
                return Err(CoelanoxError::Execution("Execution timeout".to_string()));
            }
            
            // Execute with pipeline monitoring
            match self.execute_with_pipeline(execution_state, input_data, &mut pipeline) {
                Ok(result) => {
                    execution_state.set_status(ExecutionStatus::Completed);
                    execution_state.update_progress(1.0);
                    return Ok(result);
                }
                Err(e) => {
                    let error_type = self.categorize_error(&e);
                    execution_state.set_error(error_type.clone(), pipeline.get_current_stage().map(|s| s.name.as_str()).unwrap_or("unknown"));
                    
                    if execution_state.can_retry() && error_type.is_retryable() {
                        execution_state.increment_retry();
                        warn!("Retrying execution {} (attempt {}/{})", execution_state.id, execution_state.retry_count, execution_state.max_retries);
                        
                        // Exponential backoff
                        let backoff_duration = Duration::from_millis(100 * (2_u64.pow(execution_state.retry_count as u32)));
                        std::thread::sleep(backoff_duration);
                        
                        // Reset pipeline for retry
                        pipeline = ExecutionPipeline::new();
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    
    /// Execute with state and progress tracking
    fn execute_with_state_and_progress<F>(&self, execution_state: &mut ExecutionState, input_data: &[f32], progress_callback: F) -> CoelanoxResult<Vec<f32>>
    where
        F: Fn(f32) + Send + Sync,
    {
        let mut pipeline = ExecutionPipeline::new();
        
        loop {
            // Check for timeout
            if execution_state.is_expired() {
                execution_state.set_error(
                    ExecutionErrorType::TimeoutError("Execution timeout exceeded".to_string()),
                    "timeout_check"
                );
                return Err(CoelanoxError::Execution("Execution timeout".to_string()));
            }
            
            // Update progress
            let progress = pipeline.get_progress();
            execution_state.update_progress(progress);
            progress_callback(progress);
            
            // Execute with pipeline monitoring
            match self.execute_with_pipeline(execution_state, input_data, &mut pipeline) {
                Ok(result) => {
                    execution_state.set_status(ExecutionStatus::Completed);
                    execution_state.update_progress(1.0);
                    progress_callback(1.0);
                    return Ok(result);
                }
                Err(e) => {
                    let error_type = self.categorize_error(&e);
                    execution_state.set_error(error_type.clone(), pipeline.get_current_stage().map(|s| s.name.as_str()).unwrap_or("unknown"));
                    
                    if execution_state.can_retry() && error_type.is_retryable() {
                        execution_state.increment_retry();
                        warn!("Retrying execution {} (attempt {}/{})", execution_state.id, execution_state.retry_count, execution_state.max_retries);
                        
                        // Exponential backoff
                        let backoff_duration = Duration::from_millis(100 * (2_u64.pow(execution_state.retry_count as u32)));
                        std::thread::sleep(backoff_duration);
                        
                        // Reset pipeline for retry
                        pipeline = ExecutionPipeline::new();
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    
    /// Execute with pipeline monitoring
    fn execute_with_pipeline(&self, execution_state: &mut ExecutionState, input_data: &[f32], pipeline: &mut ExecutionPipeline) -> CoelanoxResult<Vec<f32>> {
        let start_time = Instant::now();
        
        // Stage 1: Validation
        execution_state.set_status(ExecutionStatus::Running);
        let validation_result = self.execute_validation_stage(execution_state, input_data, pipeline);
        if !validation_result.success {
            return Err(CoelanoxError::Execution(format!("Validation failed: {:?}", validation_result.error)));
        }
        
        // Stage 2: Preparation
        let preparation_result = self.execute_preparation_stage(execution_state, pipeline);
        if !preparation_result.success {
            return Err(CoelanoxError::Execution(format!("Preparation failed: {:?}", preparation_result.error)));
        }
        
        // Stage 3: Execution
        let execution_result = self.execute_execution_stage(execution_state, input_data, pipeline);
        if !execution_result.success {
            return Err(CoelanoxError::Execution(format!("Execution failed: {:?}", execution_result.error)));
        }
        
        // Stage 4: Post-processing
        let postprocessing_result = self.execute_postprocessing_stage(execution_state, pipeline);
        if !postprocessing_result.success {
            return Err(CoelanoxError::Execution(format!("Post-processing failed: {:?}", postprocessing_result.error)));
        }
        
        // Extract result from execution stage
        // The output is stored in execution_state.output, not in metrics
        let result = if let Some(output) = execution_state.get_output() {
            output
        } else {
            // Fallback: try to get from metrics (legacy code path)
            execution_result.metrics.get("result")
                .and_then(|r| Some(vec![*r as f32]))
                .unwrap_or_else(|| vec![0.0])
        };
        
        Ok(result)
    }
    
    /// Execute validation stage
    fn execute_validation_stage(&self, execution_state: &mut ExecutionState, input_data: &[f32], pipeline: &mut ExecutionPipeline) -> StageResult {
        let stage_start = Instant::now();
        
        // Get container from cache (Arc clone, not full container clone)
        let container = {
            let containers = self.containers.read().unwrap();
            match containers.get(&execution_state.container_name) {
                Some((container, _)) => Arc::clone(container),
                None => {
                    return StageResult {
                        stage_name: "validation".to_string(),
                        duration: stage_start.elapsed(),
                        success: false,
                        error: Some("Container not found".to_string()),
                        metrics: HashMap::new(),
                    };
                }
            }
        };
        
        // Validate input size
        let expected_input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        // Security validation
        let input_validation = self.security_validator.validate_input(input_data, expected_input_size);
        if !input_validation.is_valid {
            return StageResult {
                stage_name: "validation".to_string(),
                duration: stage_start.elapsed(),
                success: false,
                error: Some(format!("Input validation failed: {}", input_validation.errors.join("; "))),
                metrics: HashMap::new(),
            };
        }
        
        pipeline.advance_stage(StageResult {
            stage_name: "validation".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        });
        
        StageResult {
            stage_name: "validation".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        }
    }
    
    /// Execute preparation stage
    fn execute_preparation_stage(&self, execution_state: &mut ExecutionState, pipeline: &mut ExecutionPipeline) -> StageResult {
        let stage_start = Instant::now();
        
        // Warm container if not already warmed
        let container_name = &execution_state.container_name;
        let is_warmed = {
            let warmed_containers = self.warmed_containers.read().unwrap();
            warmed_containers.contains_key(container_name)
        };
        
        if !is_warmed {
            // Warm container (preload and prepare)
            if let Err(e) = self.warm_container(container_name) {
                return StageResult {
                    stage_name: "preparation".to_string(),
                    duration: stage_start.elapsed(),
                    success: false,
                    error: Some(format!("Container warming failed: {:?}", e)),
                    metrics: HashMap::new(),
                };
            }
        }
        
        pipeline.advance_stage(StageResult {
            stage_name: "preparation".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        });
        
        StageResult {
            stage_name: "preparation".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        }
    }
    
    /// Execute execution stage
    fn execute_execution_stage(&self, execution_state: &mut ExecutionState, input_data: &[f32], pipeline: &mut ExecutionPipeline) -> StageResult {
        let stage_start = Instant::now();
        
        // Get container from cache (Arc clone, not full container clone)
        let container = {
            let containers = self.containers.read().unwrap();
            Arc::clone(&containers.get(&execution_state.container_name).unwrap().0)
        };
        
        // Execute using backend selection (auto = backend manager chooses from detected hardware)
        let result = self.run_backend_execution(container.as_ref(), input_data, "auto");
        
        match result {
            Ok(output) => {
                let mut metrics = HashMap::new();
                metrics.insert("output_size".to_string(), output.len() as f64);
                metrics.insert("execution_time_us".to_string(), stage_start.elapsed().as_micros() as f64);
                
                // Store output in execution_state for direct access
                execution_state.set_output(output.clone());
                
                pipeline.advance_stage(StageResult {
                    stage_name: "execution".to_string(),
                    duration: stage_start.elapsed(),
                    success: true,
                    error: None,
                    metrics: metrics.clone(),
                });
                
                StageResult {
                    stage_name: "execution".to_string(),
                    duration: stage_start.elapsed(),
                    success: true,
                    error: None,
                    metrics,
                }
            }
            Err(e) => {
                StageResult {
                    stage_name: "execution".to_string(),
                    duration: stage_start.elapsed(),
                    success: false,
                    error: Some(format!("Execution failed: {:?}", e)),
                    metrics: HashMap::new(),
                }
            }
        }
    }
    
    /// Execute post-processing stage
    fn execute_postprocessing_stage(&self, execution_state: &mut ExecutionState, pipeline: &mut ExecutionPipeline) -> StageResult {
        let stage_start = Instant::now();
        
        // Post-processing validation and cleanup
        pipeline.advance_stage(StageResult {
            stage_name: "postprocessing".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        });
        
        StageResult {
            stage_name: "postprocessing".to_string(),
            duration: stage_start.elapsed(),
            success: true,
            error: None,
            metrics: HashMap::new(),
        }
    }
    
    /// Categorize error for retry logic
    fn categorize_error(&self, error: &CoelanoxError) -> ExecutionErrorType {
        match error {
            CoelanoxError::Runtime(msg) => ExecutionErrorType::RuntimeError(msg.clone()),
            CoelanoxError::SecurityValidation(msg) => ExecutionErrorType::SecurityError(msg.clone()),
            CoelanoxError::FileSystem(_) => ExecutionErrorType::ResourceError("File system error".to_string()),
            CoelanoxError::Backend(msg) => ExecutionErrorType::BackendError(msg.clone()),
            CoelanoxError::Execution(msg) => ExecutionErrorType::RuntimeError(msg.clone()),
            _ => ExecutionErrorType::RuntimeError(format!("Unknown error: {:?}", error)),
        }
    }
    
    /// Warm container for faster execution
    pub fn warm_container(&self, container_name: &str) -> CoelanoxResult<()> {
        // Check if container is loaded
        {
            let containers = self.containers.read().unwrap();
            if !containers.contains_key(container_name) {
                return Err(CoelanoxError::Runtime(format!("Container not loaded: {}", container_name)));
            }
        }
        
        // Mark container as warmed
        {
            let mut warmed_containers = self.warmed_containers.write().unwrap();
            warmed_containers.insert(container_name.to_string(), Instant::now());
        }
        
        info!("🔥 Container warmed: {}", container_name);
        Ok(())
    }
    
    /// Preload container for faster access
    pub fn preload_container(&self, container_name: &str) -> CoelanoxResult<()> {
        // Check if container is already loaded
        {
            let containers = self.containers.read().unwrap();
            if containers.contains_key(container_name) {
                return Ok(()); // Already loaded
            }
        }
        
        // Load container
        self.load_container(container_name)?;
        
        // Warm container
        self.warm_container(container_name)?;
        
        info!("📦 Container preloaded: {}", container_name);
        Ok(())
    }
    
    /// Execute batch of inputs
    pub fn execute_batch(&self, container_name: &str, inputs: Vec<Vec<f32>>) -> CoelanoxResult<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        
        for (i, input) in inputs.into_iter().enumerate() {
            match self.execute(container_name, &input) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Batch execution failed at index {}: {:?}", i, e);
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get active execution states
    pub fn get_active_executions(&self) -> Vec<ExecutionState> {
        let active_executions = self.active_executions.read().unwrap();
        active_executions.values().cloned().collect()
    }
    
    /// Get execution state by ID
    pub fn get_execution_state(&self, execution_id: &str) -> Option<ExecutionState> {
        let active_executions = self.active_executions.read().unwrap();
        active_executions.get(execution_id).cloned()
    }
    
    /// Cancel execution by ID
    pub fn cancel_execution(&self, execution_id: &str) -> CoelanoxResult<()> {
        let mut active_executions = self.active_executions.write().unwrap();
        if let Some(execution_state) = active_executions.get_mut(execution_id) {
            execution_state.set_status(ExecutionStatus::Cancelled);
            info!("🚫 Execution cancelled: {}", execution_id);
            Ok(())
        } else {
            Err(CoelanoxError::Runtime(format!("Execution not found: {}", execution_id)))
        }
    }
    
    /// Execute machine code from container with proper memory management.
    /// When container has an execution plan (CLF multi-blob), uses the mini-executor to dispatch to kernel blobs.
    /// Otherwise uses single-entry-point native call (or falls back to scalar when backend != cpu).
    fn execute_machine_code(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // No embedded code (e.g. packaged with --fallback-only): force scalar fallback via error so caller's or_else runs
        if container.optimized_code.is_empty() && container.execution_plan.is_none() {
            return Err(CoelanoxError::Runtime(
                "Container has no embedded backend code (fallback-only package); use scalar execution".to_string(),
            ));
        }
        // CLF multi-blob path: use mini-executor with execution plan
        if let Some(ref plan) = container.execution_plan {
            if !container.optimized_code.is_empty() {
                return self.execute_via_plan(container, input_data, plan);
            }
        }
        // Single-blob path: require primary_backend == "cpu"
        if let Some(ref b) = container.manifest.backend_metadata {
            if b.primary_backend.to_lowercase() != "cpu" {
                return Err(CoelanoxError::Runtime(format!(
                    "Container code is for backend '{}', not CPU; use scalar fallback",
                    b.primary_backend
                )));
            }
        }
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        let mut output = vec![0.0f32; output_size];
        let expected_input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        if input_data.len() != expected_input_size {
            return Err(CoelanoxError::Runtime(format!(
                "Input data size mismatch: expected {} elements (combined), got {}",
                expected_input_size,
                input_data.len()
            )));
        }
        let weights = container.decompress_weights()?;
        type CpuFunc = unsafe extern "sysv64" fn(
            *const f32, usize,
            *mut f32, usize,
            *const f32, usize
        );
        let ptr = self.allocate_and_copy_code(container)?;
        let func: CpuFunc = unsafe { std::mem::transmute(ptr as usize) };
        info!("🔍 Calling generated CPU native machine code function with {} input elements, {} output elements",
              input_data.len(), output.len());
        info!("✅ Using CPU native execution (no Rust fallback)");
        unsafe {
            func(
                input_data.as_ptr(), input_data.len(),
                output.as_mut_ptr(), output.len(),
                weights.as_ptr(), weights.len(),
            );
        }
        info!("✅ CPU native execution completed");
        Ok(output)
    }

    /// Execute via CLF mini-executor: parse plan, allocate workspace, copy code, run_with_plan.
    fn execute_via_plan(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        plan: &[u8],
    ) -> CoelanoxResult<Vec<f32>> {
        let (workspace_elements, _steps, _offsets) = parse_plan(plan)
            .map_err(|e| CoelanoxError::Runtime(format!("Invalid execution plan: {}", e)))?;
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        let expected_input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        if input_data.len() < expected_input_size {
            return Err(CoelanoxError::Runtime(format!(
                "Input size mismatch: expected {} elements, got {}",
                expected_input_size,
                input_data.len()
            )));
        }
        // Optional timing breakdown (log with RUST_LOG=debug to see where SIMD path time goes)
        let t0 = std::time::Instant::now();
        let weights = container.decompress_weights()?;
        debug!("SIMD path: decompress_weights {} ms", t0.elapsed().as_millis());
        let t1 = std::time::Instant::now();
        let scratch_elements = max_conv_scratch_elements(plan).unwrap_or(0);
        let mut workspace = vec![0.0f32; workspace_elements as usize + scratch_elements];
        let mut output = vec![0.0f32; output_size];
        debug!("SIMD path: workspace+output alloc {} ms ({} elements)", t1.elapsed().as_millis(), workspace.len());
        // Code must be in executable memory for kernel blobs to be called
        let t2 = std::time::Instant::now();
        let _ptr = self.allocate_and_copy_code(container)?;
        debug!("SIMD path: allocate_and_copy_code {} ms", t2.elapsed().as_millis());
        // run_with_plan expects code_section - we just copied it to executable memory; the kernel
        // pointers we pass are from the plan (offsets into code section). So we need to pass the
        // code section content (same bytes now in executable memory). The executor will call into
        // the copied code via the offsets. So we pass container.optimized_code as the code_section
        // for reading offsets; the actual calls go to _ptr + offset. So run_with_plan needs to use
        // the executable region. So we need to change run_with_plan to accept code_base pointer
        // (executable) instead of &[u8], so it can call code_base + offset. Let me check the
        // executor again.
        // In run_with_plan we do: let code_base = code_section.as_ptr() as usize; ... let kernel_ptr = (code_base + k_off) as *const u8; func: KernelFn = transmute(kernel_ptr). So we're calling into code_section's memory. So code_section must be executable. So we need to pass the executable copy, not the container's optimized_code. So we need the executor to receive a pointer (the executable base) and the length (to bounds-check offsets). So change run_with_plan to take (code_ptr: *const u8, code_len: usize, plan, ...) or we pass the executable region as a slice - but we can't get a slice from raw pointer without knowing it's valid. So the runtime allocates and copies, then we need to pass the slice. The runtime has the block in executable_memory; the pointer is _ptr. We could make a slice from ptr and length: unsafe { std::slice::from_raw_parts(_ptr, container.optimized_code.len()) }. That is valid for the duration we don't free it. So we hold the guard so the block isn't freed, then run_with_plan(unsafe { std::slice::from_raw_parts(_ptr, container.optimized_code.len()) }, plan, ...). So we need to keep the allocation alive. allocate_and_copy_code pushes the block and returns the pointer; the block stays in the list. So we can safely use the slice until we return. So:
        let code_slice = unsafe {
            std::slice::from_raw_parts(_ptr, container.optimized_code.len())
        };
        let mut audit_cb = |step_ix: usize, op_id: u32, out_slice: &[f32]| {
            let op_type = format!("op_id_{}", op_id);
            crate::audit::log_after_op(step_ix, &op_type, out_slice.len(), out_slice);
        };
        let t3 = std::time::Instant::now();
        run_with_plan(
            code_slice,
            plan,
            input_data,
            &mut output,
            &weights,
            &mut workspace,
            if crate::audit::is_enabled() {
                Some(&mut audit_cb)
            } else {
                None
            },
        )?;
        debug!("SIMD path: run_with_plan (kernel execution) {} ms", t3.elapsed().as_millis());
        info!("✅ CLF mini-executor execution completed");
        Ok(output)
    }
    
    // Removed create_executable_function: we use raw calling convention above for correctness

    /// Apply protection (configure + enable) to the last allocated executable block.
    /// Prefers container-embedded CLFMP when present; otherwise uses protection registry.
    fn apply_protection(&self, container: &CoelanoxContainer, _ptr: *mut u8) -> CoelanoxResult<()> {
        let guard = self.executable_memory.lock().unwrap();
        let Some(block) = guard.last() else {
            return Ok(());
        };
        let region = block.as_region_ref();
        if let Ok(Some(prot)) = ContainerClfmpBackend::from_container(container) {
            if let Err(e) = prot.configure_region(region, RegionPermissions::code()) {
                warn!("Container CLFMP configure_region failed: {:?}", e);
            } else if let Err(e) = prot.enable_protection() {
                warn!("Container CLFMP enable_protection failed: {:?}", e);
            }
        } else if let Some((_, prot)) = self.protection_registry.read().unwrap().default_backend() {
            if let Err(e) = prot.configure_region(region, RegionPermissions::code()) {
                warn!("Protection HAL configure_region failed: {:?}", e);
            } else if let Err(e) = prot.enable_protection() {
                warn!("Protection HAL enable_protection failed: {:?}", e);
            }
        }
        Ok(())
    }

    /// Container self-bootstrap: when container has embedded CLFMM, it allocates and we copy code.
    /// Returns (ptr, block) on success; orchestrator pushes block and applies protection.
    fn try_container_self_bootstrap(
        &self,
        container: &CoelanoxContainer,
        code_size: usize,
        aligned_size: usize,
        page_size: usize,
    ) -> Option<(*mut u8, ExecutableMemoryBlock)> {
        let backend = ContainerClfmmBackend::from_container(container).ok().flatten()?;
        let backend = std::sync::Arc::new(backend);
        let opts = RegionOptions {
            size: aligned_size,
            alignment: page_size,
            tier: MemoryTier::Fast,
            executable: true,
        };
        let region = backend.allocate_region(aligned_size, page_size, &opts).ok()?;
        let ptr: *mut u8 = region.as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(
                container.optimized_code.as_ptr(),
                ptr,
                code_size,
            );
        }
        let block = ExecutableMemoryBlock::new_clfmm(region, backend);
        Some((ptr, block))
    }

    /// Allocate executable memory and copy machine code.
    /// Container self-manages when it has CLFMM (bootstrap); else orchestrator uses HAL or legacy.
    fn allocate_and_copy_code(&self, container: &CoelanoxContainer) -> CoelanoxResult<*mut u8> {
        let code_size = container.optimized_code.len();
        let page_size = 4096;
        let aligned_size = ((code_size + page_size - 1) / page_size) * page_size;

        // 1. Container self-bootstrap via embedded CLFMM (container manages its own memory)
        if let Some((ptr, block)) = self.try_container_self_bootstrap(container, code_size, aligned_size, page_size) {
            let size = block.as_region_ref().size();
            let mut executable_memory = self.executable_memory.lock().unwrap();
            executable_memory.push(block);
            let mut metrics = self.metrics.lock().unwrap();
            metrics.update_memory_allocation(size);
            metrics.executable_blocks_count = executable_memory.len();
            drop(executable_memory);
            debug!("✅ Allocated executable memory via container CLFMM (self-bootstrap): {} bytes", size);
            self.apply_protection(container, ptr)?;
            return Ok(ptr);
        }

        // 2. Fallback: Memory HAL when an executable backend is registered
        let registry = self.memory_registry.read().unwrap();
        if let Some((target, backend)) = registry.default_for_executable() {
            let opts = RegionOptions {
                size: aligned_size,
                alignment: page_size,
                tier: MemoryTier::Fast,
                executable: true,
            };
            // Allocate while holding registry (backend is borrowed from it)
            match backend.allocate_region(aligned_size, page_size, &opts) {
                Ok(region) => {
                    let ptr = region.as_ptr();
                    let size = region.size();
                    drop(registry);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            container.optimized_code.as_ptr(),
                            ptr,
                            code_size,
                        );
                    }
                    let mut executable_memory = self.executable_memory.lock().unwrap();
                    executable_memory.push(ExecutableMemoryBlock::new_hal(target, region));
                    let mut metrics = self.metrics.lock().unwrap();
                    metrics.update_memory_allocation(size);
                    metrics.executable_blocks_count = executable_memory.len();
                    debug!("✅ Allocated executable memory via HAL: {} bytes", size);
                    drop(executable_memory);
                    self.apply_protection(container, ptr)?;
                    return Ok(ptr);
                }
                Err(e) => {
                    error!("❌ HAL executable allocation failed: {:?}", e);
                    return Err(e);
                }
            }
        }
        drop(registry);

        // 3. Legacy path: orchestrator allocates via mmap/VirtualAlloc when no CLFMM and no HAL
        match self.allocate_executable_memory(aligned_size) {
            Ok((ptr, size)) => {
                debug!("✅ Successfully allocated executable memory (legacy): {} bytes", size);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        container.optimized_code.as_ptr(),
                        ptr,
                        code_size,
                    );
                }
                let mut executable_memory = self.executable_memory.lock().unwrap();
                executable_memory.push(ExecutableMemoryBlock::new_legacy(ptr, size));
                let mut metrics = self.metrics.lock().unwrap();
                metrics.update_memory_allocation(size);
                metrics.executable_blocks_count = executable_memory.len();
                debug!("✅ Created function pointer: 0x{:x}", ptr as usize);
                drop(executable_memory);
                self.apply_protection(container, ptr)?;
                Ok(ptr)
            }
            Err(e) => {
                error!("❌ Failed to allocate executable memory: {:?}", e);
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
    
    /// Execute with backend selected by the backend manager (hardware flag from CLI).
    ///
    /// The backend manager chooses the execution path based on the backend flag.
    /// No hardcoded primary - the flag determines what we try first. Scalar is
    /// always the fallback when the selected backend fails.
    pub fn execute_with_backend(
        &self,
        container_name: &str,
        input_data: &[f32],
        backend: &str,
    ) -> CoelanoxResult<Vec<f32>> {
        let container = self.get_container_info(container_name)?;
        let start = std::time::Instant::now();
        let result = self.run_backend_execution(container.as_ref(), input_data, backend);
        let elapsed_us = start.elapsed().as_micros() as u64;
        if let Ok(_) = &result {
            let mut stats = self.stats.lock().unwrap();
            stats.update_execution(elapsed_us, 0);
        }
        result
    }

    /// Core backend-driven execution. Used by execute_with_backend and execute_execution_stage.
    fn run_backend_execution(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        backend: &str,
    ) -> CoelanoxResult<Vec<f32>> {
        // Avoid allocating for common backends; only normalize when calling registry.
        if backend.eq_ignore_ascii_case("cpu") {
            return self.execute_machine_code(container, input_data).or_else(|e| {
                warn!("CPU machine code failed: {:?}, falling back to scalar", e);
                self.try_scalar_fallback(container, input_data)
            });
        }
        if backend.eq_ignore_ascii_case("scalar") {
            return self.try_scalar_fallback(container, input_data);
        }
        if backend.eq_ignore_ascii_case("auto") {
            let detected = self.select_backend_auto();
            debug!("Auto-detected backend: {}", detected);
            return self.run_backend_execution(container, input_data, &detected);
        }
        let backend_normalized = backend.to_lowercase();
        if backend.eq_ignore_ascii_case("gpu")
            || backend.eq_ignore_ascii_case("npu")
            || backend.eq_ignore_ascii_case("tpu")
            || backend.eq_ignore_ascii_case("edge")
        {
            return self.try_registry_backend_then_scalar(container, input_data, &backend_normalized);
        }
        warn!("Unknown backend '{}', trying as registered backend then scalar", backend);
        self.try_registry_backend_then_scalar(container, input_data, &backend_normalized)
    }

    /// Try backend from registry, then scalar fallback.
    /// When backend is "gpu"/"npu"/"tpu"/"edge", dynamically resolve via backend manager
    /// to a discovered backend name (e.g. "resnet_tiny_mnist.spv") and try that in the registry too,
    /// so a GPU executor registered as "gpu" or under the CLF name can be selected.
    fn try_registry_backend_then_scalar(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        backend: &str,
    ) -> CoelanoxResult<Vec<f32>> {
        // Try literal backend name first (e.g. "gpu")
        let result = {
            let registry = self.runtime_backend_registry.read().unwrap();
            registry.get(backend).map(|b| b.execute(container, input_data))
        };
        if let Some(Ok(out)) = result {
            return Ok(out);
        }
        if let Some(Err(e)) = result {
            warn!("Backend '{}' failed: {:?}, falling back to scalar", backend, e);
        } else {
            // Not registered under literal name: resolve via backend manager for target-based names
            let target_opt = match backend {
                "gpu" => Some(HardwareTarget::Gpu),
                "npu" => Some(HardwareTarget::NPU),
                "tpu" => Some(HardwareTarget::Custom("tpu".to_string())),
                "edge" => Some(HardwareTarget::Edge),
                _ => None,
            };
            if let Some(target) = target_opt {
                let resolved_name = {
                    let mgr = self.backend_manager.read().unwrap();
                    mgr.find_backend_for_target(&target).map(|b| b.name.clone())
                };
                if let Some(name) = resolved_name {
                    let result = {
                        let registry = self.runtime_backend_registry.read().unwrap();
                        registry.get(&name).map(|b| b.execute(container, input_data))
                    };
                    if let Some(Ok(out)) = result {
                        return Ok(out);
                    }
                    if let Some(Err(e)) = result {
                        warn!("Resolved backend '{}' failed: {:?}, falling back to scalar", name, e);
                    } else {
                        debug!("Backend '{}' resolved to '{}' but not registered, using scalar fallback", backend, name);
                    }
                } else {
                    debug!("Backend '{}' not registered; no backend discovered for target {:?}, using scalar fallback", backend, target);
                }
            } else {
                debug!("Backend '{}' not registered, using scalar fallback", backend);
            }
        }
        self.try_scalar_fallback(container, input_data)
    }

    /// Execute using scalar fallback backend.
    fn try_scalar_fallback(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
    ) -> CoelanoxResult<Vec<f32>> {
        let registry = self.runtime_backend_registry.read().unwrap();
        let backend = registry.scalar_fallback().ok_or_else(|| {
            CoelanoxError::Runtime(
                "Scalar fallback backend not registered in runtime backend registry".to_string(),
            )
        })?;
        backend.execute(container, input_data)
    }

    /// Auto-detect backend from hardware (e.g. env.recommended_target).
    fn select_backend_auto(&self) -> String {
        match coelanox_core::utils::SystemEnvironment::detect() {
            Ok(env) => {
                let recommended = env.recommended_target();
                match recommended {
                    coelanox_core::format::HardwareTarget::Cpu => "cpu".to_string(),
                    coelanox_core::format::HardwareTarget::Gpu => "gpu".to_string(),
                    coelanox_core::format::HardwareTarget::Edge => "edge".to_string(),
                    coelanox_core::format::HardwareTarget::NeuralEngine => "npu".to_string(),
                    coelanox_core::format::HardwareTarget::NPU => "npu".to_string(),
                    coelanox_core::format::HardwareTarget::Custom(s) => s.to_lowercase(),
                }
            }
            Err(_) => "cpu".to_string(),
        }
    }
    
    /// Get container information (thread-safe). Returns Arc so callers avoid cloning full container.
    pub fn get_container_info(&self, container_name: &str) -> CoelanoxResult<Arc<CoelanoxContainer>> {
        let containers = self.containers.read().unwrap();
        Ok(Arc::clone(&containers.get(container_name)
            .ok_or_else(|| CoelanoxError::Runtime(format!("Container not found: {}", container_name)))?.0))
    }

    /// Evict containers that exceed cache TTL (called internally when TTL is set).
    fn evict_expired_containers_inner(
        &self,
        containers: &mut HashMap<String, (Arc<CoelanoxContainer>, Instant)>,
        config: &RuntimeConfig,
    ) {
        let Some(ttl_secs) = config.container_cache_ttl_secs else { return };
        let ttl = Duration::from_secs(ttl_secs);
        let now = Instant::now();
        let expired: Vec<String> = containers
            .iter()
            .filter(|(_, (_, loaded_at))| now.duration_since(*loaded_at) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for name in expired {
            containers.remove(&name);
            info!("Evicted container from cache (TTL): {}", name);
        }
    }

    /// Evict containers that have exceeded the configured TTL. No-op if container_cache_ttl_secs is None.
    pub fn evict_expired_containers(&self) {
        let config = self.get_config();
        if config.container_cache_ttl_secs.is_none() {
            return;
        }
        let mut containers = self.containers.write().unwrap();
        self.evict_expired_containers_inner(&mut containers, &config);
    }

    /// Evict a specific container from the cache by name.
    pub fn evict_container(&self, container_name: &str) -> bool {
        self.containers.write().unwrap().remove(container_name).is_some()
    }
    
    /// List loaded containers (thread-safe)
    pub fn list_containers(&self) -> Vec<String> {
        let containers = self.containers.read().unwrap();
        containers.keys().cloned().collect()
    }
    
    /// Get runtime statistics (thread-safe)
    pub fn get_stats(&self) -> RuntimeStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }
    
    /// Get performance metrics (thread-safe)
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }
    
    /// Get runtime name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get runtime version
    pub fn version(&self) -> &str {
        &self.version
    }
    
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let metrics = self.metrics.lock().unwrap();
        let executable_memory = self.executable_memory.lock().unwrap();
        
        MemoryStats {
            total_allocated: metrics.total_memory_allocated,
            current_usage: metrics.current_memory_usage,
            peak_usage: metrics.peak_memory_usage,
            executable_blocks: executable_memory.len(),
            executable_memory_size: executable_memory.iter().map(|b| b.size()).sum(),
        }
    }
    
    /// Check if runtime is healthy
    pub fn health_check(&self) -> HealthStatus {
        let config = self.get_config();
        let metrics = self.get_metrics();
        let memory_stats = self.get_memory_stats();
        
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        
        // Check memory usage
        if memory_stats.current_usage > config.max_memory_bytes {
            issues.push(format!(
                "Memory usage exceeds limit: {} bytes > {} bytes",
                memory_stats.current_usage,
                config.max_memory_bytes
            ));
        } else if memory_stats.current_usage > config.max_memory_bytes * 80 / 100 {
            warnings.push(format!(
                "Memory usage is high: {} bytes ({}% of limit)",
                memory_stats.current_usage,
                memory_stats.current_usage * 100 / config.max_memory_bytes
            ));
        }
        
        // Check execution failure rate
        if metrics.total_executions > 0 {
            let failure_rate = metrics.failed_executions as f64 / metrics.total_executions as f64;
            if failure_rate > 0.1 {
                issues.push(format!(
                    "High failure rate: {:.1}% ({} failures out of {} executions)",
                    failure_rate * 100.0,
                    metrics.failed_executions,
                    metrics.total_executions
                ));
            } else if failure_rate > 0.05 {
                warnings.push(format!(
                    "Elevated failure rate: {:.1}% ({} failures out of {} executions)",
                    failure_rate * 100.0,
                    metrics.failed_executions,
                    metrics.total_executions
                ));
            }
        }
        
        // Check executable memory blocks
        if memory_stats.executable_blocks > 100 {
            warnings.push(format!(
                "High number of executable memory blocks: {}",
                memory_stats.executable_blocks
            ));
        }
        
        HealthStatus {
            is_healthy: issues.is_empty(),
            issues,
            warnings,
            uptime_ms: metrics.last_execution_time
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(0),
        }
    }
    
    /// Cleanup executable memory with proper thread safety. Handles both legacy
    /// (mmap/VirtualAlloc) and HAL-backed blocks.
    pub fn cleanup(&self) {
        let mut executable_memory = self.executable_memory.lock().unwrap();
        let blocks = std::mem::take(&mut *executable_memory);
        let block_count = blocks.len();
        let mut total_freed = 0usize;

        for block in blocks {
            match block {
                ExecutableMemoryBlock::Legacy { region, .. } => {
                    let ptr = region.as_ptr();
                    let size = region.size();
                    #[cfg(target_os = "linux")]
                    unsafe {
                        use libc::munmap;
                        munmap(ptr as *mut std::ffi::c_void, size);
                    }
                    #[cfg(target_os = "windows")]
                    unsafe {
                        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
                        VirtualFree(ptr as *mut std::ffi::c_void, 0, MEM_RELEASE);
                    }
                    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                    {
                        let _ = ptr;
                        // Generic: no platform free
                    }
                    total_freed += size;
                }
                ExecutableMemoryBlock::Hal { target, region, .. } => {
                    let size = region.size();
                    if let Some(backend) = self.memory_registry.read().unwrap().get(&target) {
                        let _ = backend.deallocate_region(region);
                    }
                    total_freed += size;
                }
                ExecutableMemoryBlock::Clfmm { region, backend, .. } => {
                    let size = region.size();
                    if let Err(e) = backend.deallocate_region(region) {
                        warn!("Container CLFMM deallocate failed: {:?}", e);
                    }
                    total_freed += size;
                }
            }
        }

        let mut metrics = self.metrics.lock().unwrap();
        metrics.update_memory_deallocation(total_freed);
        metrics.executable_blocks_count = 0;

        info!(
            "🧹 Cleaned up {} executable memory blocks ({} bytes)",
            block_count, total_freed
        );
    }
    
    /// Benchmark a container with full latency, memory, and optional long-run stability metrics.
    /// `load_time_ms` can be set by the caller (e.g. CLI) after measuring load_container() duration.
    pub fn benchmark(
        &mut self,
        container_name: &str,
        iterations: usize,
        load_time_ms: Option<u64>,
    ) -> CoelanoxResult<BenchmarkResult> {
        const WARM_UP_RUNS: usize = 10;
        let container = self.get_container_info(container_name)?;

        // Workspace size from plan (if CLF mini-executor path)
        let workspace_elements: u64 = if let Some(ref plan) = container.execution_plan {
            parse_plan(plan)
                .map(|(elements, _, _)| elements as u64)
                .unwrap_or(0)
        } else {
            0
        };

        // Time single decompress and get weights size
        let decompress_start = std::time::Instant::now();
        let weights = container.decompress_weights()?;
        let decompress_time_ms = decompress_start.elapsed().as_millis() as u64;
        let weights_bytes = weights.len() * std::mem::size_of::<f32>();

        // Backend label
        let backend_used = if container.execution_plan.is_some() {
            "plan"
        } else {
            "legacy"
        }
        .to_string();

        // Generate dummy input data
        let input_size: usize = container
            .manifest
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        let input_data = vec![1.0f32; input_size];

        // Warm up
        for _ in 0..WARM_UP_RUNS {
            self.execute(container_name, &input_data)?;
        }

        // Timed iterations: record every latency for percentiles and stability
        let mut times: Vec<u64> = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            self.execute(container_name, &input_data)?;
            times.push(start.elapsed().as_micros() as u64);
        }

        let total_time: u64 = times.iter().sum();
        let avg_time = total_time / iterations as u64;
        let min_time = *times.iter().min().unwrap_or(&0);
        let max_time = *times.iter().max().unwrap_or(&0);

        // Percentiles (sort copy and index)
        let mut sorted = times.clone();
        sorted.sort_unstable();
        let p50_us = percentile(&sorted, 50.0);
        let p95_us = percentile(&sorted, 95.0);
        let p99_us = percentile(&sorted, 99.0);

        // Standard deviation
        let mean = total_time as f64 / iterations as f64;
        let variance = times
            .iter()
            .map(|&t| {
                let d = t as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / iterations as f64;
        let std_dev_us = variance.sqrt();

        // Cold (first run) vs warm (rest)
        let cold_time_us = times.first().copied();
        let warm_avg_us = if times.len() > 1 {
            Some(times[1..].iter().sum::<u64>() / (times.len() - 1) as u64)
        } else {
            None
        };

        // Long-run stability: first 10% vs last 10% when we have enough samples
        const STABILITY_MIN_ITERATIONS: usize = 1000;
        const STABILITY_WINDOW_PCT: f64 = 10.0;
        let (stability_first_10pct_avg_us, stability_last_10pct_avg_us, stability_drift_us) =
            if iterations >= STABILITY_MIN_ITERATIONS {
                let n = times.len();
                let window = (n as f64 * STABILITY_WINDOW_PCT / 100.0).floor() as usize;
                let window = window.max(1);
                let first_avg = times[..window].iter().sum::<u64>() / window as u64;
                let last_avg = times[n - window..].iter().sum::<u64>() / window as u64;
                let drift = last_avg as i64 - first_avg as i64;
                (Some(first_avg), Some(last_avg), Some(drift))
            } else {
                (None, None, None)
            };

        // Memory after benchmark
        let mem = self.get_memory_stats();
        let rss_after_bytes = current_rss_bytes();

        Ok(BenchmarkResult {
            container_name: container_name.to_string(),
            iterations,
            warm_up_runs: WARM_UP_RUNS,
            input_elements: input_size,
            avg_time,
            min_time,
            max_time,
            total_time,
            p50_us,
            p95_us,
            p99_us,
            std_dev_us,
            cold_time_us,
            warm_avg_us,
            load_time_ms,
            decompress_time_ms,
            workspace_elements,
            weights_bytes,
            executable_memory_bytes: mem.executable_memory_size,
            rss_after_bytes,
            backend_used,
            stability_first_10pct_avg_us,
            stability_last_10pct_avg_us,
            stability_drift_us,
        })
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory allocated (bytes)
    pub total_allocated: usize,
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of executable memory blocks
    pub executable_blocks: usize,
    /// Total executable memory size (bytes)
    pub executable_memory_size: usize,
}

/// Health status of the runtime
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Whether the runtime is healthy
    pub is_healthy: bool,
    /// Critical issues that need attention
    pub issues: Vec<String>,
    /// Warnings that should be monitored
    pub warnings: Vec<String>,
    /// Runtime uptime in milliseconds
    pub uptime_ms: u64,
}

/// Benchmark result with full latency, memory, and optional stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub container_name: String,
    pub iterations: usize,
    pub warm_up_runs: usize,
    /// Input size used (elements); benchmark uses synthetic vec![1.0f32; input_elements], same as run without --input
    pub input_elements: usize,
    /// Average execution time (μs)
    pub avg_time: u64,
    pub min_time: u64,
    pub max_time: u64,
    pub total_time: u64,
    /// Latency percentiles (μs)
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    /// Standard deviation of latency (μs)
    pub std_dev_us: f64,
    /// First (cold) run time (μs); includes one-time decompress/code path
    pub cold_time_us: Option<u64>,
    /// Average of runs after the first (warm) (μs)
    pub warm_avg_us: Option<u64>,
    /// Time to load container until first execute-ready (ms); set by CLI when measured
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_time_ms: Option<u64>,
    /// Time to decompress weights once (ms)
    pub decompress_time_ms: u64,
    /// Workspace size from plan (elements)
    pub workspace_elements: u64,
    /// Decompressed weights size (bytes)
    pub weights_bytes: usize,
    /// Executable memory size (bytes) after benchmark
    pub executable_memory_bytes: usize,
    /// Process RSS after benchmark (bytes); None if unavailable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_after_bytes: Option<usize>,
    /// Backend used: "plan" (CLF mini-executor) or "legacy"
    pub backend_used: String,
    /// Long-run stability: average latency in first 10% of iterations (μs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability_first_10pct_avg_us: Option<u64>,
    /// Long-run stability: average latency in last 10% of iterations (μs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability_last_10pct_avg_us: Option<u64>,
    /// Long-run stability: drift (last_10%_avg - first_10%_avg) in μs; positive = slowdown over time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability_drift_us: Option<i64>,
}

impl Drop for CoelanoxRuntime {
    fn drop(&mut self) {
        // Note: We can't call self.cleanup() here because it requires &self, not &mut self
        // The cleanup will be handled by the Arc<Mutex<>> drop implementations
        info!("🧹 COELANOX Runtime dropped - cleanup handled by Arc<Mutex<>>");
    }
}

impl Default for CoelanoxRuntime {
    fn default() -> Self {
        Self::new()
    }
}
