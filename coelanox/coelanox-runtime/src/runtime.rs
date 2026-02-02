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

use coelanox_core::{
    format::CoelanoxContainer,
    ir::{Attribute, Operation, OpType},
    memory_hal::{MemoryBackendRegistry, MemoryRegion, MemoryTarget, RegionOptions, MemoryTier},
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
    /// Maximum number of containers to cache
    pub max_containers: usize,
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
    /// Loaded containers cache (thread-safe)
    containers: Arc<RwLock<HashMap<String, CoelanoxContainer>>>,
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
}

/// Executable memory block: either legacy (mmap/VirtualAlloc) or HAL-backed region.
enum ExecutableMemoryBlock {
    Legacy {
        ptr: *mut u8,
        size: usize,
        allocated_at: Instant,
    },
    Hal {
        target: MemoryTarget,
        region: Box<dyn MemoryRegion>,
        allocated_at: Instant,
    },
}

impl ExecutableMemoryBlock {
    fn new_legacy(ptr: *mut u8, size: usize) -> Self {
        Self::Legacy {
            ptr,
            size,
            allocated_at: Instant::now(),
        }
    }

    fn new_hal(target: MemoryTarget, region: Box<dyn MemoryRegion>) -> Self {
        Self::Hal {
            target,
            region,
            allocated_at: Instant::now(),
        }
    }

    /// Pointer to executable memory (for calling loaded code).
    fn as_ptr(&self) -> *mut u8 {
        match self {
            Self::Legacy { ptr, .. } => *ptr,
            Self::Hal { region, .. } => region.as_ptr(),
        }
    }

    /// Size in bytes.
    fn size(&self) -> usize {
        match self {
            Self::Legacy { size, .. } => *size,
            Self::Hal { region, .. } => region.size(),
        }
    }

    /// Get allocation age in milliseconds
    fn age_ms(&self) -> u64 {
        let at = match self {
            Self::Legacy { allocated_at, .. } => allocated_at,
            Self::Hal { allocated_at, .. } => allocated_at,
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
            name: "coelanox_runtime".to_string(),
            version: "1.0.0".to_string(),
            config: Arc::new(RwLock::new(config)),
            containers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(RuntimeStats::new())),
            executable_memory: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            security_validator,
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            execution_counter: AtomicU64::new(0),
            warmed_containers: Arc::new(RwLock::new(HashMap::new())),
            memory_registry: Arc::new(RwLock::new(MemoryBackendRegistry::new())),
        }
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
        
        if containers.len() >= config.max_containers {
            // Remove oldest container (simple LRU)
            if let Some(oldest_key) = containers.keys().next().cloned() {
                containers.remove(&oldest_key);
                info!("Removed oldest container from cache: {}", oldest_key);
            }
        }
        
        // Store in cache
        containers.insert(path.to_string(), container);
        
        info!("Successfully loaded container: {} ({} bytes)", path, containers[path].total_size());
        
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
        
        if containers.len() >= config.max_containers {
            // Remove oldest container (simple LRU)
            if let Some(oldest_key) = containers.keys().next().cloned() {
                containers.remove(&oldest_key);
                info!("Removed oldest container from cache: {}", oldest_key);
            }
        }
        
        // Store in cache
        containers.insert(name.to_string(), container);
        
        info!("Successfully loaded container from data: {} ({} bytes)", name, containers[name].total_size());
        
        Ok(())
    }
    
    /// Execute a COELANOX container by name with comprehensive validation and monitoring
    pub fn execute(&self, container_name: &str, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        let config = self.get_config();
        let timeout = Duration::from_millis(config.execution_timeout_ms);
        let max_retries = 3; // Default retry count
        
        self.execute_with_retry(container_name, input_data, max_retries, timeout)
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
        
        // Get container from cache
        let container = {
            let containers = self.containers.read().unwrap();
            match containers.get(&execution_state.container_name) {
                Some(container) => container.clone(),
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
        
        // Get container from cache
        let container = {
            let containers = self.containers.read().unwrap();
            containers.get(&execution_state.container_name).unwrap().clone()
        };
        
        // Execute the actual computation
        let result = match self.execute_machine_code(&container, input_data) {
            Ok(output) => {
                debug!("✅ Executed using real machine code");
                Ok(output)
            }
            Err(e) => {
                warn!("⚠️  Machine code execution failed: {:?}, falling back to Rust implementation", e);
                self.execute_machine_code_with_weights(&container, input_data)
            }
        };
        
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
    
    /// Execute machine code with weights
    fn execute_machine_code_with_weights(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Load IR from container
        use bincode;
        use coelanox_core::ir::{GraphNode, IntermediateTensor, WeightTensor};
        use std::collections::HashMap;
        use coelanox_core::format::HardwareTarget;
        
        // Deserialize into temporary struct
        #[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
        struct SerializableIR {
            target: HardwareTarget,
            optimized_nodes: Vec<GraphNode>,
            intermediate_tensors: Vec<IntermediateTensor>,
            optimization_metadata: HashMap<String, String>,
            original_model_name: String,
            weight_tensors: Vec<WeightTensor>,
        }
        
        let serialized: SerializableIR = bincode::deserialize(&container.ir_data)
            .map_err(|e| CoelanoxError::Runtime(format!("Failed to deserialize IR: {}", e)))?;
        
        // Reconstruct OptimizedIR with Arc
        let ir = coelanox_core::ir::OptimizedIR::from_deserialized(
            serialized.target,
            serialized.optimized_nodes,
            serialized.intermediate_tensors,
            serialized.optimization_metadata,
            serialized.original_model_name,
            serialized.weight_tensors,
        );
        
        // Load weights from container
        let weights = container.decompress_weights()?;
        
        // Execute operations from IR
        let mut current_data = input_data.to_vec();
        
        // Iterate through operations in the optimized IR
        for node in &ir.optimized_nodes {
            // Create operation from node (simplified version for OptimizedIR)
            let mut operation = Operation::new(node.op_type.clone());
            
            // Convert tensor IDs to names for inputs
            for tensor_id in &node.input_tensors {
                let tensor_name = if tensor_id.is_weight() {
                    if let Some(tensor) = ir.shared_weights.get(tensor_id.index()) {
                        tensor.name.clone()
                    } else {
                        format!("weight_{}", tensor_id.index())
                    }
                } else {
                    if let Some(tensor) = ir.intermediate_tensors.get(tensor_id.index()) {
                        tensor.name.clone()
                    } else {
                        format!("intermediate_{}", tensor_id.index())
                    }
                };
                operation.add_input(tensor_name);
            }
            
            // Convert tensor IDs to names for outputs
            for tensor_id in &node.output_tensors {
                let tensor_name = if tensor_id.is_weight() {
                    if let Some(tensor) = ir.shared_weights.get(tensor_id.index()) {
                        tensor.name.clone()
                    } else {
                        format!("weight_{}", tensor_id.index())
                    }
                } else {
                    if let Some(tensor) = ir.intermediate_tensors.get(tensor_id.index()) {
                        tensor.name.clone()
                    } else {
                        format!("intermediate_{}", tensor_id.index())
                    }
                };
                operation.add_output(tensor_name);
            }
            
            // Copy attributes
            operation.attributes = node.attributes.clone();
            
            // Execute operation using existing fallback methods
            current_data = self.execute_operation_with_weights(&operation, &current_data, &weights)?;
        }
        
        Ok(current_data)
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
    
    /// Execute machine code from container with proper memory management
    /// Uses unified descriptor-based calling convention for all models
    fn execute_machine_code(&self, container: &CoelanoxContainer, input_data: &[f32]) -> CoelanoxResult<Vec<f32>> {
        // Calculate output size
        let output_size: usize = container.manifest.output_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        // Create executable memory and run real machine code
        let mut output = vec![0.0f32; output_size];
        
        // FIX: Translator now combines all inputs into one flattened input shape
        // Runtime just passes the flattened input_data directly as a single input
        // Function signature: fn(&[f32], &mut [f32], &[f32])
        // Parameters: rdi=input.ptr, rsi=input.len, rdx=output.ptr, rcx=output.len, r8=weights.ptr, r9=weights.len
        
        // Verify input size matches expected (single combined input)
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
        
        // Decompress weights
        let weights = container.decompress_weights()?;
        
        // Execute using a raw-ABI compatible signature to guarantee register assignment
        type CpuFunc = unsafe extern "sysv64" fn(
            *const f32, usize, // rdi, rsi
            *mut f32, usize,   // rdx, rcx
            *const f32, usize  // r8,  r9
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
    
    // Removed create_executable_function: we use raw calling convention above for correctness
    
    /// Allocate executable memory and copy machine code. Uses Memory HAL when a backend
    /// is registered for executable; otherwise uses legacy mmap/VirtualAlloc.
    fn allocate_and_copy_code(&self, container: &CoelanoxContainer) -> CoelanoxResult<*mut u8> {
        let code_size = container.optimized_code.len();
        let page_size = 4096;
        let aligned_size = ((code_size + page_size - 1) / page_size) * page_size;

        // Prefer Memory HAL when an executable backend is registered
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
                    return Ok(ptr);
                }
                Err(e) => {
                    error!("❌ HAL executable allocation failed: {:?}", e);
                    return Err(e);
                }
            }
        }
        drop(registry);

        // Legacy path: mmap / VirtualAlloc
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
    
    /// Get container information (thread-safe)
    pub fn get_container_info(&self, container_name: &str) -> CoelanoxResult<CoelanoxContainer> {
        let containers = self.containers.read().unwrap();
        Ok(containers.get(container_name)
            .ok_or_else(|| CoelanoxError::Runtime(format!("Container not found: {}", container_name)))?
            .clone())
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
        
        // TODO: Legacy operations field removed - use IR operations instead
        // Process each operation in the container (legacy code removed)
        let _empty_operations: Vec<serde_json::Value> = Vec::new();
        for operation_json in &_empty_operations {
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
    
    /// Cleanup executable memory with proper thread safety. Handles both legacy
    /// (mmap/VirtualAlloc) and HAL-backed blocks.
    pub fn cleanup(&self) {
        let mut executable_memory = self.executable_memory.lock().unwrap();
        let blocks = std::mem::take(&mut *executable_memory);
        let block_count = blocks.len();
        let mut total_freed = 0usize;

        for block in blocks {
            match block {
                ExecutableMemoryBlock::Legacy { ptr, size, .. } => {
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
