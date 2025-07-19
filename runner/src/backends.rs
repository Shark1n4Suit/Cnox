use openvino::{Core, DeviceType};
use tracing::{error, info};

/// Backend-specific error types for model execution
/// 
/// This enum provides specific error types for different backend execution failures,
/// allowing for proper error handling and user feedback during model inference.
#[derive(Debug)]
pub enum BackendError {
    /// OpenVINO initialization failed
    OpenVinoInitFailed(String),
    /// Model loading or parsing failed
    ModelLoadFailed(String),
    /// Model compilation failed
    CompilationFailed(String),
    /// Device not available or incompatible
    DeviceNotAvailable(String),
    /// Invalid model format for backend
    InvalidModelFormat(String),
    /// Generic backend execution error
    ExecutionError(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::OpenVinoInitFailed(msg) => write!(f, "OpenVINO initialization failed: {}", msg),
            BackendError::ModelLoadFailed(msg) => write!(f, "Model loading failed: {}", msg),
            BackendError::CompilationFailed(msg) => write!(f, "Model compilation failed: {}", msg),
            BackendError::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            BackendError::InvalidModelFormat(msg) => write!(f, "Invalid model format: {}", msg),
            BackendError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Trait defining the interface for AI model execution backends
/// 
/// This trait provides a common interface for different execution backends
/// (CPU, GPU, NPU, etc.) to execute AI models from SAIC containers.
pub trait Backend {
    /// Execute a model using the backend's inference engine
    /// 
    /// Takes the raw model payload bytes and attempts to load, compile,
    /// and prepare the model for execution on the backend's target device.
    /// 
    /// # Arguments
    /// 
    /// * `model_payload` - Raw bytes containing the AI model data
    /// * `model_name` - Name of the model for logging and identification
    /// 
    /// # Returns
    /// 
    /// * `Result<(), BackendError>` - Ok(()) if successful, BackendError on failure
    fn execute(&mut self, model_payload: &[u8], model_name: &str) -> Result<(), BackendError>;
    
    /// Get the backend name for identification and logging
    fn name(&self) -> &str;
    
    /// Check if the backend is available on the current system
    fn is_available(&self) -> bool;
}

/// CPU execution backend using Intel OpenVINO
/// 
/// This backend uses the Intel OpenVINO toolkit to execute AI models
/// on CPU devices, providing optimized inference performance.
pub struct CpuBackend {
    /// OpenVINO Core instance for model operations
    core: Option<Core>,
}

impl CpuBackend {
    /// Create a new CPU backend instance
    /// 
    /// Initializes a new CPU backend that will use OpenVINO for model execution.
    /// The OpenVINO Core is initialized lazily on first use to avoid unnecessary
    /// overhead if the backend is not used.
    pub fn new() -> Self {
        Self { core: None }
    }
    
    /// Initialize the OpenVINO Core if not already initialized
    /// 
    /// This method ensures the OpenVINO Core is properly initialized before
    /// attempting any model operations. It's called automatically by execute().
    fn ensure_core(&mut self) -> Result<&mut Core, BackendError> {
        if self.core.is_none() {
            info!("Initializing OpenVINO Core...");
            
            match Core::new() {
                Ok(core) => {
                    info!("OpenVINO Core initialized successfully");
                    self.core = Some(core);
                }
                Err(e) => {
                    let error_msg = format!("Failed to initialize OpenVINO Core: {}", e);
                    error!("{}", error_msg);
                    return Err(BackendError::OpenVinoInitFailed(error_msg));
                }
            }
        }
        
        Ok(self.core.as_mut().unwrap())
    }
}

impl Backend for CpuBackend {
    /// Execute a model on CPU using OpenVINO
    /// 
    /// This method implements the complete model execution pipeline:
    /// 1. Initialize OpenVINO Core
    /// 2. Load model from payload bytes
    /// 3. Compile model for CPU device
    /// 4. Validate successful compilation
    /// 
    /// # Arguments
    /// 
    /// * `model_payload` - Raw bytes containing the OpenVINO IR model
    /// * `model_name` - Name of the model for logging purposes
    /// 
    /// # Returns
    /// 
    /// * `Result<(), BackendError>` - Ok(()) if compilation successful
    fn execute(&mut self, model_payload: &[u8], model_name: &str) -> Result<(), BackendError> {
        info!("Starting CPU execution for model: {}", model_name);
        info!("Model payload size: {} bytes", model_payload.len());
        
        // Validate payload is not empty
        if model_payload.is_empty() {
            let error_msg = "Model payload is empty".to_string();
            error!("{}", error_msg);
            return Err(BackendError::InvalidModelFormat(error_msg));
        }
        
        // Step 1: Ensure OpenVINO Core is initialized
        let core = self.ensure_core()?;
        
        // Step 2: Load model from payload bytes
        info!("Loading model from payload...");
        let model = match core.read_model_from_buffer(model_payload, None) {
            Ok(model) => {
                info!("Model loaded successfully from payload");
                model
            }
            Err(e) => {
                let error_msg = format!("Failed to load model from payload: {}", e);
                error!("{}", error_msg);
                return Err(BackendError::ModelLoadFailed(error_msg));
            }
        };
        
        // Step 3: Compile model for CPU device
        info!("Compiling model for CPU device...");
        let _compiled_model = match core.compile_model(&model, DeviceType::CPU) {
            Ok(compiled_model) => {
                info!("Model compiled successfully for CPU");
                compiled_model
            }
            Err(e) => {
                let error_msg = format!("Failed to compile model for CPU: {}", e);
                error!("{}", error_msg);
                return Err(BackendError::CompilationFailed(error_msg));
            }
        };
        
        // Step 4: Validate compilation success
        info!("Validating compiled model...");
        
        // The successful compilation is validation enough for now
        // In a full implementation, we could create an inference request
        // and check model properties, but for this MVP we'll keep it simple
        info!("Model validation successful: compilation completed without errors");
        
        info!("CPU backend execution completed successfully for model: {}", model_name);
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "CPU (OpenVINO)"
    }
    
    /// Check if CPU backend is available
    /// 
    /// The CPU backend should always be available on systems with OpenVINO installed,
    /// but this method verifies that OpenVINO Core can be initialized.
    fn is_available(&self) -> bool {
        match Core::new() {
            Ok(_) => {
                info!("CPU backend (OpenVINO) is available");
                true
            }
            Err(e) => {
                error!("CPU backend (OpenVINO) is not available: {}", e);
                false
            }
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
} 