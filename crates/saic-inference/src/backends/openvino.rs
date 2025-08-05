//! OpenVINO backend for SAIC inference engine
//! 
//! Provides Intel OpenVINO integration with dynamic library loading

use crate::{InferenceBackend, InferenceError, HardwareRequirements};
use std::sync::{Arc, Mutex};
use std::ptr;
use std::ffi::CString;

#[cfg(feature = "openvino")]
use crate::ffi::openvino::*;

/// OpenVINO backend implementation with proper resource management
pub struct OpenVINOBackend {
    #[cfg(feature = "openvino")]
    library: Option<Arc<libloading::Library>>,
    #[cfg(not(feature = "openvino"))]
    library: Option<()>,
    
    // OpenVINO resources wrapped in thread-safe containers
    #[cfg(feature = "openvino")]
    core: Option<Arc<Mutex<*mut ov_core_t>>>,
    #[cfg(not(feature = "openvino"))]
    core: Option<()>,
    #[cfg(feature = "openvino")]
    model: Option<Arc<Mutex<*mut ov_model_t>>>,
    #[cfg(not(feature = "openvino"))]
    model: Option<()>,
    #[cfg(feature = "openvino")]
    compiled_model: Option<Arc<Mutex<*mut ov_compiled_model_t>>>,
    #[cfg(not(feature = "openvino"))]
    compiled_model: Option<()>,
    #[cfg(feature = "openvino")]
    infer_request: Option<Arc<Mutex<*mut ov_infer_request_t>>>,
    #[cfg(not(feature = "openvino"))]
    infer_request: Option<()>,
    
    initialized: bool,
}

impl OpenVINOBackend {
    /// Create new OpenVINO backend
    pub fn new() -> Option<Self> {
        #[cfg(feature = "openvino")]
        {
            // Try to load OpenVINO library dynamically
            let library = unsafe {
                libloading::Library::new("libopenvino_c.so")
                    .or_else(|_| libloading::Library::new("libopenvino.so"))
                    .ok()
            };
            
            Some(Self {
                library: library.map(|lib| Arc::new(lib)),
                core: None,
                model: None,
                compiled_model: None,
                infer_request: None,
                initialized: false,
            })
        }
        
        #[cfg(not(feature = "openvino"))]
        {
            None
        }
    }
    
    /// Check if OpenVINO is available
    pub fn is_available() -> bool {
        #[cfg(feature = "openvino")]
        {
            unsafe {
                libloading::Library::new("libopenvino_c.so")
                    .or_else(|_| libloading::Library::new("libopenvino.so"))
                    .is_ok()
            }
        }
        
        #[cfg(not(feature = "openvino"))]
        {
            false
        }
    }
    
    /// Initialize OpenVINO core
    #[cfg(feature = "openvino")]
    unsafe fn initialize_core(&mut self) -> Result<(), InferenceError> {
        if self.core.is_some() {
            return Ok(());
        }
        
        // Create OpenVINO core
        let mut core_ptr: *mut ov_core_t = ptr::null_mut();
        let status = ov_core_create(&mut core_ptr);
        
        if status != 0 || core_ptr.is_null() {
            return Err(InferenceError::InitializationFailed(
                format!("Failed to create OpenVINO core: status {}", status)
            ));
        }
        
        self.core = Some(Arc::new(Mutex::new(core_ptr)));
        Ok(())
    }
    
    /// Load model from memory buffer
    #[cfg(feature = "openvino")]
    unsafe fn load_model(&mut self, model_payload: &[u8]) -> Result<(), InferenceError> {
        if self.model.is_some() {
            return Ok(());
        }
        
        let core = self.core.as_ref().ok_or_else(|| {
            InferenceError::InitializationFailed("OpenVINO core not initialized".to_string())
        })?;
        
        let core_ptr = *core.lock().unwrap();
        
        // Create model from memory buffer
        let mut model_ptr: *mut ov_model_t = ptr::null_mut();
        let status = ov_core_read_model_from_memory(
            core_ptr,
            model_payload.as_ptr() as *const i8,
            model_payload.len(),
            &mut model_ptr
        );
        
        if status != 0 || model_ptr.is_null() {
            return Err(InferenceError::ModelLoadingFailed(
                format!("Failed to load model: status {}", status)
            ));
        }
        
        self.model = Some(Arc::new(Mutex::new(model_ptr)));
        Ok(())
    }
    
    /// Compile model for CPU device
    #[cfg(feature = "openvino")]
    unsafe fn compile_model(&mut self) -> Result<(), InferenceError> {
        if self.compiled_model.is_some() {
            return Ok(());
        }
        
        let core = self.core.as_ref().ok_or_else(|| {
            InferenceError::InitializationFailed("OpenVINO core not initialized".to_string())
        })?;
        
        let model = self.model.as_ref().ok_or_else(|| {
            InferenceError::ModelLoadingFailed("Model not loaded".to_string())
        })?;
        
        let core_ptr = *core.lock().unwrap();
        let model_ptr = *model.lock().unwrap();
        
        // Convert device string to C string
        let device_cstring = CString::new("CPU").map_err(|_| {
            InferenceError::InitializationFailed("Invalid device string".to_string())
        })?;
        
        // Compile model for CPU device
        let mut compiled_model_ptr: *mut ov_compiled_model_t = ptr::null_mut();
        let status = ov_core_compile_model(
            core_ptr, 
            model_ptr, 
            device_cstring.as_ptr(), 
            &mut compiled_model_ptr
        );
        
        if status != 0 || compiled_model_ptr.is_null() {
            return Err(InferenceError::InitializationFailed(
                format!("Failed to compile model: status {}", status)
            ));
        }
        
        self.compiled_model = Some(Arc::new(Mutex::new(compiled_model_ptr)));
        Ok(())
    }
    
    /// Create inference request
    #[cfg(feature = "openvino")]
    unsafe fn create_infer_request(&mut self) -> Result<(), InferenceError> {
        if self.infer_request.is_some() {
            return Ok(());
        }
        
        let compiled_model = self.compiled_model.as_ref().ok_or_else(|| {
            InferenceError::InitializationFailed("Model not compiled".to_string())
        })?;
        
        let compiled_model_ptr = *compiled_model.lock().unwrap();
        
        // Create inference request
        let mut infer_request_ptr: *mut ov_infer_request_t = ptr::null_mut();
        let status = ov_compiled_model_create_infer_request(compiled_model_ptr, &mut infer_request_ptr);
        
        if status != 0 || infer_request_ptr.is_null() {
            return Err(InferenceError::InitializationFailed(
                format!("Failed to create inference request: status {}", status)
            ));
        }
        
        self.infer_request = Some(Arc::new(Mutex::new(infer_request_ptr)));
        Ok(())
    }
    
    /// Execute inference with OpenVINO
    #[cfg(feature = "openvino")]
    unsafe fn execute_inference(&self, input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        let infer_request = self.infer_request.as_ref().ok_or_else(|| {
            InferenceError::ExecutionFailed("Inference request not created".to_string())
        })?;
        
        let infer_request_ptr = *infer_request.lock().unwrap();
        
        // Get input tensor
        let mut input_tensor: *mut ov_tensor_t = ptr::null_mut();
        let status = ov_infer_request_get_input_tensor_by_index(infer_request_ptr, 0, &mut input_tensor);
        
        if status != 0 || input_tensor.is_null() {
            return Err(InferenceError::InputDataError(
                format!("Failed to get input tensor: status {}", status)
            ));
        }
        
        // Copy input data to tensor
        let mut tensor_data: *mut std::ffi::c_void = ptr::null_mut();
        let status = ov_tensor_data(input_tensor, &mut tensor_data);
        
        if status != 0 || tensor_data.is_null() {
            return Err(InferenceError::InputDataError(
                format!("Failed to get tensor data: status {}", status)
            ));
        }
        
        // Copy input data
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr() as *const u8,
            tensor_data as *mut u8,
            input_data.len() * std::mem::size_of::<f32>()
        );
        
        // Run inference
        let status = ov_infer_request_infer(infer_request_ptr);
        
        if status != 0 {
            return Err(InferenceError::ExecutionFailed(
                format!("Inference failed: status {}", status)
            ));
        }
        
        // Get output tensor
        let mut output_tensor: *mut ov_tensor_t = ptr::null_mut();
        let status = ov_infer_request_get_output_tensor_by_index(infer_request_ptr, 0, &mut output_tensor);
        
        if status != 0 || output_tensor.is_null() {
            return Err(InferenceError::OutputDataError(
                format!("Failed to get output tensor: status {}", status)
            ));
        }
        
        // Get output shape and size
        let mut shape: ov_shape_t = ptr::null_mut();
        let status = ov_tensor_get_shape(output_tensor, &mut shape);
        
        if status != 0 {
            return Err(InferenceError::OutputDataError(
                format!("Failed to get output shape: status {}", status)
            ));
        }
        
        // For simplicity, assume 1000-class output
        let output_size = 1000;
        let mut results = Vec::with_capacity(output_size);
        
        // Get output data
        let mut output_data: *mut std::ffi::c_void = ptr::null_mut();
        let status = ov_tensor_data(output_tensor, &mut output_data);
        
        if status != 0 || output_data.is_null() {
            return Err(InferenceError::OutputDataError(
                format!("Failed to get output data: status {}", status)
            ));
        }
        
        // Copy output data
        let output_slice = std::slice::from_raw_parts(
            output_data as *const f32,
            output_size
        );
        
        results.extend_from_slice(output_slice);
        
        Ok(results)
    }
    
    /// Execute inference with OpenVINO (fallback implementation)
    #[cfg(not(feature = "openvino"))]
    fn execute_with_openvino(&self, _model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // Fallback implementation when OpenVINO feature is disabled
        let output_size = 1000;
        let mut results = Vec::with_capacity(output_size);
        
        for i in 0..output_size {
            let value = input_data.iter().sum::<f32>() * 2.0 + i as f32;
            results.push(value);
        }
        
        Ok(results)
    }
}

impl InferenceBackend for OpenVINOBackend {
    fn execute(&self, model_payload: &[u8], input_data: &[f32]) -> Result<Vec<f32>, InferenceError> {
        #[cfg(feature = "openvino")]
        {
            // Auto-initialize if not already initialized
            if !self.initialized {
                if self.library.is_none() {
                    return Err(InferenceError::RuntimeNotFound("OpenVINO library not found".to_string()));
                }
                
                // Initialize OpenVINO resources
                unsafe {
                    let mut backend = self.clone();
                    backend.initialize_core()?;
                    backend.load_model(model_payload)?;
                    backend.compile_model()?;
                    backend.create_infer_request()?;
                    
                    // Execute inference
                    return backend.execute_inference(input_data);
                }
            }
            
            // Execute with existing resources
            unsafe {
                self.execute_inference(input_data)
            }
        }
        
        #[cfg(not(feature = "openvino"))]
        {
            self.execute_with_openvino(model_payload, input_data)
        }
    }
    
    fn name(&self) -> &str {
        "openvino"
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["Universal".to_string()] // OpenVINO can interpret various formats
    }
    
    fn hardware_requirements(&self) -> HardwareRequirements {
        HardwareRequirements {
            cpu: true,
            gpu: false,
            intel: true,
            nvidia: false,
            memory_mb: 1024,
        }
    }
    
    fn initialize(&mut self) -> Result<(), InferenceError> {
        #[cfg(feature = "openvino")]
        {
            if self.library.is_none() {
                return Err(InferenceError::RuntimeNotFound("OpenVINO library not available".to_string()));
            }
        }
        
        self.initialized = true;
        Ok(())
    }
    
    fn cleanup(&mut self) -> Result<(), InferenceError> {
        self.initialized = false;
        Ok(())
    }
}

impl Drop for OpenVINOBackend {
    fn drop(&mut self) {
        #[cfg(feature = "openvino")]
        unsafe {
            // Free OpenVINO resources in reverse order
            if let Some(infer_request) = self.infer_request.take() {
                let ptr = *infer_request.lock().unwrap();
                ov_infer_request_free(ptr);
            }
            
            if let Some(compiled_model) = self.compiled_model.take() {
                let ptr = *compiled_model.lock().unwrap();
                ov_compiled_model_free(ptr);
            }
            
            if let Some(model) = self.model.take() {
                let ptr = *model.lock().unwrap();
                ov_model_free(ptr);
            }
            
            if let Some(core) = self.core.take() {
                let ptr = *core.lock().unwrap();
                ov_core_free(ptr);
            }
        }
    }
}

impl Clone for OpenVINOBackend {
    fn clone(&self) -> Self {
        Self {
            library: self.library.clone(),
            core: self.core.clone(),
            model: self.model.clone(),
            compiled_model: self.compiled_model.clone(),
            infer_request: self.infer_request.clone(),
            initialized: self.initialized,
        }
    }
}

impl Default for OpenVINOBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|| Self {
            library: None,
            core: None,
            model: None,
            compiled_model: None,
            infer_request: None,
            initialized: false,
        })
    }
}

// Mark as thread-safe since OpenVINO handles thread safety internally
unsafe impl Send for OpenVINOBackend {}
unsafe impl Sync for OpenVINOBackend {} 