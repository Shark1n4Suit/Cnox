//! Plugin system traits and types
//! 
//! This module provides the core traits and types for the plugin system.

use std::path::Path;
use crate::{CoelanoxError, CoelanoxResult, format::ModelFormat};
use crate::ir::UniversalIR;

/// Translator plugin trait
pub trait TranslatorPlugin: Send + Sync + 'static {
    /// Translate model data to Universal IR
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>;
    
    /// Translate model data to Universal IR with model type context
    fn translate_with_context(&self, model_data: &[u8], _model_type: &str, _custom_input_shapes: Option<&str>, _custom_output_shapes: Option<&str>) -> CoelanoxResult<UniversalIR> {
        // Default implementation: ignore model type and use regular translate
        self.translate(model_data)
    }
    
    /// Translate model from file path (supports sharded models)
    fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR> {
        self.translate_from_path_with_weights(input_path, None)
    }

    /// Translate from path with optional external weights file (e.g. ONNX model + separate weight file).
    /// Default: ignore external_weights_path and delegate to translate_from_path.
    fn translate_from_path_with_weights(&self, input_path: &Path, external_weights_path: Option<&Path>) -> CoelanoxResult<UniversalIR> {
        if external_weights_path.is_some() {
            return Err(CoelanoxError::format(
                "External weights file is not supported by this translator",
            ));
        }
        println!("📖 Reading model file: {}...", input_path.display());
        let file_size = std::fs::metadata(input_path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read metadata for {}: {}", input_path.display(), e)))?
            .len();
        println!("📊 File size: {}MB", file_size / (1024 * 1024));
        let data = std::fs::read(input_path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read file {}: {}", input_path.display(), e)))?;
        println!("✅ File read successfully, translating...");
        self.translate(&data)
    }

    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>;

    /// Extract weights when model uses optional external weights file.
    /// Default: when None, delegate to extract_weights(model_data); when Some, error.
    fn extract_weights_with_external(&self, model_data: &[u8], external_weights_path: Option<&Path>) -> CoelanoxResult<Vec<f32>> {
        match external_weights_path {
            None => self.extract_weights(model_data),
            Some(_) => Err(CoelanoxError::format(
                "External weights file is not supported by this translator",
            )),
        }
    }
    
    /// Get supported model formats
    fn supported_formats(&self) -> Vec<ModelFormat>;
    
    /// Get plugin name
    fn name(&self) -> &str;
    
    /// Get plugin version
    fn version(&self) -> &str;
    
    /// Get plugin author
    fn author(&self) -> &str;
    
    /// Get plugin description
    fn description(&self) -> &str {
        "No description available"
    }
    
    /// Configure plugin with custom settings
    fn configure(&mut self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }
    
    /// Validate plugin configuration
    fn validate_config(&self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }
    
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: self.name().to_string(),
            version: self.version().to_string(),
            author: self.author().to_string(),
            description: self.description().to_string(),
            max_model_size: None,
            supported_data_types: vec!["float32".to_string()],
        }
    }
    
    /// Check if plugin supports a specific model format
    fn supports_format(&self, format: ModelFormat) -> bool {
        self.supported_formats().contains(&format)
    }
    
    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            supports_sharded_models: false,
            supports_custom_shapes: false,
            supports_weight_extraction: true,
            supports_model_validation: true,
        }
    }
    
    /// Convert plugin to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized {
        self
    }
}

/// Plugin configuration for customizing plugin behavior
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Enable verbose logging
    pub verbose: bool,
    /// Maximum model size in bytes
    pub max_model_size: Option<usize>,
    /// Custom optimization settings
    pub optimization_settings: std::collections::HashMap<String, String>,
    /// Enable experimental features
    pub experimental_features: bool,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            max_model_size: None,
            optimization_settings: std::collections::HashMap::new(),
            experimental_features: false,
        }
    }
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Maximum model size supported (in bytes)
    pub max_model_size: Option<usize>,
    /// Supported data types
    pub supported_data_types: Vec<String>,
}

/// Plugin capabilities
#[derive(Debug, Clone)]
pub struct PluginCapabilities {
    /// Supports sharded models
    pub supports_sharded_models: bool,
    /// Supports custom input/output shapes
    pub supports_custom_shapes: bool,
    /// Supports weight extraction
    pub supports_weight_extraction: bool,
    /// Supports model validation
    pub supports_model_validation: bool,
}
