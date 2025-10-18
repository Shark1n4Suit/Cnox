//! Plugin system for translator plugins
//! 
//! This module provides the plugin system for loading and managing translator
//! plugins that convert different model formats to Universal IR.

use std::collections::HashMap;
use std::path::Path;
use crate::{SaicError, SaicResult};
use crate::format::ModelFormat;
use super::ir::UniversalIR;
use prost::Message;

mod clean_onnx;
use clean_onnx::CleanOnnxTranslator;

/// Translator plugin trait
pub trait TranslatorPlugin: Send + Sync + 'static {
    /// Translate model data to Universal IR
    fn translate(&self, model_data: &[u8]) -> SaicResult<UniversalIR>;
    
    /// Translate model data to Universal IR with model type context
    fn translate_with_context(&self, model_data: &[u8], model_type: &str, custom_input_shapes: Option<&str>, custom_output_shapes: Option<&str>) -> SaicResult<UniversalIR> {
        // Default implementation: ignore model type and use regular translate
        self.translate(model_data)
    }
    
    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8]) -> SaicResult<Vec<f32>>;
    
    /// Get supported model formats
    fn supported_formats(&self) -> Vec<ModelFormat>;
    
    /// Get plugin name
    fn name(&self) -> &str;
    
    /// Get plugin version
    fn version(&self) -> &str;
    
    /// Get plugin author
    fn author(&self) -> &str;
    
    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized;
}

/// Plugin manager for loading and managing translator plugins
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn TranslatorPlugin>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }
    
    /// Register a plugin
    pub fn register_plugin(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>) {
        self.plugins.insert(name, plugin);
    }
    
    /// Load default plugins
    pub fn load_default_plugins(&mut self) -> SaicResult<()> {
        // Register ONNX plugin
        let onnx_plugin = Box::new(OnnxPlugin::new());
        self.plugins.insert("onnx".to_string(), onnx_plugin);
        
        println!("✅ Loaded default plugins: ONNX");
        Ok(())
    }
    
    /// Find a plugin for the given model format
    pub fn find_plugin(&self, format: ModelFormat) -> Option<&dyn TranslatorPlugin> {
        match format {
            ModelFormat::Onnx => self.plugins.get("onnx").map(|p| p.as_ref()),
            _ => None,
        }
    }
    
    /// Load a plugin from file
    pub fn load_plugin(&mut self, plugin_path: &Path) -> SaicResult<()> {
        // For now, just return Ok - plugin loading from files not implemented
        println!("⚠️  Plugin loading from file not implemented: {:?}", plugin_path);
        Ok(())
    }
}
/// ONNX plugin - now using clean implementation
pub struct OnnxPlugin {
    /// Clean ONNX translator
    translator: CleanOnnxTranslator,
}

impl OnnxPlugin {
    /// Create a new ONNX plugin
    pub fn new() -> Self {
        Self {
            translator: CleanOnnxTranslator::new(),
        }
    }
}

impl TranslatorPlugin for OnnxPlugin {
    /// Translate ONNX model to Universal IR using clean implementation
    fn translate(&self, model_data: &[u8]) -> SaicResult<UniversalIR> {
        println!("🎯 Using clean ONNX translator");
        self.translator.translate(model_data)
    }
    
    /// Translate with context - delegates to clean translator
    fn translate_with_context(&self, model_data: &[u8], _model_type: &str, _custom_input_shapes: Option<&str>, _custom_output_shapes: Option<&str>) -> SaicResult<UniversalIR> {
        println!("🎯 Using clean ONNX translator with context");
        self.translator.translate(model_data)
    }
    
    /// Extract weights using clean implementation
    fn extract_weights(&self, model_data: &[u8]) -> SaicResult<Vec<f32>> {
        println!("🔍 Extracting weights using clean ONNX translator");
        
        // Parse ONNX model
        let model = crate::onnx::ModelProto::decode(model_data)
            .map_err(|e| crate::SaicError::format(&format!("ONNX protobuf parsing failed: {}", e)))?;
        
        let mut weights = Vec::new();
        
        // Extract weights from initializers
        if let Some(graph) = &model.graph {
            for initializer in &graph.initializer {
                if initializer.data_type == 1 { // Float32
                    if !initializer.float_data.is_empty() {
                        weights.extend(&initializer.float_data);
                    } else if !initializer.raw_data.is_empty() {
                        // Convert raw bytes to float32
                        let raw_data = &initializer.raw_data;
                        for chunk in raw_data.chunks(4) {
                            if chunk.len() == 4 {
                                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let value = f32::from_le_bytes(bytes);
                                if value.is_finite() {
                weights.push(value);
            }
                            }
                        }
                    }
                }
            }
        }
        
        println!("✅ Extracted {} weights using clean method", weights.len());
        Ok(weights)
    }
    
    /// Get supported formats
    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Onnx]
    }
    
    /// Get plugin name
    fn name(&self) -> &str {
        "Clean ONNX Plugin"
    }
    
    /// Get plugin version
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    /// Get plugin author
    fn author(&self) -> &str {
        "SAIC Team"
    }
    
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized {
        self
    }
}
