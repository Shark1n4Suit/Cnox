//! Plugin system for translator plugins
//! 
//! This module provides the plugin system for loading and managing translator
//! plugins that convert different model formats to Universal IR.

use std::collections::HashMap;
use std::path::Path;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use super::ir::UniversalIR;
use prost::Message;

pub mod onnx;
pub mod onnx_translator;
pub mod onnx_helpers;

#[cfg(test)]
mod onnx_translator_tests;

/// Plugin configuration for customizing plugin behavior
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Enable verbose logging
    pub verbose: bool,
    /// Custom optimization level
    pub optimization_level: Option<String>,
    /// Custom data type preferences
    pub data_type_preferences: HashMap<String, String>,
    /// Additional custom parameters
    pub custom_params: HashMap<String, String>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            optimization_level: None,
            data_type_preferences: HashMap::new(),
            custom_params: HashMap::new(),
        }
    }
}

/// Plugin capabilities information
#[derive(Debug, Clone)]
pub struct PluginCapabilities {
    /// Supported model formats
    pub supported_formats: Vec<ModelFormat>,
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
        // Default implementation: read file and use regular translate
        let data = std::fs::read(input_path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read file {}: {}", input_path.display(), e)))?;
        self.translate(&data)
    }
    
    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>;
    
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
    fn configure(&mut self, _config: PluginConfig) -> CoelanoxResult<()> {
        // Default implementation: ignore configuration
        Ok(())
    }
    
    /// Get plugin capabilities
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            supported_formats: self.supported_formats(),
            version: self.version().to_string(),
            author: self.author().to_string(),
            description: self.description().to_string(),
            max_model_size: None,
            supported_data_types: vec!["Float32".to_string()],
        }
    }
    
    /// Validate plugin with test data
    fn validate(&self) -> CoelanoxResult<()> {
        // Default implementation: test with dummy data
        let dummy_data = vec![0u8; 100];
        let _ = self.translate(&dummy_data)?;
        Ok(())
    }
    
    /// Convert to Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized;
}

/// Plugin manager for loading and managing translator plugins
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn TranslatorPlugin>>,
    plugin_configs: HashMap<String, PluginConfig>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            plugin_configs: HashMap::new(),
        }
    }
    
    /// Register a plugin
    pub fn register_plugin(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>) {
        self.plugins.insert(name, plugin);
    }
    
    /// Register a plugin with validation
    pub fn register_plugin_with_validation(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>) -> CoelanoxResult<()> {
        // Test plugin with valid dummy data
        let dummy_data = onnx_helpers::build_min_model_with_graph(vec![], vec![]).encode_to_vec();
        let _ = plugin.translate(&dummy_data)?;
        
        // Register if validation passes
        self.plugins.insert(name, plugin);
        Ok(())
    }
    
    /// Load default plugins
    pub fn load_default_plugins(&mut self) -> CoelanoxResult<()> {
        // Register ONNX translator
        let onnx_translator = Box::new(onnx_translator::OnnxTranslator::new());
        self.register_plugin_with_validation("onnx_translator".to_string(), onnx_translator)?;
        
        println!("✅ Plugin manager loaded with ONNX translator");
        Ok(())
    }
    
    /// Find a plugin for the given model format
    pub fn find_plugin(&self, format: ModelFormat) -> Option<&dyn TranslatorPlugin> {
        // Search through all plugins to find one that supports this format
        for plugin in self.plugins.values() {
            if plugin.supported_formats().contains(&format) {
                return Some(plugin.as_ref());
            }
        }
        None
    }
    
    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&dyn TranslatorPlugin> {
        self.plugins.get(name).map(|p| p.as_ref())
    }
    
    /// Translate model from file path (supports sharded models)
    pub fn translate_from_path(&self, input_path: &Path, format: ModelFormat) -> CoelanoxResult<UniversalIR> {
        if let Some(plugin) = self.find_plugin(format.clone()) {
            plugin.translate_from_path(input_path)
        } else {
            Err(CoelanoxError::format(&format!("No plugin found for format: {:?}", format)))
        }
    }
    
    /// Get plugin by name (mutable)
    pub fn get_plugin_mut(&mut self, name: &str) -> Option<&mut dyn TranslatorPlugin> {
        self.plugins.get_mut(name).map(|p| p.as_mut())
    }
    
    /// Remove a plugin
    pub fn remove_plugin(&mut self, name: &str) -> Option<Box<dyn TranslatorPlugin>> {
        self.plugin_configs.remove(name);
        self.plugins.remove(name)
    }
    
    /// Get number of registered plugins
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
    
    /// List all available plugins
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.keys().map(|k| k.as_str()).collect()
    }
    
    /// Get plugin capabilities
    pub fn get_plugin_capabilities(&self, name: &str) -> Option<PluginCapabilities> {
        self.plugins.get(name).map(|plugin| plugin.capabilities())
    }
    
    /// Configure plugin with custom settings
    pub fn configure_plugin(&mut self, name: &str, config: PluginConfig) -> CoelanoxResult<()> {
        if let Some(plugin) = self.plugins.get_mut(name) {
            plugin.configure(config.clone())?;
            self.plugin_configs.insert(name.to_string(), config);
        } else {
            return Err(CoelanoxError::format(&format!("Plugin '{}' not found", name)));
        }
        Ok(())
    }
    
    /// Get plugin configuration
    pub fn get_plugin_config(&self, name: &str) -> Option<&PluginConfig> {
        self.plugin_configs.get(name)
    }
    
    /// Discover plugins in a directory
    pub fn discover_plugins(&mut self, plugin_dir: &Path) -> CoelanoxResult<()> {
        if !plugin_dir.exists() {
            return Ok(());
        }
        
        for entry in std::fs::read_dir(plugin_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            // Check for shared library files
            if let Some(ext) = path.extension() {
                if ext == "so" || ext == "dll" || ext == "dylib" {
                    println!("🔍 Found plugin file: {:?}", path);
                    // TODO: Implement dynamic loading of shared libraries
                    // For now, just log the discovery
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate all registered plugins
    pub fn validate_all_plugins(&self) -> CoelanoxResult<()> {
        for (name, plugin) in &self.plugins {
            plugin.validate()
                .map_err(|e| CoelanoxError::format(&format!("Plugin '{}' validation failed: {}", name, e)))?;
        }
        Ok(())
    }
    
    /// Get plugins that support a specific format
    pub fn get_plugins_for_format(&self, format: ModelFormat) -> Vec<&str> {
        self.plugins
            .iter()
            .filter(|(_, plugin)| plugin.supported_formats().contains(&format))
            .map(|(name, _)| name.as_str())
            .collect()
    }
    
    /// Load a plugin from file (placeholder for dynamic loading)
    pub fn load_plugin(&mut self, plugin_path: &Path) -> CoelanoxResult<()> {
        // For now, just return Ok - plugin loading from files not implemented
        println!("⚠️  Plugin loading from file not implemented: {:?}", plugin_path);
        Ok(())
    }
}
/// Dummy plugin for testing purposes
pub struct DummyPlugin {
    name: String,
    version: String,
    author: String,
}

impl DummyPlugin {
    pub fn new() -> Self {
        Self {
            name: "dummy_plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "COELANOX Team".to_string(),
        }
    }
}

impl TranslatorPlugin for DummyPlugin {
    fn translate(&self, _model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        let mut ir = UniversalIR::new("dummy_model".to_string());
        
        // Add a dummy operation
        let _node_id = ir.add_node(super::ir::OpType::Add);
        
        // Add a dummy tensor
        let _tensor_id = ir.add_intermediate_tensor(
            "dummy_tensor".to_string(),
            super::ir::Shape::new(vec![1, 1]),
            super::ir::DataType::Float32,
        );
        
        Ok(ir)
    }
    
    fn extract_weights(&self, _model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        Ok(vec![1.0, 2.0, 3.0])
    }
    
    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Onnx]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
    
    fn author(&self) -> &str {
        &self.author
    }
    
    fn description(&self) -> &str {
        "Dummy plugin for testing purposes"
    }
    
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manager_enhanced() {
        let mut manager = PluginManager::new();
        
        // Test initial state
        assert_eq!(manager.plugin_count(), 0);
        assert!(manager.list_plugins().is_empty());
        
        // Register a plugin with validation
        let dummy_plugin = Box::new(DummyPlugin::new());
        manager.register_plugin_with_validation("dummy".to_string(), dummy_plugin).unwrap();
        
        // Test plugin registration
        assert_eq!(manager.plugin_count(), 1);
        assert_eq!(manager.list_plugins(), vec!["dummy"]);
        
        // Test plugin capabilities
        let capabilities = manager.get_plugin_capabilities("dummy").unwrap();
        assert_eq!(capabilities.version, "1.0.0");
        assert_eq!(capabilities.author, "COELANOX Team");
        assert_eq!(capabilities.description, "Dummy plugin for testing purposes");
        
        // Test plugin configuration
        let config = PluginConfig {
            verbose: true,
            optimization_level: Some("maximum".to_string()),
            data_type_preferences: {
                let mut map = HashMap::new();
                map.insert("preferred_type".to_string(), "Float16".to_string());
                map
            },
            custom_params: HashMap::new(),
        };
        
        manager.configure_plugin("dummy", config).unwrap();
        
        // Test plugin lookup
        let plugin = manager.get_plugin("dummy").unwrap();
        assert_eq!(plugin.name(), "dummy_plugin");
        
        // Test format-based lookup
        let format_plugin = manager.find_plugin(ModelFormat::Onnx).unwrap();
        assert_eq!(format_plugin.name(), "dummy_plugin");
        
        // Test plugin removal
        let removed = manager.remove_plugin("dummy").unwrap();
        assert_eq!(manager.plugin_count(), 0);
    }
    
    #[test]
    fn test_plugin_configuration() {
        let mut manager = PluginManager::new();
        let dummy_plugin = Box::new(DummyPlugin::new());
        manager.register_plugin("test".to_string(), dummy_plugin);
        
        // Test default configuration
        assert!(manager.get_plugin_config("test").is_none());
        
        // Test custom configuration
        let config = PluginConfig {
            verbose: true,
            optimization_level: Some("advanced".to_string()),
            data_type_preferences: HashMap::new(),
            custom_params: {
                let mut map = HashMap::new();
                map.insert("test_param".to_string(), "test_value".to_string());
                map
            },
        };
        
        manager.configure_plugin("test", config).unwrap();
        
        let saved_config = manager.get_plugin_config("test").unwrap();
        assert!(saved_config.verbose);
        assert_eq!(saved_config.optimization_level, Some("advanced".to_string()));
        assert_eq!(saved_config.custom_params.get("test_param"), Some(&"test_value".to_string()));
    }
    
    #[test]
    fn test_plugin_validation() {
        let mut manager = PluginManager::new();
        let dummy_plugin = Box::new(DummyPlugin::new());
        manager.register_plugin("test".to_string(), dummy_plugin);
        
        // Test individual plugin validation
        let plugin = manager.get_plugin("test").unwrap();
        plugin.validate().unwrap();
        
        // Test all plugins validation
        manager.validate_all_plugins().unwrap();
    }
    
    #[test]
    fn test_plugin_discovery() {
        let mut manager = PluginManager::new();
        
        // Test discovery with non-existent directory
        manager.discover_plugins(Path::new("/non/existent/path")).unwrap();
        
        // Test discovery with current directory (should not find any .so files)
        manager.discover_plugins(Path::new(".")).unwrap();
    }
}
