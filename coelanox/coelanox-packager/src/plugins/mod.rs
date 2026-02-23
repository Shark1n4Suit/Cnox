//! Plugin system for translator plugins
//! 
//! This module provides the plugin system for loading and managing translator
//! plugins that convert different model formats to Universal IR.

use std::collections::HashMap;
use std::path::Path;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use coelanox_core::plugins::{TranslatorPlugin, PluginConfig, PluginMetadata, PluginCapabilities};
use coelanox_core::ir::UniversalIR;

// Import the ONNX translator from the translators directory
// use onnx_translator::OnnxTranslator;  // Commented out - onnx-translator not available
// Import the BERT translator from the translators directory
use crate::translators::{BertTranslator, ResNetTinyMnistTranslator};

/// Plugin manager for loading and managing translator plugins
pub struct PluginManager {
    /// Registered plugins
    plugins: HashMap<String, Box<dyn TranslatorPlugin>>,
    /// Plugin configuration
    config: PluginConfig,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            config: PluginConfig::default(),
        }
    }
    
    /// Create a new plugin manager with configuration
    pub fn with_config(config: PluginConfig) -> Self {
        Self {
            plugins: HashMap::new(),
            config,
        }
    }
    
    /// Register a plugin
    pub fn register_plugin(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>) {
        self.plugins.insert(name, plugin);
    }
    
    
    /// Register a plugin with validation
    pub fn register_plugin_with_validation(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>) -> CoelanoxResult<()> {
        // Register the plugin directly without validation for now
        self.plugins.insert(name, plugin);
        Ok(())
    }
    
    /// Load default plugins
    pub fn load_default_plugins(&mut self) -> CoelanoxResult<()> {
        // Register ONNX translator from the translators directory
        // Commented out - translators not available
        // let onnx_translator = Box::new(OnnxTranslator::new());
        // self.register_plugin_with_validation("onnx_translator".to_string(), onnx_translator)?;
        
        // BERT translator (bundle format: BRT1 + config + weights)
        let bert_translator = Box::new(BertTranslator::new());
        self.register_plugin("bert_base_uncased".to_string(), bert_translator);

        // ResNet-tiny-MNIST translator (bundle format: RTM1 + config + weights)
        let resnet_translator = Box::new(ResNetTinyMnistTranslator::new());
        self.register_plugin("resnet_tiny_mnist".to_string(), resnet_translator);
        
        println!("✅ Plugin manager initialized (BERT + ResNet-tiny-MNIST translators registered)");
        Ok(())
    }
    
    /// Find a plugin for the given model format
    pub fn find_plugin(&self, format: ModelFormat) -> Option<&dyn TranslatorPlugin> {
        self.find_plugin_with_preference(format, None)
    }
    
    /// Find a plugin for the given model format with optional demo/custom translator preference
    /// 
    /// - If `use_demo_translator` is Some(true), prefer specialized translators (e.g., bert_translator for NLP)
    /// - If `use_demo_translator` is None or Some(false), prefer general translators (e.g., onnx_translator)
    pub fn find_plugin_with_preference(&self, format: ModelFormat, use_demo_translator: Option<bool>) -> Option<&dyn TranslatorPlugin> {
        println!("🔍 Searching for plugin supporting format: {:?}", format);
        println!("📋 Available plugins: {:?}", self.plugins.keys().collect::<Vec<_>>());
        
        let prefer_demo = use_demo_translator.unwrap_or(false);
        
        if prefer_demo {
            println!("🎯 Demo mode: Preferring specialized translators");
            
            // For demo mode, prefer specialized translators (bert_translator) over general ones
            // Try specialized translators first
            let specialized_plugins: Vec<&str> = vec!["bert_base_uncased"];
            
            for plugin_name in specialized_plugins {
                if let Some(plugin) = self.plugins.get(plugin_name) {
                    let supported = plugin.supported_formats();
                    println!("   Checking specialized plugin '{}' (supports: {:?})", plugin_name, supported);
                    if supported.contains(&format) {
                        println!("✅ Found specialized plugin: {} (demo mode)", plugin_name);
                        return Some(plugin.as_ref());
                    }
                }
            }
            
            println!("⚠️  No specialized plugin available, falling back to general translator");
        } else {
            println!("⚙️  Standard mode: Preferring general translators");
        }
        
        // Default/preferred behavior: prefer general translators (onnx_translator)
        // This ensures we get the full ONNX translator by default
        // Note: onnx_translator not available, so general_plugins is empty for now
        let general_plugins: Vec<&str> = vec![];
        
        for plugin_name in general_plugins {
            if let Some(plugin) = self.plugins.get(plugin_name) {
                let supported = plugin.supported_formats();
                println!("   Checking general plugin '{}' (supports: {:?})", plugin_name, supported);
                if supported.contains(&format) {
                    println!("✅ Found general plugin: {} (default)", plugin_name);
                    return Some(plugin.as_ref());
                }
            }
        }
        
        // Fallback: Search through all plugins to find one that supports this format
        for (name, plugin) in &self.plugins {
            let supported = plugin.supported_formats();
            println!("   Plugin '{}' supports: {:?}", name, supported);
            if supported.contains(&format) {
                println!("✅ Found matching plugin: {}", name);
                return Some(plugin.as_ref());
            }
        }
        println!("❌ No plugin found for format: {:?}", format);
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
        self.plugins.remove(name)
    }
    
    /// Get all plugin names
    pub fn plugin_names(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }
    
    /// Get plugin count
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
    
    /// Check if a plugin is registered
    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugins.contains_key(name)
    }
    
    /// Get all plugins
    pub fn plugins(&self) -> &HashMap<String, Box<dyn TranslatorPlugin>> {
        &self.plugins
    }
    
    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Option<PluginMetadata> {
        self.plugins.get(name).map(|p| p.metadata())
    }
    
    /// Get plugin capabilities
    pub fn get_plugin_capabilities(&self, name: &str) -> Option<PluginCapabilities> {
        self.plugins.get(name).map(|p| p.capabilities())
    }
    
    /// Configure a plugin
    pub fn configure_plugin(&mut self, name: &str, config: PluginConfig) -> CoelanoxResult<()> {
        if let Some(plugin) = self.plugins.get_mut(name) {
            plugin.configure(&config)
        } else {
            Err(CoelanoxError::format(&format!("Plugin '{}' not found", name)))
        }
    }
    
    /// Validate all plugins
    pub fn validate_all_plugins(&self) -> CoelanoxResult<()> {
        for (name, plugin) in &self.plugins {
            plugin.validate_config(&self.config)
                .map_err(|e| CoelanoxError::format(&format!("Plugin '{}' validation failed: {}", name, e)))?;
        }
        Ok(())
    }
    
    /// Get supported formats across all plugins
    pub fn supported_formats(&self) -> Vec<ModelFormat> {
        let mut formats = Vec::new();
        for plugin in self.plugins.values() {
            for format in plugin.supported_formats() {
                if !formats.contains(&format) {
                    formats.push(format);
                }
            }
        }
        formats
    }
    
    /// Check if a format is supported by any plugin
    pub fn supports_format(&self, format: ModelFormat) -> bool {
        self.plugins.values().any(|p| p.supports_format(format.clone()))
    }
    
    /// Get plugin statistics
    pub fn get_stats(&self) -> PluginStats {
        PluginStats {
            total_plugins: self.plugins.len(),
            supported_formats: self.supported_formats().len(),
            plugins: self.plugins.keys().cloned().collect(),
        }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin statistics
#[derive(Debug, Clone)]
pub struct PluginStats {
    /// Total number of plugins
    pub total_plugins: usize,
    /// Number of supported formats
    pub supported_formats: usize,
    /// List of plugin names
    pub plugins: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    /// Dummy plugin for testing
    struct DummyPlugin {
        name: String,
    }
    
    impl TranslatorPlugin for DummyPlugin {
        fn translate(&self, _model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
            Ok(UniversalIR::new("dummy_model".to_string()))
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
            "1.0.0"
        }
        
        fn author(&self) -> &str {
            "Test Author"
        }
    }
    
    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.plugin_count(), 0);
    }
    
    #[test]
    fn test_plugin_registration() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(DummyPlugin {
            name: "dummy_plugin".to_string(),
        });
        
        manager.register_plugin("dummy".to_string(), plugin);
        assert_eq!(manager.plugin_count(), 1);
        assert!(manager.has_plugin("dummy"));
    }
    
    #[test]
    fn test_plugin_lookup() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(DummyPlugin {
            name: "dummy_plugin".to_string(),
        });
        
        manager.register_plugin("dummy".to_string(), plugin);
        
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
    fn test_plugin_metadata() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(DummyPlugin {
            name: "dummy_plugin".to_string(),
        });
        
        manager.register_plugin("dummy".to_string(), plugin);
        
        let metadata = manager.get_plugin_metadata("dummy").unwrap();
        assert_eq!(metadata.name, "dummy_plugin");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.author, "Test Author");
    }
    
    #[test]
    fn test_plugin_stats() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(DummyPlugin {
            name: "dummy_plugin".to_string(),
        });
        
        manager.register_plugin("dummy".to_string(), plugin);
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_plugins, 1);
        assert_eq!(stats.supported_formats, 1);
        assert!(stats.plugins.contains(&"dummy".to_string()));
    }
}
