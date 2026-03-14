//! Plugin system for Coelanox translators (compiler frontends)
//!
//! Manages translators that convert model formats to UniversalIR.

use std::collections::HashMap;
use std::path::Path;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use coelanox_core::plugins::CoelanoxTranslator;
use coelanox_core::ir::UniversalIR;

#[cfg(feature = "translator-onnx")]
use crate::translators::OnnxTranslator;
#[cfg(feature = "translator-bert")]
use crate::translators::BertTranslator;
#[cfg(feature = "translator-resnet")]
use crate::translators::ResNetTinyMnistTranslator;

/// Plugin manager for loading and managing Coelanox translators
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn CoelanoxTranslator>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn register_plugin(&mut self, name: String, plugin: Box<dyn CoelanoxTranslator>) {
        self.plugins.insert(name, plugin);
    }

    pub fn register_plugin_with_validation(
        &mut self,
        name: String,
        plugin: Box<dyn CoelanoxTranslator>,
    ) -> CoelanoxResult<()> {
        self.plugins.insert(name, plugin);
        Ok(())
    }

    pub fn load_default_plugins(&mut self) -> CoelanoxResult<()> {
        #[cfg(feature = "translator-onnx")]
        {
            let onnx_translator = Box::new(OnnxTranslator::new());
            self.register_plugin("onnx".to_string(), onnx_translator);
        }

        #[cfg(feature = "translator-bert")]
        {
            let bert_translator = Box::new(BertTranslator::new());
            self.register_plugin("bert_base_uncased".to_string(), bert_translator);
        }

        #[cfg(feature = "translator-resnet")]
        {
            let resnet_translator = Box::new(ResNetTinyMnistTranslator::new());
            self.register_plugin("resnet_tiny_mnist".to_string(), resnet_translator);
        }

        println!("✅ Plugin manager initialized");
        Ok(())
    }

    pub fn find_plugin(&self, format: ModelFormat) -> Option<&dyn CoelanoxTranslator> {
        self.find_plugin_with_preference(format, None)
    }

    pub fn find_plugin_with_preference(
        &self,
        format: ModelFormat,
        use_demo_translator: Option<bool>,
    ) -> Option<&dyn CoelanoxTranslator> {
        println!("🔍 Searching for plugin supporting format: {:?}", format);
        println!("📋 Available plugins: {:?}", self.plugins.keys().collect::<Vec<_>>());

        let prefer_demo = use_demo_translator.unwrap_or(false);

        if prefer_demo {
            println!("🎯 Demo mode: Preferring specialized translators");
            let specialized_plugins: Vec<&str> = vec!["bert_base_uncased"];
            for plugin_name in specialized_plugins {
                if let Some(plugin) = self.plugins.get(plugin_name) {
                    let supported = plugin.supported_formats();
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

        #[cfg(feature = "translator-onnx")]
        let general_plugins: Vec<&str> = vec!["onnx"];
        #[cfg(not(feature = "translator-onnx"))]
        let general_plugins: Vec<&str> = vec![];

        for plugin_name in general_plugins {
            if let Some(plugin) = self.plugins.get(plugin_name) {
                let supported = plugin.supported_formats();
                if supported.contains(&format) {
                    println!("✅ Found general plugin: {} (default)", plugin_name);
                    return Some(plugin.as_ref());
                }
            }
        }

        for (name, plugin) in &self.plugins {
            let supported = plugin.supported_formats();
            if supported.contains(&format) {
                println!("✅ Found matching plugin: {}", name);
                return Some(plugin.as_ref());
            }
        }
        println!("❌ No plugin found for format: {:?}", format);
        None
    }

    pub fn get_plugin(&self, name: &str) -> Option<&dyn CoelanoxTranslator> {
        self.plugins.get(name).map(|p| p.as_ref())
    }

    pub fn translate_from_path(&self, input_path: &Path, format: ModelFormat) -> CoelanoxResult<UniversalIR> {
        if let Some(plugin) = self.find_plugin(format.clone()) {
            plugin.translate_from_path(input_path)
        } else {
            Err(CoelanoxError::format(&format!("No plugin found for format: {:?}", format)))
        }
    }

    pub fn get_plugin_mut(&mut self, name: &str) -> Option<&mut dyn CoelanoxTranslator> {
        self.plugins.get_mut(name).map(|p| p.as_mut())
    }

    pub fn remove_plugin(&mut self, name: &str) -> Option<Box<dyn CoelanoxTranslator>> {
        self.plugins.remove(name)
    }

    pub fn plugin_names(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }

    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugins.contains_key(name)
    }

    pub fn plugins(&self) -> &HashMap<String, Box<dyn CoelanoxTranslator>> {
        &self.plugins
    }

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

    pub fn supports_format(&self, format: ModelFormat) -> bool {
        self.plugins.values().any(|p| p.supports_format(format.clone()))
    }

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

#[derive(Debug, Clone)]
pub struct PluginStats {
    pub total_plugins: usize,
    pub supported_formats: usize,
    pub plugins: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPlugin {
        name: String,
    }

    impl CoelanoxTranslator for DummyPlugin {
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

        let plugin = manager.get_plugin("dummy").unwrap();
        assert_eq!(plugin.name(), "dummy_plugin");

        let format_plugin = manager.find_plugin(ModelFormat::Onnx).unwrap();
        assert_eq!(format_plugin.name(), "dummy_plugin");

        let _removed = manager.remove_plugin("dummy").unwrap();
        assert_eq!(manager.plugin_count(), 0);
    }

    #[test]
    fn test_plugin_metadata() {
        let mut manager = PluginManager::new();
        let plugin = Box::new(DummyPlugin {
            name: "dummy_plugin".to_string(),
        });
        manager.register_plugin("dummy".to_string(), plugin);

        let plugin = manager.get_plugin("dummy").unwrap();
        assert_eq!(plugin.name(), "dummy_plugin");
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.author(), "Test Author");
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
