use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;
use std::sync::OnceLock;

/// SAIC Configuration Settings
/// 
/// This struct defines all configurable options for the SAIC runner,
/// including audit logging, performance settings, and backend preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaicConfig {
    /// Enable detailed audit logging for compliance and debugging
    #[serde(default = "default_audit_mode")]
    pub audit_mode: bool,
    
    /// Log level for tracing (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub log_level: String,
    
    /// Default backend preference (cpu, gpu, auto)
    #[serde(default = "default_backend")]
    pub default_backend: String,
    
    /// Enable performance optimizations
    #[serde(default = "default_performance_mode")]
    pub performance_mode: bool,
    
    /// Maximum model size in bytes (0 = unlimited)
    #[serde(default = "default_max_model_size")]
    pub max_model_size: u64,
    
    /// Enable model caching for repeated executions
    #[serde(default = "default_enable_caching")]
    pub enable_caching: bool,
    
    /// Cache directory for compiled models
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
}

// Static configuration cache for performance optimization
static CONFIG_CACHE: OnceLock<SaicConfig> = OnceLock::new();

impl Default for SaicConfig {
    fn default() -> Self {
        Self {
            audit_mode: default_audit_mode(),
            log_level: default_log_level(),
            default_backend: default_backend(),
            performance_mode: default_performance_mode(),
            max_model_size: default_max_model_size(),
            enable_caching: default_enable_caching(),
            cache_dir: default_cache_dir(),
        }
    }
}

// Default value functions
fn default_audit_mode() -> bool { false }
fn default_log_level() -> String { "info".to_string() }
fn default_backend() -> String { "cpu".to_string() }
fn default_performance_mode() -> bool { true }
fn default_max_model_size() -> u64 { 0 } // Unlimited
fn default_enable_caching() -> bool { true }
fn default_cache_dir() -> PathBuf { 
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("saic")
}

impl SaicConfig {
    /// Load configuration from file
    pub fn from_file(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: SaicConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration with fallback to environment variables
    /// 
    /// This function is optimized with caching to avoid repeated file system calls.
    /// The configuration is loaded once and cached for subsequent calls.
    pub fn load() -> Self {
        // Use cached configuration if available
        if let Some(cached_config) = CONFIG_CACHE.get() {
            return cached_config.clone();
        }
        
        // Load configuration and cache it
        let config = Self::load_uncached();
        
        // Cache the configuration (ignore if already cached by another thread)
        let _ = CONFIG_CACHE.set(config.clone());
        
        config
    }
    
    /// Load configuration without caching (internal use)
    fn load_uncached() -> Self {
        // Try to load from config file first
        let config_paths = vec![
            PathBuf::from("saic.yaml"),
            PathBuf::from("saic.yml"),
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("saic")
                .join("config.yaml"),
        ];
        
        for path in config_paths {
            if let Ok(config) = Self::from_file(&path) {
                return config;
            }
        }
        
        // Fallback to environment variables
        Self::from_env()
    }
    
    /// Load configuration from environment variables
    fn from_env() -> Self {
        use std::env;
        
        Self {
            audit_mode: env::var("SAIC_AUDIT_MODE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or_else(|_| default_audit_mode()),
            log_level: env::var("SAIC_LOG_LEVEL")
                .unwrap_or_else(|_| default_log_level()),
            default_backend: env::var("SAIC_DEFAULT_BACKEND")
                .unwrap_or_else(|_| default_backend()),
            performance_mode: env::var("SAIC_PERFORMANCE_MODE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or_else(|_| default_performance_mode()),
            max_model_size: env::var("SAIC_MAX_MODEL_SIZE")
                .map(|v| v.parse().unwrap_or(0))
                .unwrap_or_else(|_| default_max_model_size()),
            enable_caching: env::var("SAIC_ENABLE_CACHING")
                .map(|v| v == "true" || v == "1")
                .unwrap_or_else(|_| default_enable_caching()),
            cache_dir: env::var("SAIC_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| default_cache_dir()),
        }
    }
    
    /// Create a sample configuration file
    pub fn create_sample_config(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let sample_config = SaicConfig::default();
        sample_config.save_to_file(path)?;
        Ok(())
    }
} 