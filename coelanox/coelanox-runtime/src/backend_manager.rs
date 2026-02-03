//! Backend Manager for COELANOX
//! 
//! This module provides package-time discovery of vendor BackendTranslators via dynamic library loading.
//! The BackendManager discovers vendor-provided BackendTranslators that can be used during packaging
//! to generate hardware-specific code.
//! 
//! **Note:** This is used at package-time by the packager, not at runtime. The location in
//! `coelanox-runtime` is for organizational purposes (backends are conceptually runtime-related).

use coelanox_core::format::HardwareTarget;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Backend kind: translator (.so) or CLF (sealed kernel pack). CLF reader is implemented separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    /// Dynamic library implementing BackendTranslator
    Translator,
    /// CLF (Coelanox Library File): sealed vendor kernel pack; packager uses CLF reader when present
    Clf,
}

/// Information about a discovered backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend name (e.g., "nvidia-h100", "amd-gpu", or CLF stem "intel_gen4")
    pub name: String,
    /// Backend version
    pub version: String,
    /// Path to the dynamic library or .clf file
    pub library_path: PathBuf,
    /// Hardware targets supported by this backend
    pub supported_targets: Vec<HardwareTarget>,
    /// Whether this backend is a translator (.so) or CLF (.clf)
    pub kind: BackendKind,
}

impl BackendInfo {
    /// Check backend health (placeholder for future implementation)
    pub fn check_health(&self) -> CoelanoxResult<BackendHealth> {
        // For MVP, assume all discovered backends are healthy
        // In future, this will:
        // - Load the library
        // - Call health check function
        // - Validate backend interface version
        // - Test basic functionality
        Ok(BackendHealth::Healthy)
    }
}

/// Backend health status
#[derive(Debug, Clone, PartialEq)]
pub enum BackendHealth {
    /// Backend is healthy and ready
    Healthy,
    /// Backend is degraded but functional
    Degraded(String),
    /// Backend is unhealthy and should not be used
    Unhealthy(String),
    /// Health status unknown
    Unknown,
}

/// Result of backend discovery operation
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Successfully discovered backends
    pub discovered: Vec<BackendInfo>,
    /// Failed library paths with error messages
    pub failures: Vec<(PathBuf, String)>,
    /// Total libraries scanned
    pub total_scanned: usize,
}

impl DiscoveryResult {
    pub fn new() -> Self {
        Self {
            discovered: Vec::new(),
            failures: Vec::new(),
            total_scanned: 0,
        }
    }
    
    pub fn success_count(&self) -> usize {
        self.discovered.len()
    }
    
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }
}

/// Backend selection strategy
#[derive(Debug, Clone)]
pub enum BackendSelectionStrategy {
    /// Use primary backend only (fail if unavailable)
    PrimaryOnly(String),
    /// Use primary with fallback to scalar
    PrimaryWithFallback { primary: String, fallback: String },
    /// Try all backends in order until one succeeds
    TryAll(Vec<String>),
}

/// Backend Manager for discovering vendor BackendTranslators
/// 
/// This manager discovers vendor-provided BackendTranslators from dynamic libraries
/// at package time. The discovered backends can then be used by the packager to
/// generate hardware-specific code.
pub struct BackendManager {
    /// Map of backend name to backend information
    discovered_backends: HashMap<String, BackendInfo>,
}

impl BackendManager {
    /// Create a new BackendManager
    pub fn new() -> Self {
        Self {
            discovered_backends: HashMap::new(),
        }
    }
    
    /// Discover vendor backends from search paths
    /// 
    /// Scans the provided search paths for dynamic libraries and attempts to
    /// discover BackendTranslators from them.
    /// 
    /// # Arguments
    /// * `search_paths` - Paths to search for backend libraries
    /// 
    /// # Returns
    /// * `Ok(DiscoveryResult)` with detailed information about discovered and failed backends
    pub fn discover_backends(&mut self, search_paths: &[PathBuf]) -> CoelanoxResult<DiscoveryResult> {
        let mut result = DiscoveryResult::new();
        
        for search_path in search_paths {
            // Handle path existence and permission errors
            if !search_path.exists() {
                result.failures.push((
                    search_path.clone(),
                    "Path does not exist".to_string()
                ));
                continue;
            }
            
            // Handle directory read errors
            let entries = match std::fs::read_dir(search_path) {
                Ok(e) => e,
                Err(e) => {
                    result.failures.push((
                        search_path.clone(),
                        format!("Failed to read directory: {}", e)
                    ));
                    continue;
                }
            };
            
            for entry in entries {
                result.total_scanned += 1;
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        result.failures.push((
                            search_path.clone(),
                            format!("Failed to read directory entry: {}", e)
                        ));
                        continue;
                    }
                };
                
                let path = entry.path();
                
                if Self::is_dynamic_library(&path) {
                    match self.load_backend_library(&path) {
                        Ok(backend_info) => {
                            result.discovered.push(backend_info.clone());
                            self.register_backend(backend_info);
                        }
                        Err(e) => {
                            result.failures.push((
                                path.clone(),
                                format!("Failed to load backend: {}", e)
                            ));
                        }
                    }
                } else if Self::is_clf_file(&path) {
                    match self.load_clf_backend(&path) {
                        Ok(backend_info) => {
                            result.discovered.push(backend_info.clone());
                            self.register_backend(backend_info);
                        }
                        Err(e) => {
                            result.failures.push((
                                path.clone(),
                                format!("Failed to load CLF backend: {}", e)
                            ));
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Check if a file is a CLF (Coelanox Library File)
    fn is_clf_file(path: &Path) -> bool {
        path.extension()
            .map(|e| e.to_string_lossy().to_lowercase() == "clf")
            .unwrap_or(false)
    }
    
    /// Register a .clf file as a backend. CLF reader (extract blobs by op_id) is implemented separately.
    fn load_clf_backend(&self, path: &Path) -> CoelanoxResult<BackendInfo> {
        if !path.exists() {
            return Err(CoelanoxError::BackendLibraryNotFound(
                path.display().to_string()
            ));
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clf")
            .to_string();
        Ok(BackendInfo {
            name,
            version: "1.0".to_string(),
            library_path: path.to_path_buf(),
            supported_targets: vec![HardwareTarget::Gpu],
            kind: BackendKind::Clf,
        })
    }
    
    /// Check if a file is a dynamic library
    fn is_dynamic_library(path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            #[cfg(target_os = "linux")]
            return ext_str == "so";
            #[cfg(target_os = "macos")]
            return ext_str == "dylib";
            #[cfg(target_os = "windows")]
            return ext_str == "dll";
        }
        false
    }
    
    /// Load a backend library and extract backend information
    /// 
    /// Attempts to load a dynamic library and discover if it contains a BackendTranslator.
    /// For MVP, this is a placeholder that will be implemented when vendor backends are available.
    /// 
    /// # Arguments
    /// * `path` - Path to the dynamic library
    /// 
    /// # Returns
    /// * `Ok(BackendInfo)` if backend is successfully discovered
    /// * `Err` if library cannot be loaded or doesn't contain a valid backend
    pub fn load_backend_library(&self, path: &Path) -> CoelanoxResult<BackendInfo> {
        // Check if file exists
        if !path.exists() {
            return Err(CoelanoxError::BackendLibraryNotFound(
                path.display().to_string()
            ));
        }
        
        // Check if file is readable
        if let Err(e) = std::fs::metadata(path) {
            return Err(CoelanoxError::BackendLibraryLoadFailed(
                path.display().to_string(),
                format!("Cannot access file: {}", e)
            ));
        }
        
        // For MVP, return specific error indicating not yet implemented
        // In future, this will have platform-specific FFI loading with detailed errors:
        // - Library not found
        // - Missing exported symbol
        // - Invalid library format
        // - Version mismatch
        Err(CoelanoxError::BackendLibraryLoadFailed(
            path.display().to_string(),
            "Backend library loading not yet implemented".to_string()
        ))
    }
    
    /// Register a discovered backend
    /// 
    /// # Arguments
    /// * `info` - Backend information to register
    pub fn register_backend(&mut self, info: BackendInfo) {
        self.discovered_backends.insert(info.name.clone(), info);
    }
    
    /// Get backend information by name
    /// 
    /// # Arguments
    /// * `name` - Backend name
    /// 
    /// # Returns
    /// * `Some(&BackendInfo)` if backend is found
    /// * `None` if backend is not registered
    pub fn get_backend(&self, name: &str) -> Option<&BackendInfo> {
        self.discovered_backends.get(name)
    }
    
    /// List all discovered backends
    /// 
    /// # Returns
    /// * Vector of references to all registered backend information
    pub fn list_backends(&self) -> Vec<&BackendInfo> {
        self.discovered_backends.values().collect()
    }
    
    /// Get default search paths for backend discovery
    /// 
    /// Returns a list of paths to search for backend libraries, in order of priority:
    /// 1. `./backends/` (current directory)
    /// 2. `~/.coelanox/backends/` (user directory)
    /// 3. `COELANOX_BACKEND_PATH` environment variable
    /// 
    /// # Returns
    /// * Vector of search paths
    pub fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // 1. Current directory
        paths.push(PathBuf::from("./backends"));
        
        // 2. User directory
        if let Some(home) = std::env::var_os("HOME") {
            let mut user_path = PathBuf::from(home);
            user_path.push(".coelanox");
            user_path.push("backends");
            paths.push(user_path);
        }
        
        // 3. Environment variable
        if let Ok(env_path) = std::env::var("COELANOX_BACKEND_PATH") {
            for path_str in env_path.split(':') {
                paths.push(PathBuf::from(path_str));
            }
        }
        
        paths
    }
    
    /// Check if a backend is available for a hardware target
    /// 
    /// # Arguments
    /// * `target` - Hardware target to check
    /// 
    /// # Returns
    /// * `Some(&BackendInfo)` if a backend supporting the target is found
    /// * `None` if no backend supports the target
    pub fn find_backend_for_target(&self, target: &HardwareTarget) -> Option<&BackendInfo> {
        self.discovered_backends.values()
            .find(|info| info.supported_targets.contains(target))
    }
    
    /// Validate that a backend exists and is available
    pub fn validate_backend(&self, name: &str) -> CoelanoxResult<()> {
        if !self.discovered_backends.contains_key(name) {
            return Err(CoelanoxError::BackendNotFound(name.to_string()));
        }
        Ok(())
    }
    
    /// Validate multiple backends exist
    pub fn validate_backends(&self, names: &[String]) -> CoelanoxResult<Vec<String>> {
        let mut missing = Vec::new();
        for name in names {
            if !self.discovered_backends.contains_key(name) {
                missing.push(name.clone());
            }
        }
        
        if !missing.is_empty() {
            return Err(CoelanoxError::BackendNotFound(
                format!("Missing backends: {}", missing.join(", "))
            ));
        }
        
        Ok(Vec::new())
    }
    
    /// Select backend using fallback strategy (placeholder for runtime)
    pub fn select_backend_with_fallback(
        &self,
        strategy: &BackendSelectionStrategy,
    ) -> CoelanoxResult<&BackendInfo> {
        match strategy {
            BackendSelectionStrategy::PrimaryOnly(name) => {
                self.get_backend(name)
                    .ok_or_else(|| CoelanoxError::BackendNotFound(name.clone()))
            }
            BackendSelectionStrategy::PrimaryWithFallback { primary, fallback } => {
                // Try primary first
                if let Some(backend) = self.get_backend(primary) {
                    Ok(backend)
                } else {
                    // Fallback to scalar
                    self.get_backend(fallback)
                        .ok_or_else(|| CoelanoxError::BackendRoutingFailed(
                            format!("Neither {} nor {} available", primary, fallback)
                        ))
                }
            }
            BackendSelectionStrategy::TryAll(names) => {
                for name in names {
                    if let Some(backend) = self.get_backend(name) {
                        return Ok(backend);
                    }
                }
                Err(CoelanoxError::BackendRoutingFailed(
                    format!("None of the requested backends available: {}", names.join(", "))
                ))
            }
        }
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

