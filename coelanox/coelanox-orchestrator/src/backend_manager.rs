//! Backend Manager for COELANOX
//!
//! This module provides package-time discovery of CLF (Coelanox Library File) backends.
//! The BackendManager discovers .clfc files from search paths for use during packaging
//! and container execution.
//!
//! **Note:** Used at package-time by the packager for CLF discovery. The orchestrator
//! also holds runtime execution backends (see `execution_backend`, `fallback`).

use coelanox_core::format::HardwareTarget;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Backend kind: CLF (sealed kernel pack). CLF reader is implemented separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
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
    /// Path to the CLF file (.clfc compute, or legacy .clf)
    pub library_path: PathBuf,
    /// Hardware targets supported by this backend
    pub supported_targets: Vec<HardwareTarget>,
    /// Backend kind (CLF)
    pub kind: BackendKind,
}

impl BackendInfo {
    /// Check backend health (MVP: assumes healthy; future: load library, validate interface, run health check)
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

/// Backend Manager for discovering CLF backends
///
/// This manager discovers .clfc files from search paths at package time.
/// The discovered backends can then be used by the packager for code generation.
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
    
    /// Discover CLF backends from search paths
    ///
    /// Scans the provided search paths for .clfc (and legacy .clf) files.
    ///
    /// # Arguments
    /// * `search_paths` - Paths to search for CLF files
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

                if Self::is_clf_file(&path) {
                    // In the compute subdir (clfc/), only load .clfc files; legacy .clf is not used for compute.
                    if Self::is_compute_subdir(search_path) && !Self::is_clfc_extension(&path) {
                        continue;
                    }
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
    
    /// Check if a file is a CLF (Coelanox Library File). Accepts .clfc (compute) and .clf (legacy).
    fn is_clf_file(path: &Path) -> bool {
        path.extension()
            .map(|e| {
                let ext = e.to_string_lossy().to_lowercase();
                ext == "clfc" || ext == "clf"
            })
            .unwrap_or(false)
    }

    /// True if this path is the compute subdirectory (e.g. .../clf/clfc). Compute backends must be .clfc.
    fn is_compute_subdir(search_path: &Path) -> bool {
        search_path
            .file_name()
            .map(|n| n == "clfc")
            .unwrap_or(false)
    }

    /// True if the file has extension .clfc (compute CLF). Used so we only load .clfc in clfc/, not legacy .clf.
    fn is_clfc_extension(path: &Path) -> bool {
        path.extension()
            .map(|e| e.to_string_lossy().to_lowercase() == "clfc")
            .unwrap_or(false)
    }
    
    /// Register a .clfc (or legacy .clf) file as a backend. Reads CLF header to set supported_targets from target field.
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
        let supported_targets = Self::clf_target_to_hardware(path);
        Ok(BackendInfo {
            name,
            version: "1.0".to_string(),
            library_path: path.to_path_buf(),
            supported_targets,
            kind: BackendKind::Clf,
        })
    }

    /// Read CLF header target string and map to HardwareTarget(s). Defaults to Gpu if read fails.
    fn clf_target_to_hardware(path: &Path) -> Vec<HardwareTarget> {
        let reader = match coelanox_clf::ClfReader::open(path) {
            Ok(r) => r,
            Err(_) => return vec![HardwareTarget::Gpu],
        };
        let t = reader.header.target.to_lowercase();
        let target = if t.is_empty() || t == "gpu" || t == "vulkan" || t == "spir-v" {
            HardwareTarget::Gpu
        } else if t == "cpu" || t.contains("x86_64") || t.contains("avx") || t == "simd" {
            HardwareTarget::Cpu
        } else {
            HardwareTarget::Custom(reader.header.target.clone())
        };
        vec![target]
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
    
    /// CLF subdirectory names under the single root (compute, memory, protection, extensions).
    const CLF_SUBDIRS: &[&str] = &["clfc", "clfmm", "clfmp", "clfe"];

    /// Get default search paths for backend discovery.
    ///
    /// Uses a single locked layout so you do not need to set any env for normal use:
    /// - **Default:** `~/.coelanox/clf/` with subdirs `clfc/`, `clfmm/`, `clfmp/`, `clfe/`.
    ///   Compute backends (.clfc) go in `clfc/`. Root and subdirs are created if missing.
    /// - **Override:** `COELANOX_CLF_PATH` = directory (root for the four subdirs) or path to a
    ///   single .clfc file (its parent directory is used so the file is discovered).
    /// - **Extra:** `COELANOX_BACKEND_PATH` (colon-separated) is appended for additional paths.
    ///
    /// # Returns
    /// * Vector of directory paths to search for backend libraries
    pub fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(clf_path) = std::env::var("COELANOX_CLF_PATH") {
            let p = PathBuf::from(&clf_path);
            if p.exists() {
                let root = if p.is_file() {
                    p.parent().map(PathBuf::from).unwrap_or_else(|| p.clone())
                } else {
                    p.clone()
                };
                paths.push(root.clone());
                for sub in Self::CLF_SUBDIRS {
                    paths.push(root.join(sub));
                }
            }
        } else if let Some(home) = std::env::var_os("HOME") {
            let root = PathBuf::from(home).join(".coelanox").join("clf");
            for sub in Self::CLF_SUBDIRS {
                let subdir = root.join(sub);
                let _ = std::fs::create_dir_all(&subdir);
                paths.push(subdir);
            }
        }

        if let Ok(env_path) = std::env::var("COELANOX_BACKEND_PATH") {
            for path_str in env_path.split(':') {
                let path_str = path_str.trim();
                if !path_str.is_empty() {
                    paths.push(PathBuf::from(path_str));
                }
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
    
    /// Select backend using fallback strategy (package-time only; runtime uses ExecutionBackend registry)
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

