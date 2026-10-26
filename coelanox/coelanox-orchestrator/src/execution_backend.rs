//! Runtime Execution Backend for COELANOX
//!
//! This module provides the execution backend abstraction used at runtime.
//! The runtime routes execution through the backend registry: it tries the primary
//! path (CPU native machine code) first, then falls back to registered backends
//! (e.g. scalar-fallback) when the primary path fails.
//!
//! **Architecture:**
//! - No hardcoded primary path - the backend manager chooses execution based on
//!   the hardware flag (--backend from CLI: cpu, gpu, npu, tpu, scalar, auto).
//! - Scalar is always the fallback when the selected backend fails.

use coelanox_core::format::CoelanoxContainer;
use coelanox_core::ir::OptimizedIR;
use coelanox_core::CoelanoxResult;
use std::collections::HashMap;

/// Cached (warm) data for a container to avoid per-inference decompress and IR deserialize.
/// Filled by the orchestrator on first use; backends use it when provided.
/// When `optimized_ir` is `None`, only weights are cached (plan path); scalar path requires `Some`.
#[derive(Clone)]
pub struct ContainerWarmData {
    /// Decompressed weights (one-time decompress, reused for all inferences).
    pub decompressed_weights: Vec<f32>,
    /// Deserialized IR for scalar fallback (avoids bincode::deserialize every run).
    /// `None` when only weights were resolved (plan path); scalar fallback requires `Some`.
    pub optimized_ir: Option<OptimizedIR>,
}

/// Trait for runtime execution backends.
///
/// Execution backends can run inference given a container and input data.
/// When the orchestrator has filled the warm cache, `warm_data` is `Some` and
/// backends should use it instead of decompressing/deserializing again.
pub trait ExecutionBackend: Send + Sync {
    /// Execute inference on the given container with input data.
    ///
    /// When `warm_data` is `Some`, the backend should use cached weights and IR
    /// instead of calling `container.decompress_weights()` or deserializing `container.ir_data`.
    ///
    /// # Arguments
    /// * `container` - The loaded COELANOX container (for manifest, shapes, etc.)
    /// * `input_data` - Flattened input tensor data
    /// * `warm_data` - Optional cached decompressed weights and deserialized IR (orchestrator fills on first run)
    ///
    /// # Returns
    /// * `Ok(Vec<f32>)` - Flattened output tensor data
    /// * `Err(CoelanoxError)` - Execution failed
    fn execute(
        &self,
        container: &CoelanoxContainer,
        input_data: &[f32],
        warm_data: Option<&ContainerWarmData>,
    ) -> CoelanoxResult<Vec<f32>>;

    /// Backend name for registry and fallback selection.
    fn name(&self) -> &str;

    /// Backend version.
    fn version(&self) -> &str;
}

/// Registry of runtime execution backends.
///
/// Holds fallback backends (e.g. scalar-fallback). The runtime tries its primary
/// execution path first, then uses backends from this registry when the primary fails.
#[derive(Default)]
pub struct RuntimeBackendRegistry {
    backends: HashMap<String, Box<dyn ExecutionBackend>>,
}

impl RuntimeBackendRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register an execution backend. Replaces existing if present.
    pub fn register(&mut self, name: impl Into<String>, backend: Box<dyn ExecutionBackend>) {
        let name = name.into();
        self.backends.insert(name, backend);
    }

    /// Get a backend by name.
    pub fn get(&self, name: &str) -> Option<&dyn ExecutionBackend> {
        self.backends.get(name).map(|b| b.as_ref())
    }

    /// Get the scalar fallback backend if registered.
    pub fn scalar_fallback(&self) -> Option<&dyn ExecutionBackend> {
        self.get("scalar-fallback")
    }

    /// List all registered backend names.
    pub fn list(&self) -> Vec<&str> {
        self.backends.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a backend is registered.
    pub fn has(&self, name: &str) -> bool {
        self.backends.contains_key(name)
    }
}
