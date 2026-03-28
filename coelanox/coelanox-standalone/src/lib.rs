//! Standalone container loader and runner.
//!
//! Minimal API to load one `.cnox` file and run inference without the full CLI or
//! multi-container orchestrator. Intended for edge/drone deployments where a single
//! container is loaded and executed (container is self-contained; this crate is a
//! thin entry point that uses the orchestrator under the hood).
//!
//! The **`coelanox-run`** binary also supports **`--serve`**: framed binary IPC v1 on stdin/stdout
//! (same protocol as `coelanox serve`), implemented via `coelanox-ipc`.
//!
//! For server use (multiple containers, cache, TTL), use `coelanox-orchestrator` directly.

use coelanox_core::CoelanoxResult;
use coelanox_orchestrator::CoelanoxRuntime;
use std::path::Path;

/// Load a container from path and run inference with the given input.
/// Uses the default runtime (one container, no TTL). Returns the output tensor as f32 vector.
///
/// # Errors
/// Returns error if the file cannot be read, validation fails, or execution fails.
pub fn run_container(path: impl AsRef<Path>, input: &[f32]) -> CoelanoxResult<Vec<f32>> {
    let name = path.as_ref().to_string_lossy().into_owned();
    let runtime = CoelanoxRuntime::new();
    runtime.load_container(&name)?;
    runtime.execute(&name, input)
}

/// Load a container from path and run inference with auto-generated dummy input (for testing).
pub fn run_container_dummy_input(path: impl AsRef<Path>) -> CoelanoxResult<Vec<f32>> {
    let name = path.as_ref().to_string_lossy().into_owned();
    let runtime = CoelanoxRuntime::new();
    runtime.load_container(&name)?;
    let container = runtime.get_container_info(&name)?;
    let input_size: usize = container
        .manifest
        .input_shapes
        .iter()
        .map(|s| s.iter().product::<usize>())
        .sum();
    let input = vec![1.0f32; input_size];
    runtime.execute(&name, &input)
}
