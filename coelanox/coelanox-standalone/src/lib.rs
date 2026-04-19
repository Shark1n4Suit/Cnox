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
//! Trust and verification flags mirror the main `coelanox` CLI (`--trusted-key`, `--require-signature`,
//! `--no-verify`). You can also set `COELANOX_TRUSTED_PUBLIC_KEY` (path to a 32-byte public key file),
//! `COELANOX_CONFIG_FILE`, and `COELANOX_CONFIG_STRICT=1` (fail if the config file is missing or invalid)
//! via `coelanox_orchestrator::RuntimeConfig::from_env_with_optional_config_file`.
//!
//! For server use (multiple containers, cache, TTL), use `coelanox-orchestrator` directly.

use coelanox_core::format::is_container_signed;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_orchestrator::{CoelanoxRuntime, RuntimeConfig};
use std::path::{Path, PathBuf};

/// Options for loading and verifying a container (parity with `coelanox run` / `serve` flags).
#[derive(Debug, Clone, Default)]
pub struct ContainerRunOptions {
    /// Path to a 32-byte Ed25519 public key file (also: env `COELANOX_TRUSTED_PUBLIC_KEY`).
    pub trusted_key_path: Option<PathBuf>,
    /// Skip hash (and signature) verification at load time (`coelanox run --no-verify`).
    pub no_verify: bool,
    /// Require a trusted key when the container is signed (`coelanox run --require-signature`).
    pub require_signature: bool,
}

/// Resolve trusted public key from `opts` or `COELANOX_TRUSTED_PUBLIC_KEY` (path to 32-byte file).
pub fn resolve_trusted_key(opts: &ContainerRunOptions) -> CoelanoxResult<Option<[u8; 32]>> {
    let p = opts
        .trusted_key_path
        .as_ref()
        .map(|pb| pb.to_string_lossy().into_owned())
        .or_else(|| std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").ok());
    match p.as_deref() {
        Some(path) => {
            let key = std::fs::read(path).map_err(|e| {
                CoelanoxError::format(format!("Could not read trusted public key {}: {}", path, e))
            })?;
            if key.len() != 32 {
                return Err(CoelanoxError::format(format!(
                    "Trusted public key must be 32 bytes; got {} bytes",
                    key.len()
                )));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&key);
            Ok(Some(arr))
        }
        None => Ok(None),
    }
}

/// Load container per options (verify / no-verify, optional trusted key, `--require-signature` checks).
pub fn load_container_with_options(
    runtime: &CoelanoxRuntime,
    container_path: &str,
    opts: &ContainerRunOptions,
) -> CoelanoxResult<()> {
    if opts.require_signature {
        let signed = is_container_signed(Path::new(container_path))?;
        let has_key = opts.trusted_key_path.is_some()
            || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set.",
            ));
        }
    }
    if let Some(key) = resolve_trusted_key(opts)? {
        runtime.set_trusted_public_key(key)?;
    }
    if opts.no_verify {
        runtime.load_container_no_verify(container_path)?;
    } else {
        runtime.load_container(container_path)?;
    }
    Ok(())
}

/// Load a container from path and run inference with the given input (default load options).
pub fn run_container_simple(path: impl AsRef<Path>, input: &[f32]) -> CoelanoxResult<Vec<f32>> {
    run_container(path, input, &ContainerRunOptions::default())
}

/// Load a container from path and run inference with auto-generated dummy input (default load options).
pub fn run_container_dummy_input_simple(path: impl AsRef<Path>) -> CoelanoxResult<Vec<f32>> {
    run_container_dummy_input(path, &ContainerRunOptions::default())
}

/// Load a container from path and run inference with the given input.
/// Uses the default runtime (one container, no TTL). Returns the output tensor as f32 vector.
///
/// # Errors
/// Returns error if the file cannot be read, validation fails, or execution fails.
pub fn run_container(
    path: impl AsRef<Path>,
    input: &[f32],
    opts: &ContainerRunOptions,
) -> CoelanoxResult<Vec<f32>> {
    let name = path.as_ref().to_string_lossy().into_owned();
    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let runtime = CoelanoxRuntime::with_config(config);
    load_container_with_options(&runtime, &name, opts)?;
    runtime.execute(&name, input)
}

/// Load a container from path and run inference with auto-generated dummy input (for testing).
pub fn run_container_dummy_input(path: impl AsRef<Path>, opts: &ContainerRunOptions) -> CoelanoxResult<Vec<f32>> {
    let name = path.as_ref().to_string_lossy().into_owned();
    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let runtime = CoelanoxRuntime::with_config(config);
    load_container_with_options(&runtime, &name, opts)?;
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
