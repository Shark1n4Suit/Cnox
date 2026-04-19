//! Minimal binary: load one `.cnox` and run inference, or **`--serve`** for framed IPC (v1) on stdin/stdout.
//!
//! Usage:
//!   coelanox-run [OPTIONS] <model.cnox> [input.json]
//! Without `--serve`, if `input.json` is omitted, uses dummy input (all 1.0) and prints output.
//!
//! Trust and verification: `--trusted-key`, `--require-signature`, `--no-verify` (same semantics as `coelanox run`).

use clap::Parser;
use coelanox_ipc::{run_serve_stdio, ServeOptions};
use coelanox_orchestrator::{CoelanoxRuntime, RuntimeConfig};
use coelanox_standalone::{load_container_with_options, ContainerRunOptions};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "coelanox-run")]
#[command(
    about = "Load one .cnox and run inference, or --serve for framed binary IPC (same protocol as `coelanox serve`)"
)]
struct Cli {
    /// Path to the `.cnox` container
    model: PathBuf,
    /// Input tensor as JSON (`f32` array); ignored when using `--serve`
    input_json: Option<PathBuf>,
    /// Framed binary IPC v1 on stdin/stdout (stderr: logs and `COELANOX_SERVE_READY`)
    #[arg(long)]
    serve: bool,
    /// Backend for each inference (`--serve` only; default `auto`)
    #[arg(short, long, default_value = "auto")]
    backend: String,
    /// Trusted Ed25519 public key file (32 bytes). Alternative: `COELANOX_TRUSTED_PUBLIC_KEY`
    #[arg(long)]
    trusted_key: Option<PathBuf>,
    /// Fail if the container is signed but no trusted key is provided
    #[arg(long)]
    require_signature: bool,
    /// Skip integrity verification at load (same as `coelanox run --no-verify`)
    #[arg(long)]
    no_verify: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let path_str = cli.model.to_string_lossy().into_owned();

    let opts = ContainerRunOptions {
        trusted_key_path: cli.trusted_key.clone(),
        no_verify: cli.no_verify,
        require_signature: cli.require_signature,
    };

    if cli.serve {
        // Logs must not corrupt stdout IPC frames (match `coelanox serve`).
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .init();

        let config = RuntimeConfig::from_env_with_optional_config_file()?;
        let runtime = CoelanoxRuntime::with_config(config);
        load_container_with_options(&runtime, &path_str, &opts)?;

        let serve_opts = ServeOptions {
            container_path: path_str.as_str(),
            backend: &cli.backend,
        };
        run_serve_stdio(&runtime, &serve_opts)?;
        return Ok(());
    }

    tracing_subscriber::fmt::init();

    let input_data = if let Some(ref input_file) = cli.input_json {
        let json = std::fs::read_to_string(input_file)?;
        serde_json::from_str::<Vec<f32>>(&json)?
    } else {
        let output = coelanox_standalone::run_container_dummy_input(&path_str, &opts)?;
        println!("Output (dummy input): {:?}", output);
        return Ok(());
    };
    let output = coelanox_standalone::run_container(&path_str, &input_data, &opts)?;
    println!("Output: {:?}", output);
    Ok(())
}
