//! COELANOX CLI - Command line interface for COELANOX
//!
//! This module provides the command line interface for the COELANOX toolchain.

#![allow(clippy::too_many_arguments)]

use clap::{Parser, Subcommand};
use coelanox_core::{
    format::{
        generate_signing_keypair, header_audit_required, is_container_signed, HardwareTarget,
    },
    CoelanoxError, CoelanoxResult,
};
use coelanox_ipc::{run_serve_stdio, ServeOptions};
use coelanox_orchestrator::{
    output_node_indices_from_ir_data, record_input_snapshot, set_audit_mode, set_audit_session,
    AuditRunContext, AuditSessionConfig, AuditStrategy, CoelanoxRuntime, RuntimeConfig,
    MAX_AUDIT_SAMPLE_TOTAL,
};
use coelanox_packager::CoelanoxPackager;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use tracing::{error, info, warn};

/// True if `path` already targets `models/output/...` as the first logical directory pair.
///
/// Handles leading `./`, `../`, and `..` segments so we do not prepend `models/output/` twice
/// (e.g. `./models/output/foo` or `../models/output/foo`).
fn path_already_under_models_output(path: &Path) -> bool {
    let mut it = path.components().peekable();
    while matches!(it.peek(), Some(Component::CurDir)) {
        it.next();
    }
    while matches!(it.peek(), Some(Component::ParentDir)) {
        it.next();
    }
    match (it.next(), it.next()) {
        (Some(Component::Normal(a)), Some(Component::Normal(b))) => {
            a.to_string_lossy().eq_ignore_ascii_case("models")
                && b.to_string_lossy().eq_ignore_ascii_case("output")
        }
        _ => false,
    }
}

/// True when a load failure is due to container hash / signature verification (extra user hints).
fn should_hint_container_integrity_failure(err: &CoelanoxError) -> bool {
    err.is_container_integrity_failure()
}

/// Resolve output path: create parent dirs, default to models/output/ for relative paths, add .cnox if no extension.
fn ensure_output_path(input_path: &str, output_path: &str) -> CoelanoxResult<PathBuf> {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);

    // Relative paths (unless already under models/output) go under models/output/
    let already_in_output_dir = path_already_under_models_output(output_path);

    // If output path is relative and doesn't already start with models/output, make it relative to models/output/
    let final_output_path = if output_path.is_relative() && !already_in_output_dir {
        let mut models_output = PathBuf::from("models/output");
        models_output.push(output_path);
        models_output
    } else {
        output_path.to_path_buf()
    };

    // Ensure the output directory exists
    if let Some(parent) = final_output_path.parent() {
        fs::create_dir_all(parent).map_err(CoelanoxError::FileSystem)?;
    }

    // If no extension provided, add .cnox
    let final_path = if final_output_path.extension().is_none() {
        final_output_path.with_extension("cnox")
    } else {
        final_output_path
    };

    println!("Input:  {}", input_path.display());
    println!("Output: {}", final_path.display());

    Ok(final_path)
}

#[derive(Parser)]
#[command(name = "coelanox")]
#[command(about = "COELANOX — package, verify, and run secure AI inference containers (.cnox)")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a .cnox container from a supported model
    Package {
        /// Source model path (ONNX, bundle, or other supported input)
        #[arg(short, long)]
        input: String,

        /// Destination path for the .cnox file
        #[arg(short, long)]
        output: String,

        /// Domain hint: cv, nlp, audio, tabular, or custom (requires --input-shapes / --output-shapes)
        #[arg(short, long, default_value = "cv")]
        model_type: String,

        /// Custom models: comma-separated shapes, e.g. "[1,3,224,224],[1,1,28,28]"
        #[arg(long)]
        input_shapes: Option<String>,

        /// Custom models: comma-separated shapes, e.g. "[1,1000],[1,10]"
        #[arg(long)]
        output_shapes: Option<String>,

        /// Hardware target: cpu, gpu, edge, auto, or a backend-specific name (e.g. h100, intel-npu)
        #[arg(short, long, default_value = "auto")]
        target: String,

        /// Optimization preset: none, basic, advanced, maximum
        #[arg(long, default_value = "advanced")]
        optimization: String,

        /// Prefer demo or specialized translators (e.g. BERT) where applicable
        #[arg(long)]
        use_demo_translator: bool,

        /// Embed all backends (cannot be combined with --target or --auto)
        #[arg(long)]
        fat: bool,

        /// Auto-select backends from the host (cannot be combined with --target or --fat)
        #[arg(long)]
        auto: bool,

        /// Extra backends to include, comma-separated (e.g. amd,intel,qualcomm)
        #[arg(long)]
        backend_add: Option<String>,

        /// Omit native backend code; runtime uses scalar IR only (portable, slower)
        #[arg(long)]
        fallback_only: bool,

        /// Mark the container so inference always runs with audit logging enabled
        #[arg(long)]
        audit_required: bool,

        /// Cap workspace + scratch in MB; packaging fails if the graph needs more
        #[arg(long)]
        max_workspace: Option<u64>,

        /// External weights file (e.g. separate FP16 weights for ONNX)
        #[arg(long)]
        weights: Option<String>,

        /// Ed25519 signing key file (32-byte seed). Alternative: COELANOX_SIGNING_KEY
        #[arg(long)]
        signing_key: Option<String>,
    },

    /// Check that a .cnox file parses and is internally consistent
    Validate {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,
    },

    /// Generate an Ed25519 keypair for signing containers (secret.bin, public.bin)
    Keygen {
        /// Directory for secret.bin and public.bin (default: current directory)
        #[arg(short, long)]
        output_dir: Option<String>,
    },

    /// Verify integrity hash (and signature if present) before deployment
    Verify {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,

        /// Print additional verification detail
        #[arg(short, long)]
        verbose: bool,

        /// Trusted Ed25519 public key file (32 bytes). Alternative: COELANOX_TRUSTED_PUBLIC_KEY
        #[arg(long)]
        trusted_key: Option<String>,

        /// Error if the container is signed but no trusted key is supplied
        #[arg(long)]
        require_signature: bool,
    },

    /// Inspect manifest, sizes, and shapes for a .cnox file
    Info {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,
        /// Print convolution plan steps (debugging execution plans)
        #[arg(long)]
        dump_plan: bool,
        /// List IR operation counts; highlights Custom ops (not supported on scalar)
        #[arg(long)]
        list_ops: bool,
    },

    /// Run inference for a packaged model
    Run {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,

        /// Input tensor as JSON (f32 array); omit to use a synthetic all-ones tensor
        #[arg(short, long)]
        input: Option<String>,

        /// Path for JSON inference output
        #[arg(short, long)]
        output: String,

        /// Execution backend: auto, cpu, gpu, scalar, etc. (auto aligns with benchmark defaults)
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Per-operation audit log to stderr and optional file
        #[arg(long)]
        audit: bool,

        /// Audit log file path when audit is active (`--audit` or container `audit-required` header)
        #[arg(long)]
        audit_output: Option<String>,

        /// Audit retention: `forget` (default) — bounded RAM, append each batch to disk (JSONL for `.json`); `last` — overwrite with last batch only; `long` — session-cumulative in RAM (legacy).
        #[arg(long, default_value = "forget")]
        audit_strategy: String,

        /// Audit: leading floats per op output (capped internally).
        #[arg(long, default_value = "4")]
        audit_sample: usize,
        /// Audit: trailing floats per op output (non-overlapping with head when possible).
        #[arg(long, default_value = "0")]
        audit_tail: usize,
        /// Audit: include min/max/mean and NaN/Inf counts per op (full tensor scan up to an internal cap).
        #[arg(long)]
        audit_stats: bool,
        /// Audit: SHA-256 hex over raw `f32` bytes for inputs and each op output (when under internal cap).
        #[arg(long)]
        audit_hash: bool,
        /// Audit: only log ops that produce graph sink outputs (IR-derived).
        #[arg(long)]
        audit_final_only: bool,

        /// Audit: for `.json` output with strategy `forget`, write **pretty-printed** JSON per batch (multi-line). Default is compact JSONL (one line per batch).
        #[arg(long)]
        audit_pretty: bool,

        /// Workspace limit in MB (overrides COELANOX_MAX_WORKSPACE_MB; 0 = unlimited)
        #[arg(long)]
        max_workspace_mb: Option<u64>,

        /// Skip hash verification on load (development only)
        #[arg(long)]
        no_verify: bool,

        /// Trusted Ed25519 public key file (32 bytes). Alternative: COELANOX_TRUSTED_PUBLIC_KEY
        #[arg(long)]
        trusted_key: Option<String>,

        /// Error if the container is signed but no trusted key is supplied
        #[arg(long)]
        require_signature: bool,
    },

    /// Load a container and enter serve mode (framed binary IPC v1 on stdin/stdout; see `coelanox-ipc`)
    Serve {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,
        /// Execution backend for each inference request
        #[arg(short, long, default_value = "auto")]
        backend: String,
        /// Per-operation audit log (same semantics as `run`)
        #[arg(long)]
        audit: bool,
        /// Audit log file path when audit is active (`--audit` or container `audit-required` header; same as `run`)
        #[arg(long)]
        audit_output: Option<String>,

        /// Audit retention: `forget` (default) — bounded RAM, append each batch (JSONL for `.json`); `last` — overwrite with last batch; `long` — session-cumulative in RAM (legacy).
        #[arg(long, default_value = "forget")]
        audit_strategy: String,
        /// Audit: leading floats per op output (capped internally).
        #[arg(long, default_value = "4")]
        audit_sample: usize,
        /// Audit: trailing floats per op output (non-overlapping with head when possible).
        #[arg(long, default_value = "0")]
        audit_tail: usize,
        /// Audit: include min/max/mean and NaN/Inf counts per op (full tensor scan up to an internal cap).
        #[arg(long)]
        audit_stats: bool,
        /// Audit: SHA-256 hex over raw `f32` bytes for inputs and each op output (when under internal cap).
        #[arg(long)]
        audit_hash: bool,
        /// Audit: only log ops that produce graph sink outputs (IR-derived).
        #[arg(long)]
        audit_final_only: bool,

        /// Audit: for `.json` output with strategy `forget`, write **pretty-printed** JSON per batch (multi-line). Default is compact JSONL (one line per batch).
        #[arg(long)]
        audit_pretty: bool,

        /// Workspace limit in MB (overrides COELANOX_MAX_WORKSPACE_MB; 0 = unlimited)
        #[arg(long)]
        max_workspace_mb: Option<u64>,
        /// Skip hash verification on load (development only)
        #[arg(long)]
        no_verify: bool,
        /// Trusted Ed25519 public key file (32 bytes). Alternative: COELANOX_TRUSTED_PUBLIC_KEY
        #[arg(long)]
        trusted_key: Option<String>,
        /// Error if the container is signed but no trusted key is supplied
        #[arg(long)]
        require_signature: bool,
    },

    /// Measure latency and throughput for a container
    Benchmark {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,
        /// Iteration count for timed runs (default: 100)
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        /// Long-run sample size; overrides --iterations when set
        #[arg(long)]
        loop_iterations: Option<usize>,
        /// Backend: auto, scalar, cpu, simd, etc.
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output: String,
        /// Workspace limit in MB (overrides COELANOX_MAX_WORKSPACE_MB; 0 = unlimited)
        #[arg(long)]
        max_workspace_mb: Option<u64>,
        /// Skip hash verification on load
        #[arg(long)]
        no_verify: bool,

        /// Trusted Ed25519 public key file (32 bytes). Alternative: COELANOX_TRUSTED_PUBLIC_KEY
        #[arg(long)]
        trusted_key: Option<String>,

        /// Error if the container is signed but no trusted key is supplied
        #[arg(long)]
        require_signature: bool,
    },

    /// Print host CPU, memory, GPU, and optional toolchain detection
    Env,

    /// Show how to remove the CLI and optionally delete ./coelanox_audit
    Uninstall {
        /// Remove ./coelanox_audit under the current working directory
        #[arg(long)]
        purge_audit: bool,
    },

    /// Run inference with extra diagnostics (optional GDB)
    Debug {
        /// Path to the .cnox file
        #[arg(short, long)]
        file: String,

        /// Input tensor as JSON (f32 array); omit for synthetic input
        #[arg(short, long)]
        input: Option<String>,

        /// JSON path for structured debug output
        #[arg(short, long, default_value = "debug_output.json")]
        output: String,

        /// Run under GDB with a generated script (requires GDB on PATH)
        #[arg(long)]
        gdb: bool,

        /// On Unix, install a minimal SIGSEGV handler when not using GDB
        #[arg(long, default_value = "true")]
        dump_registers: bool,

        /// Reserved for future trace depth limits
        #[arg(short, long, default_value = "10")]
        trace_limit: usize,

        /// Trusted Ed25519 public key file (32 bytes). Alternative: COELANOX_TRUSTED_PUBLIC_KEY
        #[arg(long)]
        trusted_key: Option<String>,

        /// Fail if the container is signed but no trusted key is provided
        #[arg(long)]
        require_signature: bool,

        /// Skip integrity verification at load (same as `coelanox run --no-verify`)
        #[arg(long)]
        no_verify: bool,
    },

    /// Copy a deployable runtime binary (same architecture as this CLI)
    Extract {
        /// Output path for the copied binary (default: `coelanox` for full CLI, or use with `--minimal` → e.g. `coelanox-run`)
        #[arg(short, long, default_value = "coelanox")]
        output: String,

        /// Copy minimal `coelanox-run` from the same directory as this executable (requires `cargo build --release -p coelanox-standalone` so both binaries exist in `target/release/`)
        #[arg(long)]
        minimal: bool,

        /// Request ARM64 output (not implemented; same-arch only for now)
        #[arg(long)]
        arm: bool,

        /// Rust target triple for cross-build (not implemented; use CI artifacts)
        #[arg(long)]
        target: Option<String>,
    },
}

fn main() -> CoelanoxResult<()> {
    // Log to stderr only so stdout stays binary-clean for `coelanox serve` IPC frames.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Package {
            input,
            output,
            model_type,
            input_shapes,
            output_shapes,
            target,
            optimization,
            use_demo_translator,
            fat,
            auto,
            backend_add,
            fallback_only,
            audit_required,
            max_workspace,
            weights,
            signing_key,
        } => package_command(
            &input,
            &output,
            &model_type,
            input_shapes.as_deref(),
            output_shapes.as_deref(),
            &target,
            &optimization,
            use_demo_translator,
            fat,
            auto,
            backend_add.as_deref(),
            fallback_only,
            audit_required,
            max_workspace,
            weights.as_deref(),
            signing_key.as_deref(),
        ),
        Commands::Validate { file } => validate_command(&file),
        Commands::Keygen { output_dir } => keygen_command(output_dir.as_deref()),
        Commands::Verify {
            file,
            verbose,
            trusted_key,
            require_signature,
        } => verify_command(&file, verbose, trusted_key.as_deref(), require_signature),
        Commands::Info {
            file,
            dump_plan,
            list_ops,
        } => info_command(&file, dump_plan, list_ops),
        Commands::Run {
            file,
            input,
            output,
            backend,
            audit,
            audit_output,
            audit_strategy,
            audit_sample,
            audit_tail,
            audit_stats,
            audit_hash,
            audit_final_only,
            audit_pretty,
            max_workspace_mb,
            no_verify,
            trusted_key,
            require_signature,
        } => {
            let audit_strategy = parse_audit_strategy(&audit_strategy)?;
            run_command(
                &file,
                input.as_deref(),
                &output,
                &backend,
                audit,
                audit_output.as_deref(),
                audit_strategy,
                audit_sample,
                audit_tail,
                audit_stats,
                audit_hash,
                audit_final_only,
                audit_pretty,
                max_workspace_mb,
                no_verify,
                trusted_key.as_deref(),
                require_signature,
            )
        }
        Commands::Serve {
            file,
            backend,
            audit,
            audit_output,
            audit_strategy,
            audit_sample,
            audit_tail,
            audit_stats,
            audit_hash,
            audit_final_only,
            audit_pretty,
            max_workspace_mb,
            no_verify,
            trusted_key,
            require_signature,
        } => {
            let audit_strategy = parse_audit_strategy(&audit_strategy)?;
            serve_command(
                &file,
                &backend,
                audit,
                audit_output.as_deref(),
                audit_strategy,
                audit_sample,
                audit_tail,
                audit_stats,
                audit_hash,
                audit_final_only,
                audit_pretty,
                max_workspace_mb,
                no_verify,
                trusted_key.as_deref(),
                require_signature,
            )
        }
        Commands::Benchmark {
            file,
            iterations,
            loop_iterations,
            backend,
            output,
            max_workspace_mb,
            no_verify,
            trusted_key,
            require_signature,
        } => benchmark_command(
            &file,
            iterations,
            loop_iterations,
            &backend,
            &output,
            max_workspace_mb,
            no_verify,
            trusted_key.as_deref(),
            require_signature,
        ),
        Commands::Env => env_command(),
        Commands::Uninstall { purge_audit } => uninstall_command(purge_audit),
        Commands::Debug {
            file,
            input,
            output,
            gdb,
            dump_registers,
            trace_limit,
            trusted_key,
            require_signature,
            no_verify,
        } => debug_command(
            &file,
            input.as_deref(),
            &output,
            gdb,
            dump_registers,
            trace_limit,
            trusted_key.as_deref(),
            require_signature,
            no_verify,
        ),
        Commands::Extract {
            output,
            minimal,
            arm,
            target,
        } => extract_command(&output, minimal, arm, target.as_deref()),
    }
}

fn package_command(
    input: &str,
    output: &str,
    model_type: &str,
    input_shapes: Option<&str>,
    output_shapes: Option<&str>,
    target: &str,
    _optimization: &str,
    use_demo_translator: bool,
    fat: bool,
    auto: bool,
    backend_add: Option<&str>,
    fallback_only: bool,
    audit_required: bool,
    max_workspace: Option<u64>,
    weights: Option<&str>,
    signing_key: Option<&str>,
) -> CoelanoxResult<()> {
    info!(
        "Packaging {} into COELANOX container: {}",
        model_type, output
    );

    // Ensure proper output path and directory structure
    let final_output_path = ensure_output_path(input, output)?;

    // Validate model type
    let valid_types = ["cv", "nlp", "audio", "tabular", "custom"];
    if !valid_types.contains(&model_type.to_lowercase().as_str()) {
        return Err(CoelanoxError::format(format!(
            "Unknown model type '{}'. Expected one of: {}.",
            model_type,
            valid_types.join(", ")
        )));
    }

    // Validate custom shapes if provided
    if model_type.to_lowercase() == "custom" {
        if input_shapes.is_none() || output_shapes.is_none() {
            return Err(CoelanoxError::format(
                "Custom model type requires --input-shapes and --output-shapes",
            ));
        }

        // Validate shape format
        if let Some(input_shapes_str) = input_shapes {
            if !is_valid_shape_format(input_shapes_str) {
                return Err(CoelanoxError::format(format!(
                    "Invalid --input-shapes '{}'. Example: '[1,3,224,224],[1,1,28,28]'",
                    input_shapes_str
                )));
            }
        }

        if let Some(output_shapes_str) = output_shapes {
            if !is_valid_shape_format(output_shapes_str) {
                return Err(CoelanoxError::format(format!(
                    "Invalid --output-shapes '{}'. Example: '[1,1000],[1,10]'",
                    output_shapes_str
                )));
            }
        }

        info!(
            "Custom shapes: input={:?}, output={:?}",
            input_shapes, output_shapes
        );
    }

    let packager = CoelanoxPackager::new()?;
    let env = packager.get_environment();
    info!(
        "Host: {} {} | {} {} | {} logical CPUs | {} GB RAM",
        env.os, env.architecture, env.cpu_vendor, env.cpu_model, env.cpu_cores, env.memory_gb
    );
    if let Some(gpu) = &env.gpu_info {
        info!("GPU: {} {} ({} GB)", gpu.vendor, gpu.model, gpu.memory_gb);
    }
    info!("SIMD: {:?}", env.simd_capabilities);

    // Validate flag combinations
    if fat && auto {
        return Err(CoelanoxError::format(
            "--fat and --auto cannot be used together",
        ));
    }
    if fat && target != "auto" {
        return Err(CoelanoxError::format(
            "--fat cannot be combined with --target; use one or the other",
        ));
    }
    if auto && target != "auto" {
        return Err(CoelanoxError::format(
            "--auto cannot be combined with --target; use one or the other",
        ));
    }

    // Determine target hardware
    // If --fat or --auto, we'll handle selection in packager
    // Otherwise, parse --target (supports both old values and new hardware-specific values)
    let hardware_target = if fat || auto {
        // Packager overrides based on --fat/--auto; this is the initial env hint.
        env.recommended_target()
    } else {
        match target.to_lowercase().as_str() {
            "cpu" | "simd" => HardwareTarget::Cpu,
            "gpu" => HardwareTarget::Gpu,
            "edge" => HardwareTarget::Edge,
            "auto" => {
                let recommended = env.recommended_target();
                info!("Target (auto): {:?}", recommended);
                recommended
            }
            // Hardware-specific targets
            "h100" | "a100" | "v100" => HardwareTarget::Gpu, // NVIDIA GPUs
            "amd-gpu" | "amd" => HardwareTarget::Gpu,        // AMD GPU
            "intel-npu" | "intel" => HardwareTarget::NPU,    // Intel NPU
            "qualcomm" => HardwareTarget::NPU,               // Qualcomm NPU
            _ => {
                // Try as custom target
                if target.contains('-') || target.chars().all(|c| c.is_alphanumeric()) {
                    HardwareTarget::Custom(target.to_string())
                } else {
                    warn!("Unrecognized target '{}'; using host default", target);
                    env.recommended_target()
                }
            }
        }
    };

    info!("Hardware target: {:?}", hardware_target);

    let weights_path = weights.map(std::path::Path::new);
    let container = packager.package(
        std::path::Path::new(input),
        hardware_target,
        target, // Pass original target string for backend metadata naming
        model_type,
        input_shapes,
        output_shapes,
        use_demo_translator,
        fat,
        auto,
        backend_add,
        fallback_only,
        audit_required,
        max_workspace,
        weights_path,
    )?;

    // Resolve signing key: --signing-key or COELANOX_SIGNING_KEY (path to 32-byte file)
    let signing_key_path = signing_key
        .map(String::from)
        .or_else(|| std::env::var("COELANOX_SIGNING_KEY").ok());
    let signing_key_bytes = if let Some(ref path) = signing_key_path {
        let key = std::fs::read(path).map_err(|e| {
            CoelanoxError::format(format!("Could not read signing key {}: {}", path, e))
        })?;
        if key.len() != 32 {
            return Err(CoelanoxError::format(format!(
                "Signing key must be 32 bytes (Ed25519 seed); got {} bytes",
                key.len()
            )));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&key);
        Some(arr)
    } else {
        None
    };

    // Save container to binary file (with optional signing)
    container.write_binary_file_with_signing(&final_output_path, signing_key_bytes.as_ref())?;

    info!(
        "Wrote {} ({} bytes)",
        final_output_path.display(),
        container.total_size()
    );
    let input_size = std::fs::metadata(input)?.len();
    let compression_ratio = input_size as f64 / container.total_size() as f64;
    info!(
        "Source file size vs container: {:.1}x (reference only)",
        compression_ratio
    );

    Ok(())
}

fn validate_command(file: &str) -> CoelanoxResult<()> {
    info!("Validating {}", file);

    let container =
        coelanox_core::format::CoelanoxContainer::read_binary_file(std::path::Path::new(file))?;

    println!("Validation succeeded.");
    println!("CLI version: {}", env!("CARGO_PKG_VERSION"));
    println!("Model name: {}", container.manifest.name);
    println!("Format: {:?}", container.manifest.format);
    println!("Size: {} bytes", container.total_size());

    Ok(())
}

fn keygen_command(output_dir: Option<&str>) -> CoelanoxResult<()> {
    let (secret, public) = generate_signing_keypair();
    let base = output_dir.unwrap_or(".");
    let secret_path = std::path::Path::new(base).join("secret.bin");
    let public_path = std::path::Path::new(base).join("public.bin");
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir).map_err(CoelanoxError::FileSystem)?;
    }
    std::fs::write(&secret_path, secret).map_err(CoelanoxError::FileSystem)?;
    std::fs::write(&public_path, public).map_err(CoelanoxError::FileSystem)?;
    println!("Ed25519 signing keypair written.");
    println!("  Private key (confidential): {}", secret_path.display());
    println!(
        "  Public key (for --trusted-key): {}",
        public_path.display()
    );
    Ok(())
}

fn verify_command(
    file: &str,
    verbose: bool,
    trusted_key: Option<&str>,
    require_signature: bool,
) -> CoelanoxResult<()> {
    if verbose {
        info!("Verifying {}", file);
    }

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set."
            ));
        }
    }

    let trusted_key_bytes = resolve_trusted_key(trusted_key)?;

    match coelanox_core::format::verify_security_hash_with_trusted_key(
        std::path::Path::new(file),
        trusted_key_bytes.as_ref(),
    ) {
        Ok(()) => {
            println!("Integrity verification succeeded.");
            if verbose {
                println!("Cryptographic hash matches; container matches the packaged artifact.");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Verification failed.");
            eprintln!("{}", e);
            eprintln!("Do not deploy or run this file until the issue is resolved.");
            Err(e)
        }
    }
}

/// Resolve trusted public key from --trusted-key or COELANOX_TRUSTED_PUBLIC_KEY (path to 32-byte file).
fn resolve_trusted_key(path: Option<&str>) -> CoelanoxResult<Option<[u8; 32]>> {
    let p = path
        .map(String::from)
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

fn info_command(file: &str, dump_plan: bool, list_ops: bool) -> CoelanoxResult<()> {
    info!("Inspecting {}", file);

    let container =
        coelanox_core::format::CoelanoxContainer::read_binary_file(std::path::Path::new(file))?;

    if list_ops {
        use coelanox_core::ir::{deserialize_optimized_ir, OpType};
        let ir = deserialize_optimized_ir(&container.ir_data)?;
        let mut custom_ops: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut all_types: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for node in &ir.optimized_nodes {
            let key = format!("{:?}", node.op_type);
            *all_types.entry(key.clone()).or_insert(0) += 1;
            if let OpType::Custom(name) = &node.op_type {
                *custom_ops.entry(name.clone()).or_insert(0) += 1;
            }
        }
        println!(
            "--- IR operation types ({} nodes) ---",
            ir.optimized_nodes.len()
        );
        let mut types: Vec<_> = all_types.into_iter().collect();
        types.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, count) in &types {
            println!("  {}: {}", op, count);
        }
        if !custom_ops.is_empty() {
            println!();
            println!("--- Custom operations (not supported on scalar backend) ---");
            let mut customs: Vec<_> = custom_ops.into_iter().collect();
            customs.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, count) in &customs {
                println!("  {}: {} occurrence(s)", name, count);
            }
        }
        return Ok(());
    }

    if dump_plan {
        if let Some(ref plan) = container.execution_plan {
            let steps = coelanox_executor::dump_plan_conv_steps(plan)?;
            println!("--- Execution plan: convolution steps (op_id=30) ---");
            println!(
                "{:<8} {:>12} {:>12} {:>12} {:>14}  (scratch MB)",
                "step", "in_len", "out_len", "w_len", "scratch_el"
            );
            for (step_ix, in_len, out_len, w_len, scratch) in &steps {
                let mb = (*scratch as u64 * 4).div_ceil(1024 * 1024);
                println!(
                    "{:<8} {:>12} {:>12} {:>12} {:>14}  ~{}",
                    step_ix, in_len, out_len, w_len, scratch, mb
                );
            }
            let max_scratch = steps
                .iter()
                .map(|(_, _, _, _, s)| s)
                .max()
                .copied()
                .unwrap_or(0);
            println!();
            println!(
                "Conv steps: {}  max_scratch_elements: {}  (~{} MB)",
                steps.len(),
                max_scratch,
                (max_scratch as u64 * 4).div_ceil(1024 * 1024)
            );
            return Ok(());
        } else {
            return Err(CoelanoxError::Runtime(
                "This container has no embedded execution plan; omit --dump-plan or rebuild with a plan.".to_string(),
            ));
        }
    }

    println!("--- COELANOX container ---");
    println!("File: {}", file);
    println!("Format version: {}", env!("CARGO_PKG_VERSION"));
    println!("Status: valid");
    println!();
    println!("--- Model ---");
    println!("Name: {}", container.manifest.name);
    println!("Format: {:?}", container.manifest.format);
    println!("Created: {}", container.manifest.created_at);
    println!("Modified: {}", container.manifest.modified_at);
    println!();
    println!("--- Inputs / outputs ---");
    println!("Input shapes: {:?}", container.manifest.input_shapes);
    println!("Output shapes: {:?}", container.manifest.output_shapes);
    println!();
    println!("--- Layout ---");
    println!("Total size: {} bytes", container.total_size());
    println!(
        "Header size: {} bytes",
        coelanox_core::format::COELANOX_HEADER_SIZE
    );
    println!(
        "Manifest size: ~{} bytes",
        serde_json::to_string(&container.manifest)
            .unwrap_or_default()
            .len()
    );
    println!(
        "Machine code size: {} bytes",
        container.optimized_code.len()
    );
    println!(
        "Compressed weights size: {} bytes",
        container.compressed_weights_len()
    );
    println!("IR data size: {} bytes", container.ir_data.len());
    if let Some(elements) = container.manifest.workspace_required_elements {
        let mb = (elements * 4).div_ceil(1024 * 1024);
        println!("Workspace required: {} elements (~{} MB)", elements, mb);
    }
    println!();
    println!("--- Compression ---");
    println!("Ratio vs source model: not available (source size not recorded in manifest)");

    Ok(())
}

/// Default audit output path when container has audit-required flag and user did not set --audit-output:
/// coelanox_audit/<container_stem>_<unix_secs>.json in current directory.
fn parse_audit_strategy(s: &str) -> CoelanoxResult<AuditStrategy> {
    s.parse().map_err(|e: String| CoelanoxError::format(&e))
}

fn default_audit_path(container_file: &str) -> PathBuf {
    let stem = Path::new(container_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("container");
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let dir = PathBuf::from("coelanox_audit");
    let _ = fs::create_dir_all(&dir);
    dir.join(format!("{}_{}.json", stem, secs))
}

/// SHA-256 of file contents (hex), for audit run metadata.
fn sha256_hex_file(path: &Path) -> CoelanoxResult<String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(CoelanoxError::FileSystem)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(CoelanoxError::FileSystem)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect())
}

fn make_audit_session(
    strategy: AuditStrategy,
    sample: usize,
    tail: usize,
    stats: bool,
    hash: bool,
    final_only: bool,
    pretty: bool,
    container_path: &str,
    container_sha256: Option<String>,
    backend: &str,
    ir_data: Option<&[u8]>,
) -> CoelanoxResult<AuditSessionConfig> {
    let mut output_node_indices = Vec::new();
    if final_only {
        if let Some(data) = ir_data {
            output_node_indices = output_node_indices_from_ir_data(data)?;
        }
    }
    let sample_head = sample.min(MAX_AUDIT_SAMPLE_TOTAL);
    let sample_tail = tail.min(MAX_AUDIT_SAMPLE_TOTAL);
    let run_id = uuid::Uuid::new_v4().to_string();
    let started_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Ok(AuditSessionConfig {
        strategy,
        sample_head,
        sample_tail,
        include_stats: stats,
        include_hash: hash,
        final_output_only: final_only,
        output_node_indices,
        run: Some(AuditRunContext {
            run_id,
            started_at_unix_ms,
            container_path: container_path.to_string(),
            container_sha256_hex: container_sha256,
            backend: backend.to_string(),
        }),
        pretty_json: pretty,
    })
}

fn run_command(
    file: &str,
    input: Option<&str>,
    output: &str,
    backend: &str,
    audit: bool,
    audit_output: Option<&str>,
    audit_strategy: AuditStrategy,
    audit_sample: usize,
    audit_tail: usize,
    audit_stats: bool,
    audit_hash: bool,
    audit_final_only: bool,
    audit_pretty: bool,
    max_workspace_mb: Option<u64>,
    no_verify: bool,
    trusted_key: Option<&str>,
    require_signature: bool,
) -> CoelanoxResult<()> {
    let input_desc = input
        .map(|p| format!("input file {}", p))
        .unwrap_or_else(|| "synthetic input (all ones, manifest-shaped)".to_string());
    info!(
        "Inference: {} -> {} (backend: {})",
        input_desc, output, backend
    );
    let resolved_workspace_mb = max_workspace_mb.or_else(|| {
        std::env::var("COELANOX_MAX_WORKSPACE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
    });

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set."
            ));
        }
    }

    // Load runtime (config from COELANOX_CONFIG_FILE or COELANOX_* env vars)
    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let runtime = CoelanoxRuntime::with_config(config);
    if let Some(mb) = resolved_workspace_mb {
        runtime.set_max_workspace_mb(mb);
    }
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key)?;
    }
    if no_verify {
        info!("Loading container (integrity verification skipped per --no-verify)");
        runtime.load_container_no_verify(file)?;
        info!("Container loaded.");
    } else {
        info!("Loading container with integrity verification");
        runtime.load_container(file).inspect_err(|e| {
            if should_hint_container_integrity_failure(e) {
                eprintln!("Load aborted: integrity verification failed.");
                eprintln!("The file may be corrupted, truncated, or modified.");
            }
        })?;
        info!("Container loaded and verified.");
    }

    // If container has audit-required flag, always run with audit (compliance). Use --audit-output if set, else default folder.
    let container = runtime.get_container_info(file)?;
    let audit_required_by_container = header_audit_required(container.header.flags);
    let audit_path: Option<PathBuf> = if audit_required_by_container {
        Some(
            audit_output
                .map(PathBuf::from)
                .unwrap_or_else(|| default_audit_path(file)),
        )
    } else if audit {
        // Match audit-required behavior: --audit without --audit-output still writes under coelanox_audit/.
        Some(
            audit_output
                .map(PathBuf::from)
                .unwrap_or_else(|| default_audit_path(file)),
        )
    } else {
        None
    };
    if audit || audit_required_by_container {
        if audit_required_by_container && audit_output.is_none() {
            if let Some(ref p) = audit_path {
                info!(
                    "Audit logging required by container; writing to {}",
                    p.display()
                );
            }
        }
        if audit && !audit_required_by_container && audit_output.is_none() {
            if let Some(ref p) = audit_path {
                info!("Audit logging enabled; writing to {}", p.display());
            }
        }
        let container_sha = sha256_hex_file(Path::new(file)).ok();
        let session = make_audit_session(
            audit_strategy,
            audit_sample,
            audit_tail,
            audit_stats,
            audit_hash,
            audit_final_only,
            audit_pretty,
            file,
            container_sha,
            backend,
            Some(container.ir_data.as_slice()),
        )?;
        set_audit_session(true, audit_path.as_deref().map(Path::new), session);
    }

    // Load input data
    let input_data = if let Some(input_file) = input {
        if input_file.ends_with(".json") {
            // Load from JSON file
            let json_str =
                std::fs::read_to_string(input_file).map_err(CoelanoxError::FileSystem)?;
            serde_json::from_str::<Vec<f32>>(&json_str)
                .map_err(|e| CoelanoxError::format(format!("Invalid input JSON: {}", e)))?
        } else {
            // Generate dummy input data
            let container = runtime.get_container_info(file)?;
            let input_size: usize = container
                .manifest
                .input_shapes
                .iter()
                .map(|shape| shape.iter().product::<usize>())
                .sum();
            vec![1.0f32; input_size]
        }
    } else {
        // Generate dummy input data
        let container = runtime.get_container_info(file)?;
        let input_size: usize = container
            .manifest
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        vec![1.0f32; input_size]
    };

    record_input_snapshot(&input_data);
    // Execute inference with backend selection (backend manager chooses execution path)
    let result = runtime.execute_with_backend(file, &input_data, backend);
    if audit || audit_required_by_container {
        set_audit_mode(false, None, AuditStrategy::Forget);
    }
    let result = result?;

    // Save output
    let output_json = serde_json::to_string_pretty(&result)
        .map_err(|e| CoelanoxError::format(format!("Could not write result JSON: {}", e)))?;

    std::fs::write(output, output_json).map_err(CoelanoxError::FileSystem)?;

    info!("Inference finished.");
    println!("Input elements:   {}", input_data.len());
    println!("Output elements:  {}", result.len());
    println!(
        "Mean execution:   {} μs",
        runtime.get_stats().avg_execution_time
    );
    println!("Result written to: {}", output);

    Ok(())
}

/// Load container once, optional audit, then run the IPC request loop (`coelanox_ipc::run_serve_stdio`).
fn serve_command(
    file: &str,
    backend: &str,
    audit: bool,
    audit_output: Option<&str>,
    audit_strategy: AuditStrategy,
    audit_sample: usize,
    audit_tail: usize,
    audit_stats: bool,
    audit_hash: bool,
    audit_final_only: bool,
    audit_pretty: bool,
    max_workspace_mb: Option<u64>,
    no_verify: bool,
    trusted_key: Option<&str>,
    require_signature: bool,
) -> CoelanoxResult<()> {
    info!("Serve mode: {} (backend: {})", file, backend);
    let resolved_workspace_mb = max_workspace_mb.or_else(|| {
        std::env::var("COELANOX_MAX_WORKSPACE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
    });

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set.",
            ));
        }
    }

    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let runtime = CoelanoxRuntime::with_config(config);
    if let Some(mb) = resolved_workspace_mb {
        runtime.set_max_workspace_mb(mb);
    }
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key)?;
    }
    if no_verify {
        info!("Loading container (integrity verification skipped per --no-verify)");
        runtime.load_container_no_verify(file)?;
        info!("Container loaded.");
    } else {
        info!("Loading container with integrity verification");
        runtime.load_container(file).inspect_err(|e| {
            if should_hint_container_integrity_failure(e) {
                eprintln!("Load aborted: integrity verification failed.");
                eprintln!("The file may be corrupted, truncated, or modified.");
            }
        })?;
        info!("Container loaded and verified.");
    }

    let container = runtime.get_container_info(file)?;
    let audit_required_by_container = header_audit_required(container.header.flags);
    let audit_path: Option<PathBuf> = if audit_required_by_container {
        Some(
            audit_output
                .map(PathBuf::from)
                .unwrap_or_else(|| default_audit_path(file)),
        )
    } else if audit {
        Some(
            audit_output
                .map(PathBuf::from)
                .unwrap_or_else(|| default_audit_path(file)),
        )
    } else {
        None
    };
    if audit || audit_required_by_container {
        if audit_required_by_container && audit_output.is_none() {
            if let Some(ref p) = audit_path {
                info!(
                    "Audit logging required by container; writing to {}",
                    p.display()
                );
            }
        }
        if audit && !audit_required_by_container && audit_output.is_none() {
            if let Some(ref p) = audit_path {
                info!("Audit logging enabled; writing to {}", p.display());
            }
        }
        let container_sha = sha256_hex_file(Path::new(file)).ok();
        let session = make_audit_session(
            audit_strategy,
            audit_sample,
            audit_tail,
            audit_stats,
            audit_hash,
            audit_final_only,
            audit_pretty,
            file,
            container_sha,
            backend,
            Some(container.ir_data.as_slice()),
        )?;
        set_audit_session(true, audit_path.as_deref().map(Path::new), session);
    }

    let serve_opts = ServeOptions {
        container_path: file,
        backend,
    };
    let result = run_serve_stdio(&runtime, &serve_opts);

    if audit || audit_required_by_container {
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    result
}

fn benchmark_command(
    file: &str,
    iterations: usize,
    loop_iterations: Option<usize>,
    backend: &str,
    output_format: &str,
    max_workspace_mb: Option<u64>,
    no_verify: bool,
    trusted_key: Option<&str>,
    require_signature: bool,
) -> CoelanoxResult<()> {
    let effective_iterations = loop_iterations.unwrap_or(iterations);
    let resolved_workspace_mb = max_workspace_mb.or_else(|| {
        std::env::var("COELANOX_MAX_WORKSPACE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
    });
    let is_long_run = effective_iterations >= 1000;
    info!(
        "Benchmark {} — {} iterations, backend {}{}",
        file,
        effective_iterations,
        backend,
        if is_long_run {
            ", long-run stability sample"
        } else {
            ""
        }
    );

    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let mut runtime = CoelanoxRuntime::with_config(config);
    if let Some(mb) = resolved_workspace_mb {
        runtime.set_max_workspace_mb(mb);
    }
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key)?;
    }
    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set."
            ));
        }
    }
    let load_start = std::time::Instant::now();
    if no_verify {
        runtime.load_container_no_verify(file)?;
    } else {
        runtime.load_container(file)?;
    }
    let load_time_ms = load_start.elapsed().as_millis() as u64;

    let result = runtime.benchmark(file, effective_iterations, Some(load_time_ms), backend)?;

    if output_format.eq_ignore_ascii_case("json") {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
        return Ok(());
    }

    // Human-readable text output
    println!("--- Benchmark ---");
    println!("Container: {}", result.container_name);
    println!(
        "Iterations: {} (warm-up runs: {})",
        result.iterations, result.warm_up_runs
    );
    if let Some(load_ms) = result.load_time_ms {
        println!("Load to ready: {} ms", load_ms);
    }
    println!();
    println!("--- Latency (microseconds) ---");
    println!("  Average:  {:.2}", result.avg_time);
    println!("  Min:      {}", result.min_time);
    println!("  Max:      {}", result.max_time);
    println!("  p50:      {}", result.p50_us);
    println!("  p95:      {}", result.p95_us);
    println!("  p99:      {}", result.p99_us);
    println!("  Std dev:  {:.2}", result.std_dev_us);
    println!(
        "  True cold (first run, no warm-up): {} μs  (comparable to `run` execution time)",
        result.true_cold_time_us
    );
    if let Some(cold) = result.cold_time_us {
        println!("  First measured (after warm-up): {} μs", cold);
    }
    if let Some(warm) = result.warm_avg_us {
        println!("  Warm average (after first measured): {} μs", warm);
    }
    println!();
    println!("--- Throughput ---");
    let throughput = 1_000_000.0 / result.avg_time as f64;
    println!("  Sustained: {:.0} inferences/s", throughput);
    if result.min_time > 0 {
        println!(
            "  Peak (limited by min latency): {:.0} inferences/s",
            1_000_000.0 / result.min_time as f64
        );
    }
    println!();
    println!("--- Memory ---");
    println!("  Workspace (elements): {}", result.workspace_elements);
    println!("  Weights (decompressed): {} bytes", result.weights_bytes);
    println!(
        "  Executable memory: {} bytes",
        result.executable_memory_bytes
    );
    if let Some(rss) = result.rss_after_bytes {
        println!(
            "  RSS after benchmark: {} bytes ({:.2} MiB)",
            rss,
            rss as f64 / (1024.0 * 1024.0)
        );
    }
    println!();
    println!("--- Timing breakdown ---");
    println!("  Load time: {} ms", result.load_time_ms.unwrap_or(0));
    println!("  Decompress (one-time): {} ms", result.decompress_time_ms);
    println!("  Backend: {}", result.backend_used);
    if is_long_run {
        if let (Some(first), Some(last), Some(drift)) = (
            result.stability_first_10pct_avg_us,
            result.stability_last_10pct_avg_us,
            result.stability_drift_us,
        ) {
            println!();
            println!("--- Long-run stability ---");
            println!("  First 10% mean: {} μs", first);
            println!("  Last 10% mean:  {} μs", last);
            println!(
                "  Drift (last − first): {} μs ({})",
                drift,
                if drift > 0 { "slowdown" } else { "improvement" }
            );
        }
    }
    Ok(())
}

fn env_command() -> CoelanoxResult<()> {
    info!("Collecting host environment");

    let env = coelanox_core::utils::SystemEnvironment::detect()?;

    println!("--- COELANOX CLI ---");
    println!("CLI version: {}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("--- Host ---");
    println!("OS: {}", env.os);
    println!("Architecture: {}", env.architecture);
    println!("CPU vendor: {}", env.cpu_vendor);
    println!("CPU model: {}", env.cpu_model);
    println!("Logical CPUs: {}", env.cpu_cores);
    println!("Memory: {} GB (approx.)", env.memory_gb);
    println!();

    if let Some(ref gpu) = env.gpu_info {
        println!("--- GPU ---");
        println!("Vendor: {}", gpu.vendor);
        println!("Model: {}", gpu.model);
        println!("Memory: {} GB", gpu.memory_gb);
        if let Some(cc) = &gpu.compute_capability {
            println!("Compute capability: {}", cc);
        }
        if let Some(driver) = &gpu.driver_version {
            println!("Driver: {}", driver);
        }
        println!();
    }

    println!("--- SIMD ---");
    for capability in &env.simd_capabilities {
        println!("  {:?}", capability);
    }
    println!();

    println!("--- Optional toolchains (environment) ---");
    let mut caps: Vec<_> = env.capabilities.iter().collect();
    caps.sort_by(|a, b| a.0.cmp(b.0));
    for (capability, available) in caps {
        println!(
            "  {}: {}",
            capability,
            if *available {
                "detected"
            } else {
                "not detected"
            }
        );
    }
    println!();

    println!("--- Heuristic suggestions (packaging) ---");
    let recommended_target = env.recommended_target();
    let recommended_optimization = env.recommended_optimization();
    println!("Target: {:?}", recommended_target);
    println!("Optimization level: {:?}", recommended_optimization);

    Ok(())
}

/// Cannot delete the running executable; print removal steps and optional ./coelanox_audit cleanup.
fn uninstall_command(purge_audit: bool) -> CoelanoxResult<()> {
    let exe = std::env::current_exe().map_err(CoelanoxError::FileSystem)?;
    println!("Uninstall COELANOX CLI");
    println!();
    println!("Executable location (remove this file or its containing folder):");
    println!("  {}", exe.display());
    println!();
    println!("Common options:");
    println!("  - Installed with Cargo:  cargo uninstall coelanox-cli");
    println!("  - Manual install: delete the path above, then remove that directory from PATH if you added it.");
    #[cfg(windows)]
    println!("  - Windows PATH: Settings → System → About → Advanced system settings → Environment Variables → Path.");
    #[cfg(unix)]
    println!(
        "  - Unix PATH: adjust ~/.profile, ~/.bashrc, or files under /etc/profile.d/ as needed."
    );
    println!();
    println!("The running process cannot delete its own binary; exit, then remove the file.");
    if purge_audit {
        let audit = Path::new("coelanox_audit");
        if audit.is_dir() {
            fs::remove_dir_all(audit).map_err(CoelanoxError::FileSystem)?;
            println!("Removed ./coelanox_audit (local audit log directory).");
        } else {
            println!("No ./coelanox_audit in the current directory; nothing to remove.");
        }
    }
    Ok(())
}

/// Extract: deploy a runnable next to `.cnox` files.
///
/// - **Default:** copy this **`coelanox` CLI** (full toolchain + `serve` IPC + everything else).
/// - **`--minimal`:** copy **`coelanox-run`** from the same directory as this binary (must be built, e.g. `cargo build --release -p coelanox-standalone`); supports `run` and **`--serve`** (IPC v1).
///
/// Cross-arch (`--arm` / `--target`): not implemented; use CI or release artifacts.
fn extract_command(
    output: &str,
    minimal: bool,
    arm: bool,
    target: Option<&str>,
) -> CoelanoxResult<()> {
    info!(
        "Extract: output={}, minimal={}, arm={}, target={:?}",
        output, minimal, arm, target
    );
    let effective_target = target.map(|s| s.to_string()).or_else(|| {
        if arm {
            Some("aarch64-unknown-linux-gnu".to_string())
        } else {
            None
        }
    });

    if effective_target.is_some() {
        eprintln!("Cross-architecture extract (--arm / --target) is not available in this build.");
        eprintln!("Use extract without those flags on the build machine, or produce the target binary in CI.");
        return Err(CoelanoxError::Runtime(
            "Cross-architecture extract is not supported; use same-arch extract or CI-built binaries.".to_string(),
        ));
    }

    let out_path = Path::new(output);
    let src = if minimal {
        resolve_sibling_coelanox_run()?
    } else {
        std::env::current_exe().map_err(CoelanoxError::FileSystem)?
    };

    fs::copy(&src, out_path).map_err(CoelanoxError::FileSystem)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(out_path)
            .map_err(CoelanoxError::FileSystem)?
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(out_path, perms).map_err(CoelanoxError::FileSystem)?;
    }

    println!("Binary written to {}", output);
    if minimal {
        println!(
            "Minimal runtime: run inference as `coelanox-run model.cnox [input.json]`, or IPC with `coelanox-run model.cnox --serve` (same framing as `coelanox serve`)."
        );
    } else {
        println!(
            "Full COELANOX CLI: includes package, verify, run, serve (stdin/stdout IPC v1), benchmark, env, … Deploy alongside `.cnox` files or use `coelanox extract --minimal -o coelanox-run` for a smaller runner."
        );
    }
    Ok(())
}

/// `coelanox-run` in the same directory as the running `coelanox` (e.g. both in `target/release/`).
fn resolve_sibling_coelanox_run() -> CoelanoxResult<PathBuf> {
    let exe = std::env::current_exe().map_err(CoelanoxError::FileSystem)?;
    let dir = exe.parent().ok_or_else(|| {
        CoelanoxError::format("Could not resolve directory of current executable")
    })?;
    let name = if cfg!(windows) {
        "coelanox-run.exe"
    } else {
        "coelanox-run"
    };
    let candidate = dir.join(name);
    if candidate.is_file() {
        return Ok(candidate);
    }
    Err(CoelanoxError::format(format!(
        "Minimal runtime not found: expected `{}` beside `{}`. \
         Build it with: cargo build --release -p coelanox-standalone (from the `coelanox/` workspace), \
         then run `coelanox extract --minimal` again so both binaries live in the same directory (e.g. target/release/).",
        candidate.display(),
        exe.display()
    )))
}

/// Debug command - Run inference with detailed diagnostics and crash analysis
fn debug_command(
    file: &str,
    input: Option<&str>,
    output: &str,
    use_gdb: bool,
    dump_registers: bool,
    _trace_limit: usize,
    trusted_key: Option<&str>,
    require_signature: bool,
    no_verify: bool,
) -> CoelanoxResult<()> {
    info!("Debug session: {}", file);

    if use_gdb {
        return debug_with_gdb(
            file,
            input,
            output,
            trusted_key,
            require_signature,
            no_verify,
        );
    }

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "This container is signed; provide a trusted public key (--trusted-key or COELANOX_TRUSTED_PUBLIC_KEY). Required when --require-signature is set.",
            ));
        }
    }

    let config = RuntimeConfig::from_env_with_optional_config_file()?;
    let runtime = CoelanoxRuntime::with_config(config);

    // Load container (same trust/verify semantics as `coelanox run`)
    info!("Loading container: {}", file);
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key)?;
    }
    if no_verify {
        info!("Loading container (integrity verification skipped per --no-verify)");
        runtime.load_container_no_verify(file)?;
    } else {
        runtime.load_container(file).inspect_err(|e| {
            if should_hint_container_integrity_failure(e) {
                eprintln!("Load aborted: integrity verification failed.");
                eprintln!("The file may be corrupted, truncated, or modified.");
            }
        })?;
    }

    let container_info = runtime.get_container_info(file)?;
    println!("--- Container ---");
    println!("Model: {}", container_info.manifest.name);
    println!("Input shapes: {:?}", container_info.manifest.input_shapes);
    println!("Output shapes: {:?}", container_info.manifest.output_shapes);
    println!(
        "Machine code size: {} bytes",
        container_info.optimized_code.len()
    );
    println!(
        "Compressed weights size: {} bytes",
        container_info.compressed_weights_len()
    );
    println!();

    // Load input data
    let input_data = if let Some(input_file) = input {
        info!("Input file: {}", input_file);
        let input_json = fs::read_to_string(input_file).map_err(CoelanoxError::FileSystem)?;
        let input_vec: Vec<f32> = serde_json::from_str(&input_json)
            .map_err(|e| CoelanoxError::format(format!("Invalid input JSON: {}", e)))?;
        println!("Input elements loaded: {}", input_vec.len());
        input_vec
    } else {
        // Generate dummy input data
        let input_size: usize = container_info
            .manifest
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        println!("Synthetic input (ones): {} elements", input_size);
        vec![1.0f32; input_size]
    };

    // Verify input descriptor setup
    println!("--- Input layout ---");
    let mut offset = 0;
    for (i, shape) in container_info.manifest.input_shapes.iter().enumerate() {
        let size: usize = shape.iter().product();
        if offset + size > input_data.len() {
            error!(
                "Input data too small: need {} elements starting at offset {}, got {} total",
                size,
                offset,
                input_data.len()
            );
            return Err(CoelanoxError::Runtime(format!(
                "Input data size mismatch: expected at least {} elements, got {}",
                offset + size,
                input_data.len()
            )));
        }
        println!(
            "Input {}: shape {:?}, size {}, offset {} - {} (indices {}-{})",
            i,
            shape,
            size,
            offset,
            offset + size - 1,
            offset,
            offset + size - 1
        );
        offset += size;
    }
    println!();

    // Set up signal handler for SIGSEGV (Unix only)
    #[cfg(unix)]
    if dump_registers {
        unsafe {
            libc::signal(libc::SIGSEGV, crash_handler as libc::sighandler_t);
        }
    }

    println!("--- Inference (debug) ---");
    println!("If the process crashes (e.g. SIGSEGV), additional detail may be printed below.");
    println!();

    // Execute with error handling
    match runtime.execute(file, &input_data) {
        Ok(result) => {
            println!("Inference completed.");
            println!("Output size: {} elements", result.len());

            // Save debug output
            let debug_info = serde_json::json!({
                "status": "success",
                "input_size": input_data.len(),
                "output_size": result.len(),
                "container_info": {
                    "model_name": container_info.manifest.name,
                    "input_shapes": container_info.manifest.input_shapes,
                    "output_shapes": container_info.manifest.output_shapes,
                    "machine_code_size": container_info.optimized_code.len(),
                    "compressed_weights_size": container_info.compressed_weights_len(),
                },
                "output": result,
            });

            let output_json = serde_json::to_string_pretty(&debug_info)
                .map_err(|e| CoelanoxError::format(format!("Could not write debug JSON: {}", e)))?;

            fs::write(output, output_json).map_err(CoelanoxError::FileSystem)?;

            println!("Wrote debug output to {}", output);
            Ok(())
        }
        Err(e) => {
            error!("Inference failed: {:?}", e);
            println!();
            println!("--- Failure detail ---");
            println!("Error: {:?}", e);

            // Save error debug output
            let debug_info = serde_json::json!({
                "status": "error",
                "error": format!("{:?}", e),
                "input_size": input_data.len(),
                "container_info": {
                    "model_name": container_info.manifest.name,
                    "input_shapes": container_info.manifest.input_shapes,
                    "output_shapes": container_info.manifest.output_shapes,
                    "machine_code_size": container_info.optimized_code.len(),
                    "compressed_weights_size": container_info.compressed_weights_len(),
                },
            });

            let output_json = serde_json::to_string_pretty(&debug_info)
                .map_err(|e| CoelanoxError::format(format!("Could not write debug JSON: {}", e)))?;

            fs::write(output, output_json).map_err(CoelanoxError::FileSystem)?;

            println!("Wrote debug output to {}", output);
            Err(e)
        }
    }
}

/// Debug using GDB (external debugger)
fn debug_with_gdb(
    file: &str,
    input: Option<&str>,
    output: &str,
    trusted_key: Option<&str>,
    require_signature: bool,
    no_verify: bool,
) -> CoelanoxResult<()> {
    info!("Starting GDB session");

    // Get path to current binary
    let binary_path = std::env::current_exe()
        .map_err(|e| CoelanoxError::format(format!("Could not resolve executable path: {}", e)))?;

    let mut run_flags = String::new();
    if no_verify {
        run_flags.push_str(" --no-verify");
    }
    if let Some(k) = trusted_key {
        run_flags.push_str(&format!(" --trusted-key {}", k));
    }
    if require_signature {
        run_flags.push_str(" --require-signature");
    }

    // Create GDB script
    let gdb_script = format!(
        r#"
set confirm off
file {}
break coelanox_orchestrator::runtime::CoelanoxRuntime::execute_machine_code
commands
    print $rdi
    print $rsi
    print $rdx
    print $rcx
    print $r8
    print $r9
    print $r12
    print $r13
    print $r14
    print $r15
    print $rsp
    print $rbp
    x/16gx $rsp
    continue
end
run run{} --file {} {}
continue
bt
info registers
x/16gx $rsp
quit
"#,
        binary_path.display(),
        run_flags,
        file,
        input.map(|s| format!("--input {}", s)).unwrap_or_default()
    );

    // Write GDB script to temp file
    let script_path = "/tmp/coelanox_debug.gdb";
    fs::write(script_path, gdb_script).map_err(CoelanoxError::FileSystem)?;

    println!("GDB script: {}", script_path);
    println!("Running GDB...\n");

    // Run GDB
    let output_cmd = Command::new("gdb")
        .arg("-batch")
        .arg("-x")
        .arg(script_path)
        .arg(&binary_path)
        .output()
        .map_err(|e| {
            CoelanoxError::format(format!(
                "GDB failed to start ({}). Install GDB or check PATH.",
                e
            ))
        })?;

    // Save GDB output
    let gdb_output = String::from_utf8_lossy(&output_cmd.stdout);
    let gdb_stderr = String::from_utf8_lossy(&output_cmd.stderr);

    let debug_info = serde_json::json!({
        "status": if output_cmd.status.success() { "completed" } else { "crashed" },
        "exit_code": output_cmd.status.code(),
        "gdb_stdout": gdb_output,
        "gdb_stderr": gdb_stderr,
    });

    let output_json = serde_json::to_string_pretty(&debug_info)
        .map_err(|e| CoelanoxError::format(format!("Could not write debug JSON: {}", e)))?;

    fs::write(output, output_json).map_err(CoelanoxError::FileSystem)?;

    println!("{}", gdb_output);
    if !gdb_stderr.is_empty() {
        eprintln!("{}", gdb_stderr);
    }

    println!();
    println!("GDB output written to {}", output);

    Ok(())
}

/// Signal handler for SIGSEGV - dump register state (Unix only)
#[cfg(unix)]
extern "C" fn crash_handler(_sig: libc::c_int) {
    eprintln!();
    eprintln!("Segmentation fault (SIGSEGV)");
    eprintln!("--- Registers (limited) ---");
    eprintln!("For a full register and stack view, run: coelanox debug --file <container> --gdb");
    eprintln!("Or attach GDB manually: gdb <coelanox binary>");
    eprintln!();

    // Print stack trace if available
    eprintln!("--- Stack trace ---");
    #[cfg(backtrace)]
    {
        let backtrace = std::backtrace::Backtrace::capture();
        eprintln!("{:?}", backtrace);
    }
    #[cfg(not(backtrace))]
    {
        eprintln!("Set RUST_BACKTRACE=1 for a stack trace.");
    }

    // Additional debug info
    eprintln!();
    eprintln!("--- Likely causes (SIGSEGV) ---");
    eprintln!("  - Invalid memory access (null, out of bounds)");
    eprintln!("  - Stack overflow or corrupted frame");
    eprintln!("  - Native code path: bad weights/input pointers or alignment");

    std::process::exit(139); // SIGSEGV exit code
}

/// Validate shape format (e.g., "[1,3,224,224],[1,1,28,28]")
fn is_valid_shape_format(shapes_str: &str) -> bool {
    // Split by comma to get individual shapes
    let shapes: Vec<&str> = shapes_str.split(',').collect();

    // Each shape should be in format [dim1,dim2,dim3,...]
    for shape in shapes {
        let shape = shape.trim();
        if !shape.starts_with('[') || !shape.ends_with(']') {
            return false;
        }

        // Remove brackets and split by comma
        let dims_str = &shape[1..shape.len() - 1];
        if dims_str.is_empty() {
            return false;
        }

        // Check each dimension is a valid number
        for dim in dims_str.split(',') {
            if dim.trim().parse::<usize>().is_err() {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod output_path_tests {
    use super::path_already_under_models_output;
    use std::path::Path;

    #[test]
    fn detects_models_output_prefix_plain() {
        assert!(path_already_under_models_output(Path::new(
            "models/output/foo.cnox"
        )));
        assert!(path_already_under_models_output(Path::new("models/output")));
    }

    #[test]
    fn detects_models_output_after_dot_slash() {
        assert!(path_already_under_models_output(Path::new(
            "./models/output/bar.cnox"
        )));
    }

    #[test]
    fn detects_models_output_after_parent_dirs() {
        assert!(path_already_under_models_output(Path::new(
            "../models/output/x.cnox"
        )));
        assert!(path_already_under_models_output(Path::new(
            "../../models/output/y.cnox"
        )));
    }

    #[test]
    fn other_relative_paths_not_flagged_as_models_output() {
        assert!(!path_already_under_models_output(Path::new(
            "out/model.cnox"
        )));
        assert!(!path_already_under_models_output(Path::new(
            "foo/models/output/z.cnox"
        )));
    }
}
