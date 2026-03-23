//! COELANOX CLI - Command line interface for COELANOX
//! 
//! This module provides the command line interface for the COELANOX toolchain.

use clap::{Parser, Subcommand};
use tracing::{info, warn, error};
use coelanox_core::{CoelanoxError, CoelanoxResult, format::{HardwareTarget, header_audit_required, generate_signing_keypair, is_container_signed}};
use coelanox_packager::CoelanoxPackager;
use coelanox_orchestrator::{CoelanoxRuntime, RuntimeConfig};
use std::path::{Path, PathBuf};
use std::fs;
use std::process::Command;

/// Load runtime config: from COELANOX_CONFIG_FILE if set, else from environment (COELANOX_* vars).
fn load_runtime_config() -> RuntimeConfig {
    if let Ok(path) = std::env::var("COELANOX_CONFIG_FILE") {
        let p = Path::new(&path);
        match RuntimeConfig::from_file(p) {
            Ok(c) => {
                info!("Loaded runtime config from {}", path);
                return c;
            }
            Err(e) => {
                warn!("Failed to load config from {}: {}; using env/defaults", path, e);
            }
        }
    }
    RuntimeConfig::from_env()
}

/// Resolve output path: create parent dirs, default to models/output/ for relative paths, add .cnox if no extension.
fn ensure_output_path(input_path: &str, output_path: &str) -> CoelanoxResult<PathBuf> {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    
    // Relative paths (unless already under models/output) go under models/output/
    let normalized_output = output_path
        .iter()
        .map(|s| s.to_string_lossy().to_lowercase())
        .collect::<Vec<_>>();
    
    // Check if output already starts with "models/output" or "models\\output" (Windows)
    let already_in_output_dir = normalized_output.len() >= 2 
        && (normalized_output[0] == "models" && normalized_output[1] == "output");
    
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
        fs::create_dir_all(parent)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
    }
    
    // If no extension provided, add .cnox
    let final_path = if final_output_path.extension().is_none() {
        final_output_path.with_extension("cnox")
    } else {
        final_output_path
    };
    
    println!("📁 Input: {}", input_path.display());
    println!("📁 Output: {}", final_path.display());
    
    Ok(final_path)
}

#[derive(Parser)]
#[command(name = "coelanox")]
#[command(about = "COELANOX - Secure AI Container")]
#[command(version = "0.6.0-beta")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Package a model into a COELANOX container
    Package {
        /// Input model file
        #[arg(short, long)]
        input: String,
        
        /// Output COELANOX file
        #[arg(short, long)]
        output: String,
        
        /// Model type (cv, nlp, audio, tabular, custom)
        #[arg(short, long, default_value = "cv")]
        model_type: String,
        
        /// Input shapes for custom models (e.g., "[1,3,224,224],[1,1,28,28]")
        #[arg(long)]
        input_shapes: Option<String>,
        
        /// Output shapes for custom models (e.g., "[1,1000],[1,10]")
        #[arg(long)]
        output_shapes: Option<String>,
        
        /// Target hardware (cpu, gpu, edge, auto, or hardware-specific: h100, a100, amd-gpu, intel-npu, etc.)
        #[arg(short, long, default_value = "auto")]
        target: String,
        
        /// Optimization level (none, basic, advanced, maximum)
        #[arg(long, default_value = "advanced")]
        optimization: String,
        
        /// Use demo/custom translator instead of general translator (e.g., bert-translator for NLP)
        #[arg(long)]
        use_demo_translator: bool,
        
        /// Include all available backends (fat container, mutually exclusive with --target and --auto)
        #[arg(long)]
        fat: bool,
        
        /// Auto-detect hardware and select backends automatically (mutually exclusive with --target and --fat)
        #[arg(long)]
        auto: bool,
        
        /// Add additional backends (comma-separated: amd,intel,qualcomm)
        #[arg(long)]
        backend_add: Option<String>,
        
        /// Package without any backend code (scalar fallback only at runtime; for benchmarking)
        #[arg(long)]
        fallback_only: bool,
        
        /// Set container so runtime always runs with audit (compliance); audit logs go to default folder if not specified
        #[arg(long)]
        audit_required: bool,
        
        /// Max dynamic memory (workspace + scratch) in MB; build fails if model needs more (e.g. 64, 1024)
        #[arg(long)]
        max_workspace: Option<u64>,
        
        /// Optional external weights file (e.g. for ONNX models with separate fp16/f32 weight file)
        #[arg(long)]
        weights: Option<String>,

        /// Path to Ed25519 secret key file (32 raw bytes) to sign the container. Env: COELANOX_SIGNING_KEY.
        #[arg(long)]
        signing_key: Option<String>,
    },
    
    /// Validate a COELANOX container
    Validate {
        /// COELANOX file to validate
        #[arg(short, long)]
        file: String,
    },
    
    /// Generate Ed25519 keypair for container signing (secret.bin, public.bin)
    Keygen {
        /// Output directory for secret.bin and public.bin (default: current dir)
        #[arg(short, long)]
        output_dir: Option<String>,
    },

    /// Verify container integrity (pre-deployment check)
    Verify {
        /// COELANOX file to verify
        #[arg(short, long)]
        file: String,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Path to trusted Ed25519 public key (32 raw bytes) for signature verification. Env: COELANOX_TRUSTED_PUBLIC_KEY.
        #[arg(long)]
        trusted_key: Option<String>,

        /// Fail if container is signed but no trusted key provided (enforces provenance).
        #[arg(long)]
        require_signature: bool,
    },
    
    /// Show information about a COELANOX container
    Info {
        /// COELANOX file to inspect
        #[arg(short, long)]
        file: String,
        /// Dump execution plan conv steps (step_ix, in_len, out_len, w_len, scratch_raw) to find bad steps
        #[arg(long)]
        dump_plan: bool,
        /// List op types in the IR; show Custom ops (unsupported in scalar backend)
        #[arg(long)]
        list_ops: bool,
    },
    
    /// Run inference on a COELANOX container
    Run {
        /// COELANOX file to run
        #[arg(short, long)]
        file: String,
        
        /// Input data file (JSON)
        #[arg(short, long)]
        input: Option<String>,
        
        /// Output file
        #[arg(short, long)]
        output: String,
        
        /// Backend to use (auto, cpu, gpu, npu, tpu). Use "auto" to match benchmark behaviour.
        #[arg(short, long, default_value = "auto")]
        backend: String,
        
        /// Audit mode: log after each op (dumps op index, type, output size, sample to stderr and optionally to file)
        #[arg(long)]
        audit: bool,
        
        /// When using --audit, append audit log lines to this file
        #[arg(long)]
        audit_output: Option<String>,
        
        /// Max workspace in MB (overrides COELANOX_MAX_WORKSPACE_MB). 0 = no cap.
        #[arg(long)]
        max_workspace_mb: Option<u64>,
        
        /// Skip container integrity (hash) verification. Faster load; use only for trusted/dev.
        #[arg(long)]
        no_verify: bool,

        /// Path to trusted Ed25519 public key (32 raw bytes) for signature verification. Env: COELANOX_TRUSTED_PUBLIC_KEY.
        #[arg(long)]
        trusted_key: Option<String>,

        /// Fail if container is signed but no trusted key provided (enforces provenance).
        #[arg(long)]
        require_signature: bool,
    },
    
    /// Benchmark a COELANOX container
    Benchmark {
        /// COELANOX file to benchmark
        #[arg(short, long)]
        file: String,
        /// Number of iterations (default 100)
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        /// Loop for N iterations (long-run stability); overrides --iterations when set
        #[arg(long)]
        loop_iterations: Option<usize>,
        /// Backend to use: auto (default), scalar, cpu, simd. Use scalar for interpreted ops; auto/cpu/simd for plan (SIMD).
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Output format: text or json
        #[arg(long, default_value = "text")]
        output: String,
        /// Max workspace in MB (overrides COELANOX_MAX_WORKSPACE_MB). 0 = no cap.
        #[arg(long)]
        max_workspace_mb: Option<u64>,
        /// Skip container integrity (hash) verification for faster load
        #[arg(long)]
        no_verify: bool,

        /// Path to trusted Ed25519 public key (32 raw bytes) for signature verification. Env: COELANOX_TRUSTED_PUBLIC_KEY.
        #[arg(long)]
        trusted_key: Option<String>,

        /// Fail if container is signed but no trusted key provided (enforces provenance).
        #[arg(long)]
        require_signature: bool,
    },

    /// Show system environment information
    Env,
    
    /// Debug a COELANOX container with detailed diagnostics
    Debug {
        /// COELANOX file to debug
        #[arg(short, long)]
        file: String,
        
        /// Input data file (JSON)
        #[arg(short, long)]
        input: Option<String>,
        
        /// Output file for debug logs
        #[arg(short, long, default_value = "debug_output.json")]
        output: String,
        
        /// Enable GDB-style debugging (requires GDB)
        #[arg(long)]
        gdb: bool,
        
        /// Dump register state on crash
        #[arg(long, default_value = "true")]
        dump_registers: bool,
        
        /// Maximum operations to trace before crash
        #[arg(short, long, default_value = "10")]
        trace_limit: usize,
    },
    
    /// Extract runtime-only binary for deployment (same arch or cross-arch, e.g. ARM server).
    /// Produces a standalone runtime that matches this CLI version; deploy it with .cnox files only.
    Extract {
        /// Output path for the runtime binary (e.g. coelanox-run or coelanox-run-arm)
        #[arg(short, long, default_value = "coelanox-run")]
        output: String,
        
        /// Extract runtime for ARM64 (aarch64). Default is current platform.
        #[arg(long)]
        arm: bool,
        
        /// Target triple for cross-arch (e.g. aarch64-unknown-linux-gnu). Overrides --arm if set.
        #[arg(long)]
        target: Option<String>,
    },
}

fn main() -> CoelanoxResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Package { input, output, model_type, input_shapes, output_shapes, target, optimization, use_demo_translator, fat, auto, backend_add, fallback_only, audit_required, max_workspace, weights, signing_key } => {
            package_command(&input, &output, &model_type, input_shapes.as_deref(), output_shapes.as_deref(), &target, &optimization, use_demo_translator, fat, auto, backend_add.as_deref(), fallback_only, audit_required, max_workspace, weights.as_deref(), signing_key.as_deref())
        }
        Commands::Validate { file } => {
            validate_command(&file)
        }
        Commands::Keygen { output_dir } => {
            keygen_command(output_dir.as_deref())
        }
        Commands::Verify { file, verbose, trusted_key, require_signature } => {
            verify_command(&file, verbose, trusted_key.as_deref(), require_signature)
        }
        Commands::Info { file, dump_plan, list_ops } => {
            info_command(&file, dump_plan, list_ops)
        }
        Commands::Run { file, input, output, backend, audit, audit_output, max_workspace_mb, no_verify, trusted_key, require_signature } => {
            run_command(&file, input.as_deref(), &output, &backend, audit, audit_output.as_deref(), max_workspace_mb, no_verify, trusted_key.as_deref(), require_signature)
        }
        Commands::Benchmark { file, iterations, loop_iterations, backend, output, max_workspace_mb, no_verify, trusted_key, require_signature } => {
            benchmark_command(&file, iterations, loop_iterations, &backend, &output, max_workspace_mb, no_verify, trusted_key.as_deref(), require_signature)
        }
        Commands::Env => {
            env_command()
        }
        Commands::Debug { file, input, output, gdb, dump_registers, trace_limit } => {
            debug_command(&file, input.as_deref(), &output, gdb, dump_registers, trace_limit)
        }
        Commands::Extract { output, arm, target } => {
            extract_command(&output, arm, target.as_deref())
        }
    }
}

fn package_command(input: &str, output: &str, model_type: &str, input_shapes: Option<&str>, output_shapes: Option<&str>, target: &str, _optimization: &str, use_demo_translator: bool, fat: bool, auto: bool, backend_add: Option<&str>, fallback_only: bool, audit_required: bool, max_workspace: Option<u64>, weights: Option<&str>, signing_key: Option<&str>) -> CoelanoxResult<()> {
    info!("Packaging model: {} -> {}", input, output);
    info!("Model type: {}", model_type);
    
    // Ensure proper output path and directory structure
    let final_output_path = ensure_output_path(input, output)?;
    
    // Validate model type
    let valid_types = ["cv", "nlp", "audio", "tabular", "custom"];
    if !valid_types.contains(&model_type.to_lowercase().as_str()) {
        return Err(CoelanoxError::format(&format!(
            "Invalid model type '{}'. Valid types: {}", 
            model_type, 
            valid_types.join(", ")
        )));
    }
    
    // Validate custom shapes if provided
    if model_type.to_lowercase() == "custom" {
        if input_shapes.is_none() || output_shapes.is_none() {
            return Err(CoelanoxError::format(
                "Custom model type requires both --input-shapes and --output-shapes to be specified"
            ));
        }
        
        // Validate shape format
        if let Some(input_shapes_str) = input_shapes {
            if !is_valid_shape_format(input_shapes_str) {
                return Err(CoelanoxError::format(&format!(
                    "Invalid input shapes format: '{}'. Expected format: '[1,3,224,224],[1,1,28,28]'",
                    input_shapes_str
                )));
            }
        }
        
        if let Some(output_shapes_str) = output_shapes {
            if !is_valid_shape_format(output_shapes_str) {
                return Err(CoelanoxError::format(&format!(
                    "Invalid output shapes format: '{}'. Expected format: '[1,1000],[1,10]'",
                    output_shapes_str
                )));
            }
        }
        
        info!("Custom shapes - Input: {:?}, Output: {:?}", input_shapes, output_shapes);
    }
    
    // Create packager with environment detection
    eprintln!("🔧 Creating packager...");
    let packager = CoelanoxPackager::new()?;
    eprintln!("✅ Packager created successfully");
    
    // Get environment info
    eprintln!("🔍 Getting environment info...");
    let env = packager.get_environment();
    eprintln!("✅ Environment info retrieved");
    info!("Detected environment:");
    info!("  OS: {}", env.os);
    info!("  Architecture: {}", env.architecture);
    info!("  CPU: {} {} ({} cores)", env.cpu_vendor, env.cpu_model, env.cpu_cores);
    info!("  Memory: {} GB", env.memory_gb);
    if let Some(gpu) = &env.gpu_info {
        info!("  GPU: {} {} ({} GB)", gpu.vendor, gpu.model, gpu.memory_gb);
    }
    info!("  SIMD: {:?}", env.simd_capabilities);
    
    // Validate flag combinations
    if fat && auto {
        return Err(CoelanoxError::format("Cannot use both --fat and --auto flags"));
    }
    if fat && target != "auto" {
        return Err(CoelanoxError::format("Cannot use both --fat and --target flags (use --fat alone or --target alone)"));
    }
    if auto && target != "auto" {
        return Err(CoelanoxError::format("Cannot use both --auto and --target flags (use --auto alone or --target alone)"));
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
                info!("Auto-detected target: {:?}", recommended);
                recommended
            }
            // Hardware-specific targets
            "h100" | "a100" | "v100" => HardwareTarget::Gpu, // NVIDIA GPUs
            "amd-gpu" | "amd" => HardwareTarget::Gpu, // AMD GPU
            "intel-npu" | "intel" => HardwareTarget::NPU, // Intel NPU
            "qualcomm" => HardwareTarget::NPU, // Qualcomm NPU
            _ => {
                // Try as custom target
                if target.contains('-') || target.chars().all(|c| c.is_alphanumeric()) {
                    HardwareTarget::Custom(target.to_string())
                } else {
                    warn!("Unknown target '{}', using auto-detection", target);
                    env.recommended_target()
                }
            }
        }
    };
    
    info!("Target hardware: {:?}", hardware_target);
    
    // Package the model
    eprintln!("🚀 About to call packager.package()...");
    eprintln!("   Input: {}", input);
    eprintln!("   Target: {:?}", hardware_target);
    eprintln!("   Model type: {}", model_type);
    
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
        weights_path.as_deref(),
    )?;
    eprintln!("✅ packager.package() returned successfully");

    // Resolve signing key: --signing-key or COELANOX_SIGNING_KEY (path to 32-byte file)
    let signing_key_path = signing_key
        .map(String::from)
        .or_else(|| std::env::var("COELANOX_SIGNING_KEY").ok());
    let signing_key_bytes = if let Some(ref path) = signing_key_path {
        let key = std::fs::read(path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read signing key from {}: {}", path, e)))?;
        if key.len() != 32 {
            return Err(CoelanoxError::format(&format!(
                "Signing key must be exactly 32 bytes (Ed25519 seed), got {} bytes",
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
    
    info!("✅ Container packaged successfully");
    info!("Container saved to: {}", final_output_path.display());
    info!("Container size: {} bytes", container.total_size());
    
    // Show compression info
    let input_size = std::fs::metadata(input)?.len();
    let compression_ratio = input_size as f64 / container.total_size() as f64;
    info!("Compression ratio: {:.1}x", compression_ratio);
    
    Ok(())
}

fn validate_command(file: &str) -> CoelanoxResult<()> {
    info!("Validating COELANOX container: {}", file);
    
    let container = coelanox_core::format::CoelanoxContainer::read_binary_file(std::path::Path::new(file))?;
    
    println!("✅ Container is valid");
    println!("Version: 0.6.0-beta");
    println!("Model: {}", container.manifest.name);
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
    std::fs::write(&secret_path, &secret).map_err(CoelanoxError::FileSystem)?;
    std::fs::write(&public_path, &public).map_err(CoelanoxError::FileSystem)?;
    println!("✅ Ed25519 keypair generated");
    println!("   Secret: {} (keep private)", secret_path.display());
    println!("   Public: {} (use with --trusted-key)", public_path.display());
    Ok(())
}

fn verify_command(file: &str, verbose: bool, trusted_key: Option<&str>, require_signature: bool) -> CoelanoxResult<()> {
    if verbose {
        info!("Verifying container integrity: {}", file);
    }

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "Container is signed but no trusted key provided. Use --trusted-key or COELANOX_TRUSTED_PUBLIC_KEY (--require-signature enforces provenance)"
            ));
        }
    }

    let trusted_key_bytes = resolve_trusted_key(trusted_key)?;

    match coelanox_core::format::verify_security_hash_with_trusted_key(
        std::path::Path::new(file),
        trusted_key_bytes.as_ref(),
    ) {
        Ok(()) => {
            println!("✅ Container integrity verified");
            if verbose {
                println!("Hash verification successful - container is valid and untampered");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Container integrity verification failed");
            eprintln!("Error: {}", e);
            eprintln!("The container file may have been tampered with or corrupted.");
            eprintln!("Do not load or execute this container.");
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
            let key = std::fs::read(path)
                .map_err(|e| CoelanoxError::format(&format!("Failed to read trusted key from {}: {}", path, e)))?;
            if key.len() != 32 {
                return Err(CoelanoxError::format(&format!(
                    "Trusted public key must be exactly 32 bytes, got {} bytes",
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
    info!("Getting info for COELANOX container: {}", file);
    
    let container = coelanox_core::format::CoelanoxContainer::read_binary_file(std::path::Path::new(file))?;
    
    if list_ops {
        use coelanox_core::ir::{deserialize_optimized_ir, OpType};
        let ir = deserialize_optimized_ir(&container.ir_data)?;
        let mut custom_ops: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut all_types: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for node in &ir.optimized_nodes {
            let key = format!("{:?}", node.op_type);
            *all_types.entry(key.clone()).or_insert(0) += 1;
            if let OpType::Custom(name) = &node.op_type {
                *custom_ops.entry(name.clone()).or_insert(0) += 1;
            }
        }
        println!("=== Op types in IR (total nodes: {}) ===", ir.optimized_nodes.len());
        let mut types: Vec<_> = all_types.into_iter().collect();
        types.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, count) in &types {
            println!("  {}: {}", op, count);
        }
        if !custom_ops.is_empty() {
            println!();
            println!("=== Custom ops (not supported in scalar backend) ===");
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
            println!("=== Execution plan conv steps (op_id=30) ===");
            println!("{:<8} {:>12} {:>12} {:>12} {:>14}  (scratch MB)", "step", "in_len", "out_len", "w_len", "scratch_el");
            for (step_ix, in_len, out_len, w_len, scratch) in &steps {
                let mb = (*scratch as u64 * 4 + 1024 * 1024 - 1) / (1024 * 1024);
                println!("{:<8} {:>12} {:>12} {:>12} {:>14}  ~{}", step_ix, in_len, out_len, w_len, scratch, mb);
            }
            let max_scratch = steps.iter().map(|(_, _, _, _, s)| s).max().copied().unwrap_or(0);
            println!();
            println!("Conv steps: {}  max_scratch_elements: {}  (~{} MB)", steps.len(), max_scratch, (max_scratch as u64 * 4 + 1024 * 1024 - 1) / (1024 * 1024));
            return Ok(());
        } else {
            return Err(CoelanoxError::Runtime("Container has no execution plan (--dump-plan requires a plan)".to_string()));
        }
    }
    
    println!("=== COELANOX Container Information ===");
    println!("File: {}", file);
    println!("Version: 0.6.0-beta");
    println!("Status: ✅ Valid");
    println!();
    println!("=== Model Information ===");
    println!("Name: {}", container.manifest.name);
    println!("Format: {:?}", container.manifest.format);
    println!("Created: {}", container.manifest.created_at);
    println!("Modified: {}", container.manifest.modified_at);
    println!();
    println!("=== Input/Output ===");
    println!("Input shapes: {:?}", container.manifest.input_shapes);
    println!("Output shapes: {:?}", container.manifest.output_shapes);
    println!();
    println!("=== Container Details ===");
    println!("Total size: {} bytes", container.total_size());
    println!("Header size: {} bytes", coelanox_core::format::COELANOX_HEADER_SIZE);
    println!("Manifest size: ~{} bytes", serde_json::to_string(&container.manifest).unwrap_or_default().len());
    println!("Machine code size: {} bytes", container.optimized_code.len());
    println!("Compressed weights size: {} bytes", container.compressed_weights_len());
    println!("IR data size: {} bytes", container.ir_data.len());
    if let Some(elements) = container.manifest.workspace_required_elements {
        let mb = (elements * 4 + 1024 * 1024 - 1) / (1024 * 1024);
        println!("Workspace required: {} elements (~{} MB)", elements, mb);
    }
    println!();
    println!("=== Performance ===");
    // Manifest does not store source model size; compression ratio N/A
    println!("Compression ratio: N/A (original size not available)");
    
    Ok(())
}

/// Default audit output path when container has audit-required flag and user did not set --audit-output:
/// coelanox_audit/<container_stem>_<unix_secs>.json in current directory.
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

fn run_command(file: &str, input: Option<&str>, output: &str, backend: &str, audit: bool, audit_output: Option<&str>, max_workspace_mb: Option<u64>, no_verify: bool, trusted_key: Option<&str>, require_signature: bool) -> CoelanoxResult<()> {
    info!("Running inference: {} -> {} (backend: {})", input.unwrap_or("auto"), output, backend);
    let resolved_workspace_mb = max_workspace_mb.or_else(|| std::env::var("COELANOX_MAX_WORKSPACE_MB").ok().and_then(|s| s.parse().ok()));

    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "Container is signed but no trusted key provided. Use --trusted-key or COELANOX_TRUSTED_PUBLIC_KEY (--require-signature enforces provenance)"
            ));
        }
    }

    // Load runtime (config from COELANOX_CONFIG_FILE or COELANOX_* env vars)
    let config = load_runtime_config();
    let runtime = CoelanoxRuntime::with_config(config);
    if let Some(mb) = resolved_workspace_mb {
        runtime.set_max_workspace_mb(mb);
    }
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key);
    }
    if no_verify {
        info!("Loading container (--no-verify: skipping integrity hash)...");
        runtime.load_container_no_verify(file)?;
        info!("✅ Container loaded (verification skipped)");
    } else {
        info!("Loading container (verify + parse from mmap)...");
        runtime.load_container(file).map_err(|e| {
            if e.to_string().contains("verification") || e.to_string().contains("hash") {
                eprintln!("❌ Container verification failed - refusing to load");
                eprintln!("The container file may have been tampered with or corrupted.");
            }
            e
        })?;
        info!("✅ Container loaded and verified");
    }
    
    // If container has audit-required flag, always run with audit (compliance). Use --audit-output if set, else default folder.
    let container = runtime.get_container_info(file)?;
    let audit_required_by_container = header_audit_required(container.header.flags);
    let audit_path: Option<PathBuf> = if audit_required_by_container {
        Some(audit_output.map(PathBuf::from).unwrap_or_else(|| default_audit_path(file)))
    } else if audit {
        audit_output.map(PathBuf::from)
    } else {
        None
    };
    if audit || audit_required_by_container {
        if audit_required_by_container && audit_output.is_none() {
            if let Some(ref p) = audit_path {
                info!("Container requires audit; writing audit log to {}", p.display());
            }
        }
        coelanox_orchestrator::audit::set_audit_mode(true, audit_path.as_deref().map(Path::new));
    }
    
    // Load input data
    let input_data = if let Some(input_file) = input {
        if input_file.ends_with(".json") {
            // Load from JSON file
            let json_str = std::fs::read_to_string(input_file)
                .map_err(|e| CoelanoxError::FileSystem(e))?;
            serde_json::from_str::<Vec<f32>>(&json_str)
                .map_err(|e| CoelanoxError::format(&format!("Invalid JSON input: {}", e)))?
        } else {
            // Generate dummy input data
            let container = runtime.get_container_info(file)?;
            let input_size: usize = container.manifest.input_shapes.iter()
                .map(|shape| shape.iter().product::<usize>())
                .sum();
            vec![1.0f32; input_size]
        }
    } else {
        // Generate dummy input data
        let container = runtime.get_container_info(file)?;
        let input_size: usize = container.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        vec![1.0f32; input_size]
    };

    // Execute inference with backend selection (backend manager chooses execution path)
    let result = runtime.execute_with_backend(file, &input_data, backend);
    if audit || audit_required_by_container {
        coelanox_orchestrator::audit::set_audit_mode(false, None);
    }
    let result = result?;
    
    // Save output
    let output_json = serde_json::to_string_pretty(&result)
        .map_err(|e| CoelanoxError::format(&format!("Failed to serialize output: {}", e)))?;
    
    std::fs::write(output, output_json)
        .map_err(|e| CoelanoxError::FileSystem(e))?;
    
    info!("✅ Inference completed successfully");
    println!("Input size: {} elements", input_data.len());
    println!("Output size: {} elements", result.len());
    println!("Execution time: {} μs", runtime.get_stats().avg_execution_time);
    
    Ok(())
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
    let resolved_workspace_mb = max_workspace_mb.or_else(|| std::env::var("COELANOX_MAX_WORKSPACE_MB").ok().and_then(|s| s.parse().ok()));
    let is_long_run = effective_iterations >= 1000;
    info!(
        "Benchmarking COELANOX container: {} ({} iterations, backend: {}{})",
        file,
        effective_iterations,
        backend,
        if is_long_run { ", long-run stability" } else { "" }
    );

    let config = load_runtime_config();
    let mut runtime = CoelanoxRuntime::with_config(config);
    if let Some(mb) = resolved_workspace_mb {
        runtime.set_max_workspace_mb(mb);
    }
    if let Some(key) = resolve_trusted_key(trusted_key)? {
        runtime.set_trusted_public_key(key);
    }
    if require_signature {
        let signed = is_container_signed(std::path::Path::new(file))?;
        let has_key = trusted_key.is_some() || std::env::var("COELANOX_TRUSTED_PUBLIC_KEY").is_ok();
        if signed && !has_key {
            return Err(CoelanoxError::format(
                "Container is signed but no trusted key provided. Use --trusted-key or COELANOX_TRUSTED_PUBLIC_KEY (--require-signature enforces provenance)"
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
        println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        return Ok(());
    }

    // Human-readable text output
    println!("=== Benchmark Results ===");
    println!("File: {}", result.container_name);
    println!("Iterations: {} (warm-up: {})", result.iterations, result.warm_up_runs);
    if let Some(load_ms) = result.load_time_ms {
        println!("Load time (to first execute-ready): {} ms", load_ms);
    }
    println!();
    println!("=== Latency (μs) ===");
    println!("  Average:  {:.2}", result.avg_time);
    println!("  Min:      {}", result.min_time);
    println!("  Max:      {}", result.max_time);
    println!("  p50:      {}", result.p50_us);
    println!("  p95:      {}", result.p95_us);
    println!("  p99:      {}", result.p99_us);
    println!("  Std dev:  {:.2}", result.std_dev_us);
    println!("  True cold (1st run, no warm-up): {} μs  (same metric as single \"run\" Execution time)", result.true_cold_time_us);
    if let Some(cold) = result.cold_time_us {
        println!("  First measured (after warm-up): {} μs", cold);
    }
    if let Some(warm) = result.warm_avg_us {
        println!("  Warm (avg after 1st measured): {} μs", warm);
    }
    println!();
    println!("=== Throughput ===");
    let throughput = 1_000_000.0 / result.avg_time as f64;
    println!("  Sustained: {:.0} inf/s", throughput);
    if result.min_time > 0 {
        println!("  Max burst: {:.0} inf/s (1/min_latency)", 1_000_000.0 / result.min_time as f64);
    }
    println!();
    println!("=== Memory ===");
    println!("  Workspace (elements): {}", result.workspace_elements);
    println!("  Weights (decompressed): {} bytes", result.weights_bytes);
    println!("  Executable memory: {} bytes", result.executable_memory_bytes);
    if let Some(rss) = result.rss_after_bytes {
        println!("  RSS after benchmark: {} bytes ({:.2} MiB)", rss, rss as f64 / (1024.0 * 1024.0));
    }
    println!();
    println!("=== Breakdown ===");
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
            println!("=== Long-run stability ===");
            println!("  First 10% avg: {} μs", first);
            println!("  Last 10% avg:  {} μs", last);
            println!("  Drift (last - first): {} μs {}", drift, if drift > 0 { "(slowdown)" } else { "(improvement)" });
        }
    }
    Ok(())
}

fn env_command() -> CoelanoxResult<()> {
    info!("Detecting system environment");
    
    let env = coelanox_core::utils::SystemEnvironment::detect()?;
    
    println!("=== System Environment ===");
    println!("OS: {}", env.os);
    println!("Architecture: {}", env.architecture);
    println!("CPU Vendor: {}", env.cpu_vendor);
    println!("CPU Model: {}", env.cpu_model);
    println!("CPU Cores: {}", env.cpu_cores);
    println!("Memory: {} GB", env.memory_gb);
    println!();
    
    if let Some(ref gpu) = env.gpu_info {
        println!("=== GPU Information ===");
        println!("Vendor: {}", gpu.vendor);
        println!("Model: {}", gpu.model);
        println!("Memory: {} GB", gpu.memory_gb);
        if let Some(cc) = &gpu.compute_capability {
            println!("Compute Capability: {}", cc);
        }
        if let Some(driver) = &gpu.driver_version {
            println!("Driver Version: {}", driver);
        }
        println!();
    }
    
    println!("=== SIMD Capabilities ===");
    for capability in &env.simd_capabilities {
        println!("  {:?}", capability);
    }
    println!();
    
    println!("=== System Capabilities ===");
    for (capability, available) in &env.capabilities {
        println!("  {}: {}", capability, if *available { "✅" } else { "❌" });
    }
    println!();
    
    println!("=== Recommendations ===");
    let recommended_target = env.recommended_target();
    let recommended_optimization = env.recommended_optimization();
    println!("Recommended target: {:?}", recommended_target);
    println!("Recommended optimization: {:?}", recommended_optimization);
    
    Ok(())
}

/// Extract command - Produce runtime-only binary for deployment.
/// Same-arch: copies the current executable to --output so you can deploy that binary + .cnox files only.
/// Cross-arch (--arm / --target): not yet implemented; use CI or release artifacts for ARM.
fn extract_command(output: &str, arm: bool, target: Option<&str>) -> CoelanoxResult<()> {
    info!("Extract runtime: output={}, arm={}, target={:?}", output, arm, target);
    let effective_target = target
        .map(|s| s.to_string())
        .or_else(|| if arm { Some("aarch64-unknown-linux-gnu".to_string()) } else { None });

    if effective_target.is_some() {
        eprintln!("❌ Cross-arch extract (--arm / --target) is not yet implemented.");
        eprintln!("   Use the same-arch extract without --arm/--target, or build the runtime for the target in CI.");
        return Err(CoelanoxError::Runtime(
            "Cross-arch extract not implemented; use same-arch extract or CI/release artifacts".to_string(),
        ));
    }

    let current_exe = std::env::current_exe().map_err(CoelanoxError::FileSystem)?;
    let out_path = Path::new(output);
    fs::copy(&current_exe, out_path).map_err(CoelanoxError::FileSystem)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(out_path).map_err(CoelanoxError::FileSystem)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(out_path, perms).map_err(CoelanoxError::FileSystem)?;
    }

    println!("✅ Runtime extracted to {}", output);
    println!("   Deploy this binary with your .cnox file(s); run, verify, and benchmark are supported.");
    Ok(())
}

/// Debug command - Run inference with detailed diagnostics and crash analysis
fn debug_command(
    file: &str,
    input: Option<&str>,
    output: &str,
    use_gdb: bool,
    dump_registers: bool,
    _trace_limit: usize,
) -> CoelanoxResult<()> {
    info!("🔍 Debugging COELANOX container: {}", file);
    
    if use_gdb {
        return debug_with_gdb(file, input, output);
    }
    
    let config = load_runtime_config();
    let runtime = CoelanoxRuntime::with_config(config);

    // Load container
    info!("📦 Loading container: {}", file);
    runtime.load_container(file)?;
    
    let container_info = runtime.get_container_info(file)?;
    println!("=== Container Information ===");
    println!("Model: {}", container_info.manifest.name);
    println!("Input shapes: {:?}", container_info.manifest.input_shapes);
    println!("Output shapes: {:?}", container_info.manifest.output_shapes);
    println!("Machine code size: {} bytes", container_info.optimized_code.len());
    println!("Compressed weights size: {} bytes", container_info.compressed_weights_len());
    println!();
    
    // Load input data
    let input_data = if let Some(input_file) = input {
        info!("📥 Loading input from: {}", input_file);
        let input_json = fs::read_to_string(input_file)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        let input_vec: Vec<f32> = serde_json::from_str(&input_json)
            .map_err(|e| CoelanoxError::format(&format!("Failed to parse input JSON: {}", e)))?;
        println!("Loaded {} input elements", input_vec.len());
        input_vec
    } else {
        // Generate dummy input data
        let input_size: usize = container_info.manifest.input_shapes.iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        println!("Generating dummy input data: {} elements", input_size);
        vec![1.0f32; input_size]
    };
    
    // Verify input descriptor setup
    println!("=== Input Descriptor Setup ===");
    let mut offset = 0;
    for (i, shape) in container_info.manifest.input_shapes.iter().enumerate() {
        let size: usize = shape.iter().product();
        if offset + size > input_data.len() {
            error!("Input data too small: need {} elements starting at offset {}, got {} total", size, offset, input_data.len());
            return Err(CoelanoxError::Runtime(format!(
                "Input data size mismatch: expected at least {} elements, got {}",
                offset + size,
                input_data.len()
            )));
        }
        println!("Input {}: shape {:?}, size {}, offset {} - {} (indices {}-{})", 
            i, shape, size, offset, offset + size - 1, offset, offset + size - 1);
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
    
    println!("=== Executing Inference (Debug Mode) ===");
    println!("⚠️  If a segmentation fault occurs, debug information will be printed");
    println!();
    
    // Execute with error handling
    match runtime.execute(file, &input_data) {
        Ok(result) => {
            println!("✅ Inference completed successfully!");
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
                .map_err(|e| CoelanoxError::format(&format!("Failed to serialize debug output: {}", e)))?;
            
            fs::write(output, output_json)
                .map_err(|e| CoelanoxError::FileSystem(e))?;
            
            println!("Debug output saved to: {}", output);
            Ok(())
        }
        Err(e) => {
            error!("❌ Inference failed: {:?}", e);
            println!("\n=== Debug Analysis ===");
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
                .map_err(|e| CoelanoxError::format(&format!("Failed to serialize debug output: {}", e)))?;
            
            fs::write(output, output_json)
                .map_err(|e| CoelanoxError::FileSystem(e))?;
            
            println!("Debug output saved to: {}", output);
            Err(e)
        }
    }
}

/// Debug using GDB (external debugger)
fn debug_with_gdb(file: &str, input: Option<&str>, output: &str) -> CoelanoxResult<()> {
    info!("🐛 Starting GDB debug session");
    
    // Get path to current binary
    let binary_path = std::env::current_exe()
        .map_err(|e| CoelanoxError::format(&format!("Failed to get binary path: {}", e)))?;
    
    // Create GDB script
    let gdb_script = format!(r#"
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
run run --file {} {}
continue
bt
info registers
x/16gx $rsp
quit
"#, 
        binary_path.display(),
        file,
        input.map(|s| format!("--input {}", s)).unwrap_or_default()
    );
    
    // Write GDB script to temp file
    let script_path = "/tmp/coelanox_debug.gdb";
    fs::write(script_path, gdb_script)
        .map_err(|e| CoelanoxError::FileSystem(e))?;
    
    println!("📝 GDB script written to: {}", script_path);
    println!("🔍 Running GDB...\n");
    
    // Run GDB
    let output_cmd = Command::new("gdb")
        .arg("-batch")
        .arg("-x")
        .arg(script_path)
        .arg(&binary_path)
        .output()
        .map_err(|e| CoelanoxError::format(&format!("Failed to run GDB: {}. Make sure GDB is installed.", e)))?;
    
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
        .map_err(|e| CoelanoxError::format(&format!("Failed to serialize debug output: {}", e)))?;
    
    fs::write(output, output_json)
        .map_err(|e| CoelanoxError::FileSystem(e))?;
    
    println!("{}", gdb_output);
    if !gdb_stderr.is_empty() {
        eprintln!("{}", gdb_stderr);
    }
    
    println!("\n📄 GDB output saved to: {}", output);
    
    Ok(())
}

/// Signal handler for SIGSEGV - dump register state (Unix only)
#[cfg(unix)]
extern "C" fn crash_handler(_sig: libc::c_int) {
    eprintln!("\n💥 SEGMENTATION FAULT DETECTED!");
    eprintln!("=== Register State (best effort) ===");
    eprintln!("⚠️  Full register dump requires GDB. Run with --gdb flag for detailed analysis.");
    eprintln!("\nTo debug:");
    eprintln!("  1. Run with: coelanox debug --file <file> --gdb");
    eprintln!("  2. Or attach GDB manually: gdb <binary>");
    eprintln!();
    
    // Print stack trace if available
    eprintln!("=== Stack Trace ===");
    #[cfg(backtrace)]
    {
        let backtrace = std::backtrace::Backtrace::capture();
        eprintln!("{:?}", backtrace);
    }
    #[cfg(not(backtrace))]
    {
        eprintln!("Stack trace disabled. Enable with RUST_BACKTRACE=1");
    }
    
    // Additional debug info
    eprintln!("\n=== Debug Information ===");
    eprintln!("Signal: SIGSEGV (segmentation fault)");
    eprintln!("This usually indicates:");
    eprintln!("  - Invalid memory access (null pointer, out of bounds)");
    eprintln!("  - Stack overflow");
    eprintln!("  - Register corruption");
    eprintln!("  - Incorrect stack alignment");
    eprintln!("\nCommon causes in generated code:");
    eprintln!("  - Operation accessing invalid memory address");
    eprintln!("  - Weights pointer not correctly loaded from stack");
    eprintln!("  - Input pointer corrupted by operation");
    eprintln!("  - Stack frame corrupted (rbp/rsp mismatch)");
    
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
        let dims_str = &shape[1..shape.len()-1];
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
