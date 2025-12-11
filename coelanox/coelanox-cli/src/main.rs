//! COELANOX CLI - Command line interface for COELANOX
//! 
//! This module provides the command line interface for the COELANOX toolchain.

use clap::{Parser, Subcommand};
use tracing::{info, warn, error};
use coelanox_core::{CoelanoxError, CoelanoxResult, format::HardwareTarget};
use coelanox_packager::CoelanoxPackager;
use coelanox_runtime::CoelanoxRuntime;
use std::path::{Path, PathBuf};
use std::fs;
use std::process::Command;

/// Ensure output directory exists and return proper output path
fn ensure_output_path(input_path: &str, output_path: &str) -> CoelanoxResult<PathBuf> {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    
    // Normalize the output path to check if it already starts with models/output
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
        
        /// Target hardware (cpu, gpu, edge, auto)
        #[arg(short, long, default_value = "auto")]
        target: String,
        
        /// Optimization level (none, basic, advanced, maximum)
        #[arg(long, default_value = "advanced")]
        optimization: String,
        
        /// Use demo/custom translator instead of general translator (e.g., bert-translator for NLP)
        #[arg(long)]
        use_demo_translator: bool,
    },
    
    /// Validate a COELANOX container
    Validate {
        /// COELANOX file to validate
        #[arg(short, long)]
        file: String,
    },
    
    /// Show information about a COELANOX container
    Info {
        /// COELANOX file to inspect
        #[arg(short, long)]
        file: String,
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
        
        /// Backend to use (cpu, gpu, npu, tpu)
        #[arg(short, long, default_value = "cpu")]
        backend: String,
    },
    
    /// Benchmark a COELANOX container
    Benchmark {
        /// COELANOX file to benchmark
        #[arg(short, long)]
        file: String,
        
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
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
}

fn main() -> CoelanoxResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Package { input, output, model_type, input_shapes, output_shapes, target, optimization, use_demo_translator } => {
            package_command(&input, &output, &model_type, input_shapes.as_deref(), output_shapes.as_deref(), &target, &optimization, use_demo_translator)
        }
        Commands::Validate { file } => {
            validate_command(&file)
        }
        Commands::Info { file } => {
            info_command(&file)
        }
        Commands::Run { file, input, output, backend } => {
            run_command(&file, input.as_deref(), &output)
        }
        Commands::Benchmark { file, iterations } => {
            benchmark_command(&file, iterations)
        }
        Commands::Env => {
            env_command()
        }
        Commands::Debug { file, input, output, gdb, dump_registers, trace_limit } => {
            debug_command(&file, input.as_deref(), &output, gdb, dump_registers, trace_limit)
        }
    }
}

fn package_command(input: &str, output: &str, model_type: &str, input_shapes: Option<&str>, output_shapes: Option<&str>, target: &str, _optimization: &str, use_demo_translator: bool) -> CoelanoxResult<()> {
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
    
    // Determine target hardware
    let hardware_target = match target.to_lowercase().as_str() {
        "cpu" => HardwareTarget::Cpu,
        "gpu" => HardwareTarget::Gpu,
        "edge" => HardwareTarget::Edge,
        "auto" => {
            let recommended = env.recommended_target();
            info!("Auto-detected target: {:?}", recommended);
            recommended
        }
        _ => {
            warn!("Unknown target '{}', using auto-detection", target);
            env.recommended_target()
        }
    };
    
    info!("Target hardware: {:?}", hardware_target);
    
    // Package the model
    eprintln!("🚀 About to call packager.package()...");
    eprintln!("   Input: {}", input);
    eprintln!("   Target: {:?}", hardware_target);
    eprintln!("   Model type: {}", model_type);
    
    let container = packager.package(std::path::Path::new(input), hardware_target, model_type, input_shapes, output_shapes, use_demo_translator)?;
    eprintln!("✅ packager.package() returned successfully");
    
    // Save container to binary file using the proper output path
    container.write_binary_file(&final_output_path)?;
    
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

fn info_command(file: &str) -> CoelanoxResult<()> {
    info!("Getting info for COELANOX container: {}", file);
    
    let container = coelanox_core::format::CoelanoxContainer::read_binary_file(std::path::Path::new(file))?;
    
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
    println!();
    println!("=== Performance ===");
    // Calculate actual compression ratio (original file size / COELANOX size)
    // For now, we don't have access to original file size in info command
    // This would need to be stored in the manifest or calculated during packaging
    println!("Compression ratio: N/A (original size not available)");
    
    Ok(())
}

fn run_command(file: &str, input: Option<&str>, output: &str) -> CoelanoxResult<()> {
    info!("Running inference: {} -> {}", input.unwrap_or("auto"), output);
    
    // Load runtime
    let mut runtime = CoelanoxRuntime::new();
    
    // Load container
    runtime.load_container(file)?;
    
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
    
    // Execute inference
    let result = runtime.execute(file, &input_data)?;
    
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

fn benchmark_command(file: &str, iterations: usize) -> CoelanoxResult<()> {
    info!("Benchmarking COELANOX container: {} ({} iterations)", file, iterations);
    
    // Load runtime
    let mut runtime = CoelanoxRuntime::new();
    
    // Load container
    runtime.load_container(file)?;
    
    // Run benchmark
    let benchmark_result = runtime.benchmark(file, iterations)?;
    
    println!("=== Benchmark Results ===");
    println!("File: {}", file);
    println!("Iterations: {}", iterations);
    println!();
    println!("=== Performance ===");
    println!("Average execution time: {:.2} μs", benchmark_result.avg_time);
    println!("Min execution time: {:.2} μs", benchmark_result.min_time);
    println!("Max execution time: {:.2} μs", benchmark_result.max_time);
    println!("Standard deviation: {:.2} μs", 0.0); // TODO: Calculate std deviation
    println!();
    println!("=== Throughput ===");
    let throughput = 1_000_000.0 / benchmark_result.avg_time as f64; // ops/second
    println!("Throughput: {:.0} ops/second", throughput);
    
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
    
    // Load runtime
    let mut runtime = CoelanoxRuntime::new();
    
    // Load container
    info!("📦 Loading container: {}", file);
    runtime.load_container(file)?;
    
    let container_info = runtime.get_container_info(file)?;
    println!("=== Container Information ===");
    println!("Model: {}", container_info.manifest.name);
    println!("Input shapes: {:?}", container_info.manifest.input_shapes);
    println!("Output shapes: {:?}", container_info.manifest.output_shapes);
    println!("Machine code size: {} bytes", container_info.optimized_code.len());
    println!("Compressed weights size: {} bytes", container_info.compressed_weights.len());
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
                    "compressed_weights_size": container_info.compressed_weights.len(),
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
                    "compressed_weights_size": container_info.compressed_weights.len(),
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
break coelanox_runtime::runtime::CoelanoxRuntime::execute_machine_code
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
