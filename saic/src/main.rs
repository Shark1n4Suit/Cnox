//! SAIC CLI - Command line interface for SAIC
//! 
//! This module provides the command line interface for the SAIC toolchain.

use clap::{Parser, Subcommand};
use tracing::{info, warn, error};
use saic::{SaicError, SaicResult, SaicPackager, SaicRuntime, format::HardwareTarget};

#[derive(Parser)]
#[command(name = "saic")]
#[command(about = "SAIC - Secure AI Container")]
#[command(version = "0.5.0-beta")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Package a model into a SAIC container
    Package {
        /// Input model file
        #[arg(short, long)]
        input: String,
        
        /// Output SAIC file
        #[arg(short, long)]
        output: String,
        
        /// Target hardware (cpu, gpu, edge, auto)
        #[arg(short, long, default_value = "auto")]
        target: String,
        
        /// Optimization level (none, basic, advanced, maximum)
        #[arg(long, default_value = "advanced")]
        optimization: String,
    },
    
    /// Validate a SAIC container
    Validate {
        /// SAIC file to validate
        #[arg(short, long)]
        file: String,
    },
    
    /// Show information about a SAIC container
    Info {
        /// SAIC file to inspect
        #[arg(short, long)]
        file: String,
    },
    
    /// Run inference on a SAIC container
    Run {
        /// SAIC file to run
        #[arg(short, long)]
        file: String,
        
        /// Input data file (JSON)
        #[arg(short, long)]
        input: Option<String>,
        
        /// Output file
        #[arg(short, long)]
        output: String,
    },
    
    /// Benchmark a SAIC container
    Benchmark {
        /// SAIC file to benchmark
        #[arg(short, long)]
        file: String,
        
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
    },
    
    /// Show system environment information
    Env,
}

fn main() -> SaicResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Package { input, output, target, optimization } => {
            package_command(&input, &output, &target, &optimization)
        }
        Commands::Validate { file } => {
            validate_command(&file)
        }
        Commands::Info { file } => {
            info_command(&file)
        }
        Commands::Run { file, input, output } => {
            run_command(&file, input.as_deref(), &output)
        }
        Commands::Benchmark { file, iterations } => {
            benchmark_command(&file, iterations)
        }
        Commands::Env => {
            env_command()
        }
    }
}

fn package_command(input: &str, output: &str, target: &str, optimization: &str) -> SaicResult<()> {
    info!("Packaging model: {} -> {}", input, output);
    
    // Create packager with environment detection
    let packager = SaicPackager::new()?;
    
    // Get environment info
    let env = packager.get_environment();
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
    let container = packager.package(std::path::Path::new(input), hardware_target)?;
    
    // Save container to binary file
    container.write_binary_file(std::path::Path::new(output))?;
    
    info!("✅ Container packaged successfully");
    info!("Container saved to: {}", output);
    info!("Container size: {} bytes", container.total_size());
    
    // Show compression info
    let input_size = std::fs::metadata(input)?.len();
    let compression_ratio = input_size as f64 / container.total_size() as f64;
    info!("Compression ratio: {:.1}x", compression_ratio);
    
    Ok(())
}

fn validate_command(file: &str) -> SaicResult<()> {
    info!("Validating SAIC container: {}", file);
    
    let container = saic::format::SaicContainer::read_binary_file(std::path::Path::new(file))?;
    
    println!("✅ Container is valid");
    println!("Version: 0.5.0-beta");
    println!("Model: {}", container.manifest.name);
    println!("Format: {:?}", container.manifest.format);
    println!("Size: {} bytes", container.total_size());
    
    Ok(())
}

fn info_command(file: &str) -> SaicResult<()> {
    info!("Getting info for SAIC container: {}", file);
    
    let container = saic::format::SaicContainer::read_binary_file(std::path::Path::new(file))?;
    
    println!("=== SAIC Container Information ===");
    println!("File: {}", file);
    println!("Version: 0.5.0-beta");
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
    println!("Header size: {} bytes", saic::format::SAIC_HEADER_SIZE);
    println!("Manifest size: ~{} bytes", serde_json::to_string(&container.manifest).unwrap_or_default().len());
    println!("Machine code size: {} bytes", container.optimized_code.len());
    println!();
    println!("=== Performance ===");
    // Calculate actual compression ratio (original file size / SAIC size)
    // For now, we don't have access to original file size in info command
    // This would need to be stored in the manifest or calculated during packaging
    println!("Compression ratio: N/A (original size not available)");
    
    Ok(())
}

fn run_command(file: &str, input: Option<&str>, output: &str) -> SaicResult<()> {
    info!("Running inference: {} -> {}", input.unwrap_or("auto"), output);
    
    // Load runtime
    let mut runtime = saic::SaicRuntime::new();
    
    // Load container
    runtime.load_container(file)?;
    
    // Load input data
    let input_data = if let Some(input_file) = input {
        if input_file.ends_with(".json") {
            // Load from JSON file
            let json_str = std::fs::read_to_string(input_file)
                .map_err(|e| saic::SaicError::FileSystem(e))?;
            serde_json::from_str::<Vec<f32>>(&json_str)
                .map_err(|e| saic::SaicError::format(&format!("Invalid JSON input: {}", e)))?
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
        .map_err(|e| saic::SaicError::format(&format!("Failed to serialize output: {}", e)))?;
    
    std::fs::write(output, output_json)
        .map_err(|e| saic::SaicError::FileSystem(e))?;
    
    info!("✅ Inference completed successfully");
    println!("Input size: {} elements", input_data.len());
    println!("Output size: {} elements", result.len());
    println!("Execution time: {} μs", runtime.get_stats().avg_execution_time);
    
    Ok(())
}

fn benchmark_command(file: &str, iterations: usize) -> SaicResult<()> {
    info!("Benchmarking SAIC container: {} ({} iterations)", file, iterations);
    
    // Load runtime
    let mut runtime = saic::SaicRuntime::new();
    
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

fn env_command() -> SaicResult<()> {
    info!("Detecting system environment");
    
    let env = saic::utils::SystemEnvironment::detect()?;
    
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
