//! SAIC Inference Engine
//! 
//! Executes AI models from SAIC containers

use clap::Parser;
use saic_core::SaicResult;
use saic_inference::BackendRegistry;
use saic_core::{validation, memory};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    name = "saic-inference",
    version = "0.5.0",
    about = "SAIC inference engine for executing AI models"
)]
struct Cli {
    /// Path to the .saic file to execute
    #[arg(value_name = "FILE")]
    file: std::path::PathBuf,
    
    /// Input data for inference (comma-separated floats)
    #[arg(short = 'i', long = "input")]
    input: Option<String>,
    
    /// Backend to use (auto, fallback, openvino, tensorrt)
    #[arg(short = 'b', long = "backend", default_value = "auto")]
    backend: String,
    
    /// Enable verbose logging
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn parse_input_data(input_str: &str) -> Vec<f32> {
    input_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

fn execute_inference(saic_file: &Path, input_data: &[f32], backend_name: &str, verbose: bool) -> SaicResult<Vec<f32>> {
    // Load SAIC file
    let mmap = memory::map_file(saic_file)?;
    
    // Extract manifest and model payload
    let manifest = validation::extract_manifest(&mmap)?;
    let model_payload = validation::extract_model_payload(&mmap)?;
    
    // Get model format from manifest
    let framework = manifest["framework"].as_str()
        .unwrap_or("unknown");
    
    if verbose {
        println!("Model format: {}", framework);
        println!("Model payload size: {} bytes", model_payload.len());
        println!("Input data size: {} floats", input_data.len());
    }
    
    // Create backend registry
    let registry = BackendRegistry::new();
    
    // Select backend
    let backend = if backend_name == "auto" {
        // Auto-select based on model format and hardware
        let hardware = saic_inference::hardware::HardwareInfo::detect();
        registry.select_backend(framework, &hardware)
    } else {
        // Use specified backend
        registry.get_backend(backend_name)
            .ok_or_else(|| saic_core::SaicError::InvalidFormat(format!("Backend not found: {}", backend_name)))?
    };
    
    if verbose {
        println!("Selected backend: {}", backend.name());
        println!("Available backends: {:?}", registry.available_backends());
    }
    
    // Execute inference (backends handle their own initialization)
    let results = backend.execute(model_payload, input_data).map_err(|e| saic_core::SaicError::InferenceError(e.to_string()))?;
    
    Ok(results)
}

fn main() -> SaicResult<()> {
    let cli = Cli::parse();
    
    if cli.verbose {
        println!("SAIC Inference Engine v0.5.0");
        println!("File: {}", cli.file.display());
        println!("Backend: {}", cli.backend);
    }
    
    // Parse input data or use default
    let input_data = if let Some(input_str) = &cli.input {
        parse_input_data(input_str)
    } else {
        // Default input data for testing
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    };
    
    // Execute inference
    match execute_inference(&cli.file, &input_data, &cli.backend, cli.verbose) {
        Ok(results) => {
            println!("✅ Inference completed successfully!");
            println!("📊 Results (first 5): {:?}", &results[..5.min(results.len())]);
            println!("📈 Total outputs: {}", results.len());
        }
        Err(e) => {
            eprintln!("❌ Inference failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
} 