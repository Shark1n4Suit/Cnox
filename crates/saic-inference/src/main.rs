//! SAIC Inference Engine
//! 
//! Executes AI models from SAIC containers

use clap::Parser;
use saic_core::SaicResult;
use saic_core::{validation, memory};
use std::path::Path;
use std::collections::HashMap;
use saic_inference::hardware::HardwareInfo;
use saic_inference::BackendRegistry;
use image::{ImageBuffer, Rgb, DynamicImage};

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
    
    /// Image file for inference (PNG, JPEG, etc.)
    #[arg(short = 'm', long = "image")]
    image: Option<std::path::PathBuf>,
    
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

fn load_and_process_image(image_path: &Path) -> SaicResult<Vec<f32>> {
    // Load image using the image crate
    let img = image::open(image_path)
        .map_err(|e| saic_core::SaicError::InvalidFormat(format!("Failed to open image: {}", e)))?;
    
    // Resize to 224x224 (MobileNet-v2 standard input size)
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    
    // Convert to RGB and normalize to [0, 1]
    let rgb_img = resized.to_rgb8();
    let mut float_data = Vec::with_capacity(3 * 224 * 224);
    
    // MobileNet-v2 expects ImageNet normalization
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    
    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb_img.get_pixel(x, y);
            for c in 0..3 {
                let normalized = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                float_data.push(normalized);
            }
        }
    }
    
    Ok(float_data)
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
    let mut registry = BackendRegistry::new();
    
    if verbose {
        println!("Available backends: {:?}", registry.available_backends());
        println!("Registry healthy: {}", registry.is_healthy());
    }
    
    // Select backend
    let backend = if backend_name == "auto" {
        // Auto-select based on model format and hardware
        let hardware = HardwareInfo::detect();
        
        if verbose {
            println!("Detected hardware: {:?}", hardware);
        }
        
        let recommended_name = registry.get_recommended_backend(&hardware)
            .ok_or_else(|| saic_core::SaicError::InvalidFormat("No suitable backend found".to_string()))?;
        
        // Store the name to avoid borrow checker issues
        let name = recommended_name.to_string();
        registry.get_backend(&name)
            .ok_or_else(|| saic_core::SaicError::InvalidFormat(format!("Backend not found: {}", name)))?
    } else {
        // Use specified backend
        registry.get_backend(backend_name)
            .ok_or_else(|| saic_core::SaicError::InvalidFormat(format!("Backend not found: {}", backend_name)))?
    };
    
    if verbose {
        println!("Selected backend: {}", backend.name());
    }
    
    // Load model in backend if needed (especially for fallback)
    if let Err(e) = backend.load_model(saic_file.to_str().unwrap()) {
        if verbose {
            println!("⚠️  Model loading failed, trying direct execution: {}", e);
        }
        // Continue with direct execution for backends that don't need explicit loading
    }
    
    // Execute inference (backends handle their own initialization)
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_data.to_vec());
    
    let results = backend.execute(inputs).map_err(|e| saic_core::SaicError::InferenceError(e.to_string()))?;
    
    // Extract the output data
    let output_data = results.values().next()
        .ok_or_else(|| saic_core::SaicError::InvalidFormat("No output tensors found".to_string()))?;
    
    Ok(output_data.clone())
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
    } else if let Some(image_path) = &cli.image {
        load_and_process_image(image_path)?
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