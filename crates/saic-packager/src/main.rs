//! SAIC Packager
//! 
//! Creates SAIC containers from AI models

use clap::Parser;
use saic_core::{SaicError, SaicResult, format::SaicManifest};
use std::fs::File;
use std::io::{Write, BufReader, BufWriter, copy};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(
    name = "saic-packager",
    version = "0.5.0",
    about = "SAIC packager for creating AI model containers"
)]
struct Cli {
    /// Path to the input model file
    #[arg(long)]
    model: std::path::PathBuf,

    /// Name for the model (stored in manifest)
    #[arg(long)]
    name: String,

    /// Output path for the .saic file
    #[arg(long)]
    output: std::path::PathBuf,

    /// Enable verbose logging
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn init_logging() -> SaicResult<()> {
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .init();
    Ok(())
}

fn create_saic_file(args: &Cli) -> SaicResult<()> {
    info!("Starting SAIC file creation process");
    
    // Read the input model file
    let model_file = File::open(&args.model)
        .map_err(|e| SaicError::Io(e))?;
    
    let model_metadata = model_file.metadata()
        .map_err(|e| SaicError::Io(e))?;
    
    info!("Model file size: {} bytes", model_metadata.len());
    
    // Create manifest
    let manifest = SaicManifest {
        model_name: args.name.clone(),
        model_version: None,
        framework: None,
        input_shape: None,
        output_shape: None,
        optimization_hints: None,
    };
    
    let manifest_json = serde_json::to_string(&manifest)
        .map_err(|e| SaicError::ManifestError(e))?;
    
    let manifest_bytes = manifest_json.as_bytes();
    let manifest_length = manifest_bytes.len() as u32;
    
    info!("Manifest size: {} bytes", manifest_length);
    
    // Create output file
    let output_file = File::create(&args.output)
        .map_err(|e| SaicError::Io(e))?;
    let mut writer = BufWriter::new(output_file);
    
    // Write SAIC header
    writer.write_all(saic_core::MAGIC_NUMBER)
        .map_err(|e| SaicError::Io(e))?;
    writer.write_all(&saic_core::FORMAT_VERSION)
        .map_err(|e| SaicError::Io(e))?;
    
    // Write manifest length
    writer.write_all(&manifest_length.to_be_bytes())
        .map_err(|e| SaicError::Io(e))?;
    
    // Write manifest
    writer.write_all(manifest_bytes)
        .map_err(|e| SaicError::Io(e))?;
    
    // Copy model payload
    info!("Copying model payload");
    let mut model_reader = BufReader::new(File::open(&args.model)
        .map_err(|e| SaicError::Io(e))?);
    
    let bytes_copied = copy(&mut model_reader, &mut writer)
        .map_err(|e| SaicError::Io(e))?;
    
    info!("Model payload copied: {} bytes", bytes_copied);
    
    // Flush writer to ensure all data is written
    writer.flush()
        .map_err(|e| SaicError::Io(e))?;
    
    info!("SAIC file creation completed successfully");
    
    // Log final file structure
    let output_metadata = std::fs::metadata(&args.output)
        .map_err(|e| SaicError::Io(e))?;
    
    info!("Final SAIC file structure:");
    info!("  Header: {} bytes (magic + version)", saic_core::SaicFormat::HEADER_SIZE);
    info!("  Manifest length: {} bytes", saic_core::SaicFormat::MANIFEST_LENGTH_SIZE);
    info!("  Manifest: {} bytes", manifest_length);
    info!("  Model payload: {} bytes", bytes_copied);
    info!("  Total file size: {} bytes", output_metadata.len());
    
    Ok(())
}

fn main() -> SaicResult<()> {
    let cli = Cli::parse();
    
    if cli.verbose {
        init_logging()?;
    }
    
    info!("SAIC Packager v0.5.0");
    info!("Model: {}", cli.model.display());
    info!("Name: {}", cli.name);
    info!("Output: {}", cli.output.display());
    
    create_saic_file(&cli)?;
    
    println!("✅ SAIC file created successfully!");
    println!("📦 File: {}", cli.output.display());
    
    Ok(())
} 