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
    let subscriber = FmtSubscriber::builder()
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
    info!("Model: {}", args.model.display());
    info!("Output: {}", args.output.display());
    
    // Step 1: Create the JSON manifest using shared format
    info!("Creating JSON manifest");
    let manifest = SaicManifest {
        model_name: args.name.clone(),
        model_version: None,
        framework: None,
        input_shape: None,
        output_shape: None,
        optimization_hints: None,
    };
    
    // Step 2: Serialize manifest to UTF-8 string
    info!("Serializing manifest to JSON");
    let manifest_string = serde_json::to_string(&manifest)
        .map_err(SaicError::ManifestError)?;
    let manifest_bytes = manifest_string.as_bytes();
    
    // SECURITY: Prevent DoS attacks through large manifests
    if manifest_bytes.len() > 1024 * 1024 { // 1MB limit
        return Err(SaicError::InvalidFormat(
            "Manifest too large (maximum 1MB allowed)".to_string()
        ));
    }
    
    info!("Manifest size: {} bytes", manifest_bytes.len());
    
    // Step 3: Open output file for writing
    info!("Opening output file for writing");
    let mut output_file = File::create(&args.output)
        .map_err(SaicError::Io)?;
    
    // Step 4: Write the 8-byte SAIC header using shared constants
    info!("Writing SAIC header");
    
    // Write magic number "SAIC" (4 bytes)
    output_file.write_all(saic_core::MAGIC_NUMBER)
        .map_err(SaicError::Io)?;
    
    // Write version 0.1.0.0 (4 bytes: major.minor.patch.build)
    output_file.write_all(&saic_core::FORMAT_VERSION)
        .map_err(SaicError::Io)?;
    
    // Step 5: Write manifest length as 32-bit big-endian unsigned integer
    info!("Writing manifest length");
    let manifest_length = manifest_bytes.len() as u32;
    let length_bytes = manifest_length.to_be_bytes(); // big-endian format
    output_file.write_all(&length_bytes)
        .map_err(SaicError::Io)?;
    
    // Step 6: Write the JSON manifest string
    info!("Writing JSON manifest");
    output_file.write_all(manifest_bytes)
        .map_err(SaicError::Io)?;
    
    // Step 7: Copy the input model file contents with optimized I/O
    info!("Copying model payload");
    let input_file = File::open(&args.model)
        .map_err(SaicError::Io)?;
    
    // Use Rust's optimized copy with large buffers for maximum performance
    let mut reader = BufReader::with_capacity(64 * 1024, input_file); // 64KB buffer
    let mut writer = BufWriter::with_capacity(64 * 1024, output_file);
    let bytes_copied = copy(&mut reader, &mut writer)
            .map_err(SaicError::Io)?;
    
    // Ensure all data is written to disk
    writer.flush().map_err(SaicError::Io)?;
    
    info!("Model payload copied: {} bytes", bytes_copied);
    info!("SAIC file creation completed successfully");
    
    // Log final file structure summary
    info!("Final SAIC file structure:");
    info!("  Header: 8 bytes (magic + version)");
    info!("  Manifest length: 4 bytes");
    info!("  Manifest: {} bytes", manifest_bytes.len());
    info!("  Model payload: {} bytes", bytes_copied);
    info!("  Total file size: {} bytes", 8 + 4 + manifest_bytes.len() as u64 + bytes_copied);
    
    Ok(())
}

fn main() -> SaicResult<()> {
    let cli = Cli::parse();
    
    // Initialize logging if verbose mode
    if cli.verbose {
        init_logging()?;
    }
    
    info!("SAIC Packager v0.5.0");
    info!("Model: {}", cli.model.display());
    info!("Name: {}", cli.name);
    info!("Output: {}", cli.output.display());
    
    // Create the SAIC file
    create_saic_file(&cli)?;
    
    println!("✅ SAIC file created successfully!");
    println!("📦 File: {}", cli.output.display());
    
    Ok(())
} 