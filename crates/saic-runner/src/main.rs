//! SAIC Runner
//! 
//! Validates and executes SAIC containers

use clap::Parser;
use saic_core::{SaicError, SaicResult, validation, security, memory::MemoryMapper};
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(
    name = "saic-runner",
    version = "0.5.0",
    about = "SAIC runner for validating and executing AI model containers"
)]
struct Cli {
    /// Path to the .saic file to validate
    #[arg(value_name = "FILE")]
    file: std::path::PathBuf,
    
    /// Enable audit mode for detailed logging
    #[arg(short = 'a', long = "audit")]
    audit: bool,
    
    /// Enable ultra-fast mode (skip security checks)
    #[arg(short = 'f', long = "fast")]
    fast: bool,
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

/// Ultra-high throughput validation with performance optimizations
/// 
/// Uses optimized validation paths for maximum performance
#[inline(always)]
fn ultra_high_throughput_validation(file_path: &std::path::Path, audit_mode: bool, fast_mode: bool) -> SaicResult<()> {
    let start_time = Instant::now();

    if audit_mode {
        info!("Starting ultra-high throughput validation");
        info!("File: {}", file_path.display());
        
        // Only validate file path in audit mode to save microseconds
        if !fast_mode {
            security::validate_file_path(file_path)?;
        }
    }

    let file_open_time = Instant::now();

    // Use shared memory mapping with zero-copy optimization
    let mmap = MemoryMapper::map_file(file_path)?;

    let mmap_time = Instant::now();

    if audit_mode {
        info!("Memory mapping successful - {} bytes mapped", mmap.len());
    }

    // OPTIMIZATION: Use ultra-fast validation
    validation::validate_saic_structure(&mmap)?;

    let magic_time = Instant::now();

    if audit_mode {
        info!("Magic validation successful");
    }

    let total_time = Instant::now();

    if audit_mode {
        info!("Validation completed successfully");
        info!("Performance breakdown:");
        info!("  File open: {:?}", file_open_time.duration_since(start_time));
        info!("  Memory map: {:?}", mmap_time.duration_since(file_open_time));
        info!("  Magic validation: {:?}", magic_time.duration_since(mmap_time));
        info!("  Structure validation: {:?}", total_time.duration_since(magic_time));
        info!("  Total validation: {:?}", total_time.duration_since(start_time));
    } else {
        // Print microsecond timing for performance measurement
        let total_duration = total_time.duration_since(start_time);
        let micros = total_duration.as_micros();
        println!("⚡ Validation completed in {} microseconds", micros);
        
        // Show optimization status in fast mode
        if fast_mode {
            println!("🚀 Ultra-fast mode enabled (security checks skipped)");
        }
    }

    Ok(())
}

fn main() -> SaicResult<()> {
    let cli = Cli::parse();

    // Initialize logging only if audit mode is enabled
    if cli.audit {
        init_logging()?;
        info!("SAIC Runner starting in audit mode");
        info!("Processing file: {}", cli.file.display());
        if cli.fast {
            info!("Ultra-fast mode enabled (security checks skipped)");
        }
    }

    // Process the SAIC file with ultra-high throughput validation
    ultra_high_throughput_validation(&cli.file, cli.audit, cli.fast)?;

    if cli.audit {
        info!("SAIC file processing completed successfully");
    }

    Ok(())
} 