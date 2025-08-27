//! SAIC Runner
//! 
//! Validates and executes SAIC containers

use clap::Parser;
use saic_core::{SaicResult};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod validation;

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

    /// Enable marketing-fast mode (maximum performance)
    #[arg(short = 'm', long = "marketing")]
    marketing: bool,
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

    // Use appropriate validation mode
    if cli.marketing {
        // Marketing-fast mode: maximum performance
        let micros = validation::marketing_fast_validation(&cli.file)?;
        println!("⚡ Marketing-fast validation: {} microseconds", micros);
        if micros < 50 {
            println!("🚀 World's fastest AI container validation!");
        }
    } else {
        // Normal or fast mode
        validation::ultra_high_throughput_validation(&cli.file, cli.audit, cli.fast)?;
    }

    if cli.audit {
        info!("SAIC file processing completed successfully");
    }

    Ok(())
} 