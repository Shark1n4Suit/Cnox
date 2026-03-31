//! Legacy CLI module (not used by the `coelanox` binary).
//!
//! The supported entry point is the `coelanox` executable (`src/main.rs`). This module remains
//! for API experiments only.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Deprecated placeholder; use the `coelanox` binary instead.
#[derive(Parser)]
#[command(name = "coelanox")]
#[command(about = "Use the `coelanox` binary from this crate (see src/main.rs).")]
#[command(version = "0.7.1")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Not implemented in this module
    Package {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "cpu")]
        target: String,
        #[arg(short, long, default_value = "onnx")]
        format: String,
        #[arg(short, long)]
        verbose: bool,
    },
    Validate {
        #[arg(short, long)]
        container: PathBuf,
        #[arg(short, long)]
        verbose: bool,
    },
    Run {
        #[arg(short, long)]
        container: PathBuf,
        #[arg(short, long)]
        input: Option<PathBuf>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long, default_value = "cpu")]
        backend: String,
        #[arg(short, long)]
        verbose: bool,
    },
    Benchmark {
        #[arg(short, long)]
        container: PathBuf,
        #[arg(short, long, default_value = "100")]
        iterations: u32,
        #[arg(short, long, default_value = "cpu")]
        backend: String,
        #[arg(short, long)]
        verbose: bool,
    },
    Info {
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Stub entry point; the real CLI is `coelanox` (`src/main.rs`).
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _cli = Cli::parse();
    eprintln!("This API path is not active. Run the `coelanox` executable built from this crate.");
    Ok(())
}
