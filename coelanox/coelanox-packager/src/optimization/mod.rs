//! Optimization parameter loading system for COELANOX
//!
//! This module provides CSV-based optimization parameter loading for hardware-specific
//! code generation optimizations.

pub mod csv_loader;
pub mod params;

pub use csv_loader::CsvOptimizationLoader;
pub use params::OptimizationParams;
