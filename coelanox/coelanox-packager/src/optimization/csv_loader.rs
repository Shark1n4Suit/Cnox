//! CSV-based optimization parameter loader for COELANOX
//! 
//! This module provides functionality to load and parse CSV files containing
//! hardware-specific optimization parameters for code generation.

use std::path::Path;
use std::fs::File;
use coelanox_core::ir::OpType;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use crate::optimization::params::{OptimizationParams, OptimizationParamsRegistry};

/// CSV optimization parameter loader
/// 
/// Loads and parses CSV files containing hardware-specific optimization parameters.
/// Supports vendor-provided CSVs, default fallbacks, and embedded data.
#[derive(Debug)]
pub struct CsvOptimizationLoader {
    /// Registry of loaded optimization parameters
    registry: OptimizationParamsRegistry,
    /// Source of the loaded data (for debugging)
    source: String,
}

impl CsvOptimizationLoader {
    /// Create a new CSV loader with default parameters
    pub fn new() -> Self {
        Self {
            registry: OptimizationParamsRegistry::new(),
            source: "defaults".to_string(),
        }
    }
    
    /// Create loader from CSV file path
    /// 
    /// Attempts to load CSV from:
    /// 1. Vendor-provided: `coelanox/data/vendor/{vendor}/{hardware}.csv`
    /// 2. Default fallback: `coelanox/data/default/cpu.csv`
    /// 3. Embedded option: compile-time embedded data
    pub fn from_file<P: AsRef<Path>>(path: P) -> CoelanoxResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        let mut loader = Self::new();
        loader.load_from_reader(file, path.to_string_lossy().to_string())?;
        loader.source = path.to_string_lossy().to_string();
        
        Ok(loader)
    }
    
    /// Load CSV from reader
    fn load_from_reader<R: std::io::Read>(&mut self, reader: R, _source: String) -> CoelanoxResult<()> {
        let mut csv_reader = csv::Reader::from_reader(reader);
        
        for result in csv_reader.records() {
            let record = result.map_err(|e| CoelanoxError::InvalidParameter(
                format!("Failed to parse CSV record: {}", e)
            ))?;
            
            if record.len() < 6 {
                continue; // Skip invalid rows
            }
            
            // Parse operation name
            let op_name = record.get(0).ok_or_else(|| CoelanoxError::InvalidParameter(
                "Missing operation name in CSV".to_string()
            ))?;
            
            // Parse precision
            let precision = record.get(1).ok_or_else(|| CoelanoxError::InvalidParameter(
                "Missing precision in CSV".to_string()
            ))?;
            
            // Parse tile_size_x
            let tile_size_x: usize = record.get(2)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CoelanoxError::InvalidParameter(
                    format!("Invalid tile_size_x in CSV: {}", record.get(2).unwrap_or(""))
                ))?;
            
            // Parse tile_size_y
            let tile_size_y: usize = record.get(3)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CoelanoxError::InvalidParameter(
                    format!("Invalid tile_size_y in CSV: {}", record.get(3).unwrap_or(""))
                ))?;
            
            // Parse unroll_factor
            let unroll_factor: usize = record.get(4)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CoelanoxError::InvalidParameter(
                    format!("Invalid unroll_factor in CSV: {}", record.get(4).unwrap_or(""))
                ))?;
            
            // Parse simd_width
            let simd_width: usize = record.get(5)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| CoelanoxError::InvalidParameter(
                    format!("Invalid simd_width in CSV: {}", record.get(5).unwrap_or(""))
                ))?;
            
            // Parse latency (optional)
            let latency = record.get(6)
                .and_then(|s| {
                    // Remove "ms" suffix if present
                    let s = s.trim().trim_end_matches("ms").trim();
                    s.parse::<f64>().ok()
                });
            
        // Convert operation name to OpType
        let op_type = Self::parse_operation_name(op_name)?;
        
        // Create optimization parameters
        let params = OptimizationParams::new(
            precision.to_string(),
            tile_size_x,
            tile_size_y,
            unroll_factor,
            simd_width,
            latency,
        );
        
        // Validate and register
        params.validate()?;
        // Clone op_type before moving it
        let op_type_clone = op_type.clone();
        self.registry.register(op_type_clone, params)?;
        }
        
        Ok(())
    }
    
    /// Parse operation name string to OpType
    fn parse_operation_name(name: &str) -> CoelanoxResult<OpType> {
        match name.trim().to_lowercase().as_str() {
            "add" => Ok(OpType::Add),
            "subtract" => Ok(OpType::Subtract),
            "multiply" => Ok(OpType::Multiply),
            "divide" => Ok(OpType::Divide),
            "conv" | "convolution" => Ok(OpType::Convolution),
            "matmul" | "matrix_multiply" | "gemm" => Ok(OpType::MatMul),
            "maxpool" | "max_pool" => Ok(OpType::MaxPool),
            "avgpool" | "avg_pool" => Ok(OpType::AvgPool),
            "relu" => Ok(OpType::Relu),
            "sigmoid" => Ok(OpType::Sigmoid),
            "tanh" => Ok(OpType::Tanh),
            "gelu" => Ok(OpType::Gelu),
            "swish" => Ok(OpType::Swish),
            "batchnorm" | "batch_norm" => Ok(OpType::BatchNorm),
            "layernorm" | "layer_norm" => Ok(OpType::LayerNorm),
            "transpose" => Ok(OpType::Transpose),
            "reshape" => Ok(OpType::Reshape),
            _ => Err(CoelanoxError::InvalidParameter(
                format!("Unknown operation name in CSV: {}", name)
            )),
        }
    }
    
    /// Load CSV from embedded/default data
    /// 
    /// Creates default parameters for common operations.
    pub fn from_defaults() -> Self {
        let mut loader = Self::new();
        loader.source = "embedded_defaults".to_string();
        
        // Register default parameters for common operations
        let default_ops = vec![
            (OpType::Add, OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.1))),
            (OpType::Multiply, OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.1))),
            (OpType::Convolution, OptimizationParams::new("FP32".to_string(), 32, 32, 4, 8, Some(0.5))),
            (OpType::MatMul, OptimizationParams::new("FP32".to_string(), 256, 256, 2, 8, Some(1.2))),
            (OpType::MaxPool, OptimizationParams::new("FP32".to_string(), 32, 32, 2, 8, Some(0.2))),
            (OpType::AvgPool, OptimizationParams::new("FP32".to_string(), 32, 32, 2, 8, Some(0.2))),
            (OpType::Relu, OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.05))),
            (OpType::Sigmoid, OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.1))),
            (OpType::Tanh, OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.1))),
        ];
        
        for (op_type, params) in default_ops {
            if let Err(e) = params.validate() {
                tracing::warn!("Invalid default params for {:?}: {}", op_type, e);
                continue;
            }
            let op_type_clone = op_type.clone();
            if let Err(e) = loader.registry.register(op_type_clone, params) {
                tracing::warn!("Failed to register default params for {:?}: {}", op_type, e);
            }
        }
        
        loader
    }
    
    /// Try to load CSV from multiple locations
    /// 
    /// Attempts to load in order:
    /// 1. Vendor-specific path: `data/vendor/{vendor}/{hardware}.csv`
    /// 2. Default path: `data/default/cpu.csv`
    /// 3. Embedded defaults
    pub fn load_with_fallback<P: AsRef<Path>>(base_path: P, vendor: Option<&str>, hardware: Option<&str>) -> Self {
        let base_path = base_path.as_ref();
        
        // Try vendor-specific path first
        if let (Some(vendor), Some(hardware)) = (vendor, hardware) {
            let vendor_path = base_path.join("vendor").join(vendor).join(format!("{}.csv", hardware));
            if vendor_path.exists() {
                match Self::from_file(&vendor_path) {
                    Ok(loader) => {
                        tracing::info!("Loaded optimization CSV from vendor path: {:?}", vendor_path);
                        return loader;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load vendor CSV {:?}: {}, falling back", vendor_path, e);
                    }
                }
            }
        }
        
        // Try default path
        let default_path = base_path.join("default").join("cpu.csv");
        if default_path.exists() {
            match Self::from_file(&default_path) {
                Ok(loader) => {
                    tracing::info!("Loaded optimization CSV from default path: {:?}", default_path);
                    return loader;
                }
                Err(e) => {
                    tracing::warn!("Failed to load default CSV {:?}: {}, using embedded defaults", default_path, e);
                }
            }
        }
        
        // Fall back to embedded defaults
        tracing::info!("Using embedded default optimization parameters");
        Self::from_defaults()
    }
    
    /// Get optimization parameters for an operation
    /// 
    /// Returns operation-specific parameters if available, otherwise returns defaults.
    pub fn get_params(&self, op_type: &OpType) -> &OptimizationParams {
        self.registry.get_params(op_type)
    }
    
    /// Check if operation has specific parameters registered
    pub fn has_params(&self, op_type: &OpType) -> bool {
        self.registry.has_params(op_type)
    }
    
    /// Get source of the loaded data (for debugging)
    pub fn source(&self) -> &str {
        &self.source
    }
    
    /// Get the underlying registry (for advanced use cases)
    pub fn registry(&self) -> &OptimizationParamsRegistry {
        &self.registry
    }
}

impl Default for CsvOptimizationLoader {
    fn default() -> Self {
        Self::from_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_load_from_defaults() {
        let loader = CsvOptimizationLoader::from_defaults();
        
        // Check that default operations are registered
        assert!(loader.has_params(&OpType::Add));
        assert!(loader.has_params(&OpType::Convolution));
        assert!(loader.has_params(&OpType::MatMul));
        
        // Check that default parameters are valid
        let params = loader.get_params(&OpType::Add);
        assert_eq!(params.precision, "FP32");
        assert_eq!(params.simd_width, 8);
    }
    
    #[test]
    fn test_load_from_csv_file() {
        // Create temporary CSV file
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "operation,precision,tile_size_x,tile_size_y,unroll_factor,simd_width,latency").unwrap();
        writeln!(file, "Add,FP32,64,64,2,8,0.1ms").unwrap();
        writeln!(file, "Conv,FP32,32,32,4,8,0.5ms").unwrap();
        writeln!(file, "MatMul,FP32,256,256,2,8,1.2ms").unwrap();
        file.flush().unwrap();
        
        let loader = CsvOptimizationLoader::from_file(file.path()).unwrap();
        
        // Check that operations are loaded
        assert!(loader.has_params(&OpType::Add));
        assert!(loader.has_params(&OpType::Convolution));
        assert!(loader.has_params(&OpType::MatMul));
        
        // Check parameter values
        let add_params = loader.get_params(&OpType::Add);
        assert_eq!(add_params.precision, "FP32");
        assert_eq!(add_params.tile_size_x, 64);
        assert_eq!(add_params.simd_width, 8);
        assert_eq!(add_params.latency, Some(0.1));
        
        let conv_params = loader.get_params(&OpType::Convolution);
        assert_eq!(conv_params.tile_size_x, 32);
        assert_eq!(conv_params.unroll_factor, 4);
    }
    
    #[test]
    fn test_parse_operation_name() {
        assert_eq!(CsvOptimizationLoader::parse_operation_name("Add").unwrap(), OpType::Add);
        assert_eq!(CsvOptimizationLoader::parse_operation_name("conv").unwrap(), OpType::Convolution);
        assert_eq!(CsvOptimizationLoader::parse_operation_name("MATMUL").unwrap(), OpType::MatMul);
        assert_eq!(CsvOptimizationLoader::parse_operation_name("maxpool").unwrap(), OpType::MaxPool);
        assert!(CsvOptimizationLoader::parse_operation_name("InvalidOp").is_err());
    }
    
    #[test]
    fn test_fallback_to_defaults() {
        // Use non-existent path - should fall back to defaults
        let loader = CsvOptimizationLoader::load_with_fallback("/nonexistent/path", None, None);
        
        // Should still have default parameters
        assert!(loader.has_params(&OpType::Add));
        let params = loader.get_params(&OpType::Add);
        assert_eq!(params.precision, "FP32");
    }
}

