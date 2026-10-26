//! Optimization parameter structures for CSV-based hardware optimization
//!
//! This module defines the structures that hold optimization parameters loaded from CSV files.

use coelanox_core::ir::OpType;
use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;

/// Helper function to convert OpType to string key
fn op_type_to_key(op_type: &OpType) -> String {
    format!("{:?}", op_type)
}

/// Hardware-specific optimization parameters loaded from CSV
///
/// These parameters control code generation optimizations like precision,
/// tile sizes, loop unrolling, and SIMD width.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationParams {
    /// Precision: FP32, FP16, INT8
    pub precision: String,
    /// Tile dimension X
    pub tile_size_x: usize,
    /// Tile dimension Y
    pub tile_size_y: usize,
    /// Loop unrolling factor
    pub unroll_factor: usize,
    /// SIMD width (8 for AVX2, 4 for SSE4.1)
    pub simd_width: usize,
    /// Expected latency in milliseconds (optional)
    pub latency: Option<f64>,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            precision: "FP32".to_string(),
            tile_size_x: 64,
            tile_size_y: 64,
            unroll_factor: 2,
            simd_width: 4, // Default to SSE4.1 (safe minimum)
            latency: None,
        }
    }
}

impl OptimizationParams {
    /// Create optimization parameters with specific values
    pub fn new(
        precision: String,
        tile_size_x: usize,
        tile_size_y: usize,
        unroll_factor: usize,
        simd_width: usize,
        latency: Option<f64>,
    ) -> Self {
        Self {
            precision,
            tile_size_x,
            tile_size_y,
            unroll_factor,
            simd_width,
            latency,
        }
    }

    /// Validate parameters
    pub fn validate(&self) -> CoelanoxResult<()> {
        // Validate precision
        match self.precision.as_str() {
            "FP32" | "FP16" | "INT8" => {}
            _ => {
                return Err(CoelanoxError::InvalidParameter(format!(
                    "Invalid precision: {}. Must be FP32, FP16, or INT8",
                    self.precision
                )))
            }
        }

        // Validate tile sizes
        if self.tile_size_x == 0 || self.tile_size_y == 0 {
            return Err(CoelanoxError::InvalidParameter(
                "Tile sizes must be greater than 0".to_string(),
            ));
        }

        // Validate unroll factor
        if self.unroll_factor == 0 {
            return Err(CoelanoxError::InvalidParameter(
                "Unroll factor must be greater than 0".to_string(),
            ));
        }

        // Validate SIMD width (must be power of 2 and reasonable)
        if self.simd_width == 0 || !self.simd_width.is_power_of_two() || self.simd_width > 16 {
            return Err(CoelanoxError::InvalidParameter(format!(
                "Invalid SIMD width: {}. Must be power of 2 between 1 and 16",
                self.simd_width
            )));
        }

        Ok(())
    }

    /// Get instruction mnemonic for precision
    pub fn precision_instruction(&self) -> &'static str {
        match self.precision.as_str() {
            "FP32" => "movss",
            "FP16" => "mov",
            "INT8" => "movb",
            _ => "movss", // Default fallback
        }
    }

    /// Get SIMD register prefix based on SIMD width
    pub fn simd_register_prefix(&self) -> &'static str {
        match self.simd_width {
            4 => "xmm",  // SSE4.1
            8 => "ymm",  // AVX2
            16 => "zmm", // AVX-512
            _ => "xmm",  // Default fallback
        }
    }

    /// Get SIMD add instruction based on SIMD width
    pub fn simd_add_instruction(&self) -> &'static str {
        match self.simd_width {
            4 => "addps",   // SSE4.1
            8 => "vaddps",  // AVX2
            16 => "vaddps", // AVX-512
            _ => "addps",   // Default fallback
        }
    }
}

/// Registry for operation-specific optimization parameters
///
/// Maps OpType to OptimizationParams for efficient lookup during code generation.
/// Uses string keys internally since OpType doesn't implement Hash.
#[derive(Debug, Clone)]
pub struct OptimizationParamsRegistry {
    /// Operation-specific parameters (keyed by string representation of OpType)
    params_map: HashMap<String, OptimizationParams>,
    /// Default parameters used when operation not found
    default_params: OptimizationParams,
}

impl OptimizationParamsRegistry {
    /// Create new registry with default parameters
    pub fn new() -> Self {
        Self {
            params_map: HashMap::new(),
            default_params: OptimizationParams::default(),
        }
    }

    /// Create registry with custom default parameters
    pub fn with_defaults(default_params: OptimizationParams) -> Self {
        Self {
            params_map: HashMap::new(),
            default_params,
        }
    }

    /// Register parameters for an operation
    pub fn register(&mut self, op_type: OpType, params: OptimizationParams) -> CoelanoxResult<()> {
        params.validate()?;
        let key = op_type_to_key(&op_type);
        self.params_map.insert(key, params);
        Ok(())
    }

    /// Get parameters for an operation
    ///
    /// Returns operation-specific parameters if available, otherwise returns defaults.
    pub fn get_params(&self, op_type: &OpType) -> &OptimizationParams {
        let key = op_type_to_key(op_type);
        self.params_map.get(&key).unwrap_or(&self.default_params)
    }

    /// Check if operation has specific parameters registered
    pub fn has_params(&self, op_type: &OpType) -> bool {
        let key = op_type_to_key(op_type);
        self.params_map.contains_key(&key)
    }

    /// Set default parameters
    pub fn set_defaults(&mut self, params: OptimizationParams) -> CoelanoxResult<()> {
        params.validate()?;
        self.default_params = params;
        Ok(())
    }

    /// Get default parameters
    pub fn defaults(&self) -> &OptimizationParams {
        &self.default_params
    }

    /// Get all registered operations (as strings - caller must parse)
    pub fn registered_operations(&self) -> Vec<String> {
        self.params_map.keys().cloned().collect()
    }
}

impl Default for OptimizationParamsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_params_default() {
        let params = OptimizationParams::default();
        assert_eq!(params.precision, "FP32");
        assert_eq!(params.tile_size_x, 64);
        assert_eq!(params.simd_width, 4);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_optimization_params_validation() {
        // Valid params
        let params = OptimizationParams::new("FP32".to_string(), 32, 32, 4, 8, None);
        assert!(params.validate().is_ok());

        // Invalid precision
        let params = OptimizationParams::new("INVALID".to_string(), 32, 32, 4, 8, None);
        assert!(params.validate().is_err());

        // Invalid tile size
        let params = OptimizationParams::new("FP32".to_string(), 0, 32, 4, 8, None);
        assert!(params.validate().is_err());

        // Invalid SIMD width
        let params = OptimizationParams::new(
            "FP32".to_string(),
            32,
            32,
            4,
            5,
            None, // Not power of 2
        );
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_precision_instructions() {
        let fp32 = OptimizationParams::new("FP32".to_string(), 32, 32, 4, 8, None);
        assert_eq!(fp32.precision_instruction(), "movss");

        let fp16 = OptimizationParams::new("FP16".to_string(), 32, 32, 4, 8, None);
        assert_eq!(fp16.precision_instruction(), "mov");

        let int8 = OptimizationParams::new("INT8".to_string(), 32, 32, 4, 8, None);
        assert_eq!(int8.precision_instruction(), "movb");
    }

    #[test]
    fn test_simd_register_prefix() {
        let sse = OptimizationParams::new("FP32".to_string(), 32, 32, 4, 4, None);
        assert_eq!(sse.simd_register_prefix(), "xmm");

        let avx2 = OptimizationParams::new("FP32".to_string(), 32, 32, 4, 8, None);
        assert_eq!(avx2.simd_register_prefix(), "ymm");

        let avx512 = OptimizationParams::new("FP32".to_string(), 32, 32, 4, 16, None);
        assert_eq!(avx512.simd_register_prefix(), "zmm");
    }

    #[test]
    fn test_optimization_params_registry() {
        let mut registry = OptimizationParamsRegistry::new();

        // Register params for Add operation
        let add_params = OptimizationParams::new("FP32".to_string(), 64, 64, 2, 8, Some(0.1));
        registry.register(OpType::Add, add_params.clone()).unwrap();

        // Get params for Add
        let retrieved = registry.get_params(&OpType::Add);
        assert_eq!(retrieved.precision, add_params.precision);
        assert_eq!(retrieved.simd_width, add_params.simd_width);

        // Get params for unregistered operation (should return defaults)
        let default = registry.get_params(&OpType::Multiply);
        assert_eq!(default.precision, "FP32");
        assert!(registry.has_params(&OpType::Add));
        assert!(!registry.has_params(&OpType::Multiply));
    }
}
