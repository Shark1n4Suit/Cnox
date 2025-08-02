//! SAIC file format specification

use serde::{Deserialize, Serialize};
use crate::{MAGIC_NUMBER, FORMAT_VERSION, MAX_MANIFEST_SIZE};

/// SAIC file format specification
/// 
/// This module defines the binary format for SAIC files according to
/// the SAIC v0.1 specification.
pub struct SaicFormat;

impl SaicFormat {
    /// SAIC file header size (8 bytes)
    pub const HEADER_SIZE: usize = 8;
    
    /// Manifest length field size (4 bytes)
    pub const MANIFEST_LENGTH_SIZE: usize = 4;
    
    /// Total header size including manifest length (12 bytes)
    pub const TOTAL_HEADER_SIZE: usize = Self::HEADER_SIZE + Self::MANIFEST_LENGTH_SIZE;
}

/// SAIC file manifest structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaicManifest {
    /// Model name
    pub model_name: String,
    
    /// Model version (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    
    /// Framework (e.g., "ONNX", "PyTorch", "TensorFlow")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
    
    /// Input shape (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_shape: Option<Vec<usize>>,
    
    /// Output shape (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_shape: Option<Vec<usize>>,
    
    /// Optimization hints (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_hints: Option<OptimizationHints>,
}

/// Hardware optimization hints for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// SIMD instructions available
    pub simd_instructions: Vec<String>,
    
    /// Recommended thread count
    pub recommended_threads: usize,
    
    /// Memory alignment requirements
    pub memory_alignment: usize,
    
    /// Cache line size
    pub cache_line_size: usize,
    
    /// Recommended batch size
    pub batch_size: usize,
    
    /// Security level (1-10)
    pub security_level: u8,
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            simd_instructions: vec!["AVX2".to_string()],
            recommended_threads: 4,
            memory_alignment: 32,
            cache_line_size: 64,
            batch_size: 8,
            security_level: 8,
        }
    }
}

/// SAIC file structure information
#[derive(Debug)]
pub struct SaicFileInfo {
    /// File size in bytes
    pub file_size: usize,
    
    /// Manifest size in bytes
    pub manifest_size: usize,
    
    /// Model payload size in bytes
    pub payload_size: usize,
    
    /// Total overhead (header + manifest)
    pub overhead: usize,
}

impl SaicFileInfo {
    /// Calculate file information from raw data
    pub fn from_file_data(file_size: usize, manifest_size: usize) -> Self {
        let payload_size = file_size - SaicFormat::TOTAL_HEADER_SIZE - manifest_size;
        let overhead = SaicFormat::TOTAL_HEADER_SIZE + manifest_size;
        
        Self {
            file_size,
            manifest_size,
            payload_size,
            overhead,
        }
    }
    
    /// Get overhead percentage
    pub fn overhead_percentage(&self) -> f64 {
        (self.overhead as f64 / self.file_size as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saic_format_constants() {
        assert_eq!(SaicFormat::HEADER_SIZE, 8);
        assert_eq!(SaicFormat::MANIFEST_LENGTH_SIZE, 4);
        assert_eq!(SaicFormat::TOTAL_HEADER_SIZE, 12);
    }

    #[test]
    fn test_saic_manifest_serialization() {
        let manifest = SaicManifest {
            model_name: "TestModel".to_string(),
            model_version: Some("1.0.0".to_string()),
            framework: Some("ONNX".to_string()),
            input_shape: Some(vec![1, 3, 224, 224]),
            output_shape: Some(vec![1, 1000]),
            optimization_hints: Some(OptimizationHints::default()),
        };
        
        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: SaicManifest = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.model_name, "TestModel");
        assert_eq!(deserialized.model_version, Some("1.0.0".to_string()));
        assert_eq!(deserialized.framework, Some("ONNX".to_string()));
    }

    #[test]
    fn test_saic_file_info() {
        let file_size = 1000000; // 1MB
        let manifest_size = 1000;
        
        let info = SaicFileInfo::from_file_data(file_size, manifest_size);
        
        assert_eq!(info.file_size, 1000000);
        assert_eq!(info.manifest_size, 1000);
        assert_eq!(info.payload_size, 1000000 - 12 - 1000);
        assert_eq!(info.overhead, 12 + 1000);
        
        // Overhead should be small percentage
        assert!(info.overhead_percentage() < 1.0);
    }
} 