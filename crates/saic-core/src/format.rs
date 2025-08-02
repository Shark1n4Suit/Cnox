//! SAIC file format specification

use serde::{Deserialize, Serialize};

/// SAIC file format specification
pub struct SaicFormat;

impl SaicFormat {
    /// Header size in bytes (magic + version)
    pub const HEADER_SIZE: usize = 8;
    
    /// Manifest length field size in bytes
    pub const MANIFEST_LENGTH_SIZE: usize = 4;
    
    /// Total header size including manifest length
    pub const TOTAL_HEADER_SIZE: usize = 12;
}

/// SAIC file manifest structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaicManifest {
    pub model_name: String,
    pub model_version: Option<String>,
    pub framework: Option<String>,
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
    pub optimization_hints: Option<OptimizationHints>,
}

/// Hardware optimization hints for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    pub simd_instructions: Vec<String>,
    pub recommended_threads: usize,
    pub memory_alignment: usize,
    pub cache_line_size: usize,
    pub batch_size: usize,
    pub security_level: u8,
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            simd_instructions: vec!["AVX2".to_string()],
            recommended_threads: 4,
            memory_alignment: 32,
            cache_line_size: 64,
            batch_size: 1,
            security_level: 1,
        }
    }
}

/// SAIC file structure information
#[derive(Debug)]
pub struct SaicFileInfo {
    pub file_size: usize,
    pub manifest_size: usize,
    pub payload_size: usize,
    pub overhead: usize,
}

impl SaicFileInfo {
    /// Create file info from file data
    pub fn from_file_data(file_size: usize, manifest_size: usize) -> Self {
        let payload_size = file_size.saturating_sub(SaicFormat::TOTAL_HEADER_SIZE + manifest_size);
        let overhead = SaicFormat::TOTAL_HEADER_SIZE + manifest_size;
        
        Self {
            file_size,
            manifest_size,
            payload_size,
            overhead,
        }
    }
    
    /// Calculate overhead percentage
    pub fn overhead_percentage(&self) -> f64 {
        if self.file_size == 0 {
            return 0.0;
        }
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
    fn test_saic_manifest() {
        let manifest = SaicManifest {
            model_name: "TestModel".to_string(),
            model_version: Some("1.0.0".to_string()),
            framework: Some("ONNX".to_string()),
            input_shape: Some(vec![1, 3, 224, 224]),
            output_shape: Some(vec![1, 1000]),
            optimization_hints: Some(OptimizationHints::default()),
        };
        
        assert_eq!(manifest.model_name, "TestModel");
        assert_eq!(manifest.model_version, Some("1.0.0".to_string()));
        assert_eq!(manifest.framework, Some("ONNX".to_string()));
    }

    #[test]
    fn test_optimization_hints_default() {
        let hints = OptimizationHints::default();
        
        assert!(hints.simd_instructions.contains(&"AVX2".to_string()));
        assert_eq!(hints.recommended_threads, 4);
        assert_eq!(hints.memory_alignment, 32);
        assert_eq!(hints.cache_line_size, 64);
        assert_eq!(hints.batch_size, 1);
        assert_eq!(hints.security_level, 1);
    }

    #[test]
    fn test_saic_file_info() {
        let file_info = SaicFileInfo::from_file_data(1000, 100);
        
        assert_eq!(file_info.file_size, 1000);
        assert_eq!(file_info.manifest_size, 100);
        assert_eq!(file_info.payload_size, 888); // 1000 - 12 - 100
        assert_eq!(file_info.overhead, 112); // 12 + 100
        
        let overhead_pct = file_info.overhead_percentage();
        assert!((overhead_pct - 11.2).abs() < 0.01);
    }
} 