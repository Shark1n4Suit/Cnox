//! Simple tests for SAIC weight extraction and compression
//! 
//! This module tests the core weight functionality without complex dependencies.

use saic::{SaicPackager, SaicRuntime, format::{HardwareTarget, CompressionType}};
use std::path::Path;
use tempfile::TempDir;

#[cfg(test)]
mod simple_weight_tests {
    use super::*;

    #[test]
    fn test_weight_compression_basic() {
        println!("🧪 Testing basic weight compression...");
        
        // Create test weights
        let test_weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        // Create packager
        let packager = SaicPackager::new().expect("Failed to create packager");
        
        // Compress weights
        let (compressed_data, compression_type) = packager.compress_weights(&test_weights)
            .expect("Failed to compress weights");
        
        println!("✅ Compression type: {:?}", compression_type);
        println!("✅ Original size: {} bytes", test_weights.len() * 4);
        println!("✅ Compressed size: {} bytes", compressed_data.len());
        
        // Verify compression worked
        assert!(!compressed_data.is_empty(), "Compression failed");
        println!("✅ Basic compression test passed");
    }

    #[test]
    fn test_weight_decompression_basic() {
        println!("🧪 Testing basic weight decompression...");
        
        // Create test weights
        let test_weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        // Create packager
        let packager = SaicPackager::new().expect("Failed to create packager");
        
        // Compress weights
        let (compressed_data, compression_type) = packager.compress_weights(&test_weights)
            .expect("Failed to compress weights");
        
        // Create a simple container for testing
        let mut manifest = saic::format::SaicManifest::default();
        manifest.name = "test_model".to_string();
        manifest.input_shapes = vec![vec![5]];
        manifest.output_shapes = vec![vec![3]];
        manifest.operations = vec![]; // Empty operations for now
        
        let container = saic::format::SaicContainer::new(
            manifest,
            vec![0x90, 0x90, 0x90, 0x90], // Simple machine code
            compressed_data,
            compression_type,
        );
        
        // Test decompression
        let decompressed = container.decompress_weights()
            .expect("Failed to decompress weights");
        
        assert_eq!(decompressed.len(), test_weights.len(), "Decompressed size mismatch");
        
        // Verify values match (within floating point precision)
        for (i, (&original, &decompressed)) in test_weights.iter().zip(decompressed.iter()).enumerate() {
            assert!((original - decompressed).abs() < 0.001, 
                "Weight {} mismatch: {} vs {}", i, original, decompressed);
        }
        
        println!("✅ Basic decompression test passed");
    }

    #[test]
    fn test_saic_container_creation() {
        println!("🧪 Testing SAIC container creation...");
        
        // Create manifest
        let mut manifest = saic::format::SaicManifest::default();
        manifest.name = "test_model".to_string();
        manifest.input_shapes = vec![vec![5]];
        manifest.output_shapes = vec![vec![3]];
        manifest.operations = vec![];
        
        // Create test weights
        let test_weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_bytes: Vec<u8> = test_weights.iter()
            .flat_map(|&w| w.to_le_bytes())
            .collect();
        
        // Create container
        let container = saic::format::SaicContainer::new(
            manifest,
            vec![0x90, 0x90, 0x90, 0x90], // Simple machine code
            weight_bytes,
            CompressionType::None,
        );
        
        // Verify container structure
        assert_eq!(container.compressed_weights.len(), test_weights.len() * 4);
        assert_eq!(container.compression_type, CompressionType::None);
        assert_eq!(container.manifest.name, "test_model");
        
        println!("✅ Container creation test passed");
    }

    #[test]
    fn test_compression_types() {
        println!("🧪 Testing compression types...");
        
        let compression_types = vec![
            CompressionType::None,
            CompressionType::Zstd,
            CompressionType::Lz4,
            CompressionType::Deflate,
        ];
        
        for comp_type in compression_types {
            let test_weights = vec![1.0f32, 2.0, 3.0];
            let weight_bytes: Vec<u8> = test_weights.iter()
                .flat_map(|&w| w.to_le_bytes())
                .collect();
            
            // Create container with this compression type
            let mut manifest = saic::format::SaicManifest::default();
            manifest.name = "test_model".to_string();
            manifest.input_shapes = vec![vec![3]];
            manifest.output_shapes = vec![vec![1]];
            manifest.operations = vec![];
            
            let container = saic::format::SaicContainer::new(
                manifest,
                vec![0x90, 0x90, 0x90, 0x90],
                weight_bytes,
                comp_type.clone(),
            );
            
            // Test decompression
            let decompressed = container.decompress_weights()
                .expect("Failed to decompress weights");
            
            assert_eq!(decompressed.len(), test_weights.len());
            println!("✅ Compression type {:?} works", comp_type);
        }
        
        println!("✅ All compression types test passed");
    }
}
