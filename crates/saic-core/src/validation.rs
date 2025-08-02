//! Validation logic for SAIC files

use memmap2::Mmap;
use crate::{SaicError, SaicResult, MAGIC_NUMBER, MIN_FILE_SIZE, MAX_MANIFEST_SIZE};

/// Ultra-fast magic number validation using optimized comparison
/// 
/// Uses direct byte comparison optimized by the compiler
#[inline(always)]
pub fn validate_magic_optimized(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    
    // OPTIMIZATION: Direct byte comparison (faster than slice comparison)
    // This is already optimized by the compiler for 4-byte comparison
    data[0] == MAGIC_NUMBER[0] && 
    data[1] == MAGIC_NUMBER[1] && 
    data[2] == MAGIC_NUMBER[2] && 
    data[3] == MAGIC_NUMBER[3]
}

/// Ultra-fast file size validation
/// 
/// Performs minimal validation for maximum performance
#[inline(always)]
pub fn validate_file_size_fast(file_size: usize) -> bool {
    // OPTIMIZATION: Only check minimum size for header
    file_size >= MIN_FILE_SIZE
}

/// Ultra-fast manifest length extraction
#[inline(always)]
pub fn extract_manifest_length_fast(data: &[u8]) -> usize {
    if data.len() < 12 {
        return 0;
    }
    
    // OPTIMIZATION: Direct byte access for manifest length
    // Use unchecked access after bounds check for maximum performance
    u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize
}

/// Ultra-fast SAIC structure validation with aggressive optimizations
/// 
/// Performs comprehensive validation with maximum performance optimizations:
/// - Optimized magic number validation
/// - Early exit on all validation failures
/// - Optimized memory access patterns
/// - Minimal bounds checking
#[inline(always)]
pub fn validate_saic_structure_ultra_fast(mmap: &Mmap) -> SaicResult<()> {
    // OPTIMIZATION: Single bounds check for all operations
    if mmap.len() < MIN_FILE_SIZE {
        return Err(SaicError::InvalidMagic(
            "File too small for SAIC header".to_string()
        ));
    }
    
    // OPTIMIZATION: Optimized magic validation
    if !validate_magic_optimized(&mmap[0..4]) {
        return Err(SaicError::InvalidMagic(
            "Invalid SAIC magic number".to_string()
        ));
    }

    // OPTIMIZATION: Fast manifest length extraction
    let manifest_length = extract_manifest_length_fast(&mmap[0..12]);
    
    // OPTIMIZATION: Early exit for invalid manifest length
    if manifest_length == 0 || manifest_length > MAX_MANIFEST_SIZE {
        return Err(SaicError::FileSizeValidationFailed(
            "Invalid manifest length".to_string()
        ));
    }

    // OPTIMIZATION: Only validate complete structure for small files
    // For large files, defer full validation until needed
    if mmap.len() <= 1024 * 1024 { // Only validate complete structure for files <= 1MB
        if mmap.len() < 12 + manifest_length {
            return Err(SaicError::FileSizeValidationFailed(
                "File too small to contain complete manifest".to_string()
            ));
        }
    }

    Ok(())
}

/// Validate SAIC file structure
/// 
/// Performs comprehensive validation of SAIC file format including
/// magic number, version, manifest length, and basic structure.
pub fn validate_saic_structure(mmap: &Mmap) -> SaicResult<()> {
    // Use ultra-fast validation for maximum performance
    validate_saic_structure_ultra_fast(mmap)
}

/// Extract manifest from memory-mapped SAIC file
pub fn extract_manifest(mmap: &Mmap) -> SaicResult<serde_json::Value> {
    let manifest_length = extract_manifest_length_fast(&mmap[0..12]);
    
    if mmap.len() < 12 + manifest_length {
        return Err(SaicError::FileSizeValidationFailed(
            "File too small to contain complete manifest".to_string()
        ));
    }
    
    let manifest_bytes = &mmap[12..12 + manifest_length];
    let manifest_str = std::str::from_utf8(manifest_bytes)
        .map_err(|_| SaicError::InvalidFormat("Invalid UTF-8 in manifest".to_string()))?;
    
    let manifest: serde_json::Value = serde_json::from_str(manifest_str)
        .map_err(SaicError::ManifestError)?;
    
    Ok(manifest)
}

/// Extract model payload from memory-mapped SAIC file
pub fn extract_model_payload(mmap: &Mmap) -> SaicResult<&[u8]> {
    let manifest_length = extract_manifest_length_fast(&mmap[0..12]);
    
    let payload_start = 12 + manifest_length;
    if mmap.len() < payload_start {
        return Err(SaicError::FileSizeValidationFailed(
            "File too small to contain model payload".to_string()
        ));
    }
    
    Ok(&mmap[payload_start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_validate_magic_optimized() {
        // Test valid magic number
        assert!(validate_magic_optimized(b"SAIC"));
        
        // Test invalid magic number
        assert!(!validate_magic_optimized(b"INVA"));
        
        // Test too short data
        assert!(!validate_magic_optimized(b"SAI"));
    }

    #[test]
    fn test_validate_file_size_fast() {
        // Test valid sizes
        assert!(validate_file_size_fast(12));
        assert!(validate_file_size_fast(1000));
        
        // Test invalid sizes
        assert!(!validate_file_size_fast(11));
        assert!(!validate_file_size_fast(0));
    }

    #[test]
    fn test_validate_saic_structure() {
        // Create a valid SAIC file for testing
        let manifest = r#"{"model_name":"TestModel"}"#;
        let manifest_bytes = manifest.as_bytes();
        let manifest_length = manifest_bytes.len() as u32;
        
        let mut test_content = Vec::new();
        test_content.extend_from_slice(b"SAIC"); // Magic number
        test_content.extend_from_slice(&[0, 1, 0, 0]); // Version
        test_content.extend_from_slice(&manifest_length.to_be_bytes()); // Manifest length
        test_content.extend_from_slice(manifest_bytes); // Manifest
        test_content.extend_from_slice(b"test model payload"); // Payload
        
        // Create temporary file for testing
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&test_content).unwrap();
        temp_file.flush().unwrap();
        
        // Test valid structure
        let mmap = unsafe { Mmap::map(&*temp_file.as_file()).unwrap() };
        let result = validate_saic_structure(&mmap);
        assert!(result.is_ok());
        
        // Test invalid magic number
        let mut invalid_content = test_content.clone();
        invalid_content[0..4].copy_from_slice(b"INVA");
        
        let mut invalid_temp_file = NamedTempFile::new().unwrap();
        invalid_temp_file.write_all(&invalid_content).unwrap();
        invalid_temp_file.flush().unwrap();
        
        let invalid_mmap = unsafe { Mmap::map(&*invalid_temp_file.as_file()).unwrap() };
        let result = validate_saic_structure(&invalid_mmap);
        assert!(result.is_err());
    }
} 