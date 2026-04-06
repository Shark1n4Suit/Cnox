//! Tests for security hash and container validation.
//!
//! These tests verify that:
//! - SHA-256 hash is calculated during packaging
//! - Hash verification works correctly
//! - Tampered containers are detected
//! - Corrupted containers are detected

use coelanox_core::format::{
    verify_security_hash, CoelanoxContainer, CoelanoxManifest, CompressionType, ModelFormat,
};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper function to create a minimal test container
fn create_test_container() -> CoelanoxResult<CoelanoxContainer> {
    let manifest = CoelanoxManifest::new(
        "test_model".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        1024, // machine_code_size
    );

    let machine_code = vec![0u8; 1024]; // Dummy machine code
    let compressed_weights = vec![0u8; 512]; // Dummy compressed weights
    let ir_data = vec![0u8; 256]; // Dummy IR data

    Ok(CoelanoxContainer::new(
        manifest,
        machine_code,
        compressed_weights,
        CompressionType::None,
        ir_data,
    ))
}

/// Helper function to tamper with a file (modify a byte)
fn tamper_with_file(path: &std::path::Path) -> CoelanoxResult<()> {
    let mut data = fs::read(path)?;
    // Modify a byte after the header (to avoid breaking magic number)
    if data.len() > 100 {
        data[100] = data[100].wrapping_add(1);
    }
    fs::write(path, data)?;
    Ok(())
}

#[test]
fn test_hash_calculation_during_packaging() {
    // Create a test container
    let container = create_test_container().expect("Failed to create test container");

    // Write container to temporary file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Read container back and check hash
    let read_container =
        CoelanoxContainer::read_binary_file(file_path).expect("Failed to read container");

    // Verify hash is not all zeros
    assert_ne!(
        read_container.header.security_hash, [0u8; 32],
        "Security hash should not be all zeros after packaging"
    );

    // Verify hash is 32 bytes
    assert_eq!(
        read_container.header.security_hash.len(),
        32,
        "Security hash should be 32 bytes"
    );

    // Verify hash matches calculated hash of file
    verify_security_hash(file_path).expect("Hash verification should succeed for valid container");
}

#[test]
fn test_verify_security_hash_valid_container() {
    // Create and write a valid container
    let container = create_test_container().expect("Failed to create test container");
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Verify hash
    let result = verify_security_hash(file_path);
    assert!(
        result.is_ok(),
        "Hash verification should succeed for valid container"
    );
}

#[test]
fn test_verify_security_hash_tampered_container() {
    // Create and write a valid container
    let container = create_test_container().expect("Failed to create test container");
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Tamper with the file
    tamper_with_file(file_path).expect("Failed to tamper with file");

    // Verify hash should fail
    let result = verify_security_hash(file_path);
    assert!(
        result.is_err(),
        "Hash verification should fail for tampered container"
    );

    // Verify it's a SecurityValidation error
    match result {
        Err(CoelanoxError::SecurityValidation(_)) => {
            // Expected error type
        }
        Err(e) => panic!("Expected SecurityValidation error, got: {:?}", e),
        Ok(_) => panic!("Expected error, but verification succeeded"),
    }
}

#[test]
fn test_verify_security_hash_corrupted_container() {
    // Create and write a valid container
    let container = create_test_container().expect("Failed to create test container");
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Corrupt the file by truncating it
    let mut file = fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(file_path)
        .expect("Failed to open file for corruption");

    // Write garbage data
    file.write_all(b"GARBAGE_DATA")
        .expect("Failed to write garbage");

    // Verify hash should fail
    let result = verify_security_hash(file_path);
    assert!(
        result.is_err(),
        "Hash verification should fail for corrupted container"
    );
}

#[test]
fn test_hash_verification_with_zero_hash_bytes() {
    // Create and write a valid container
    let container = create_test_container().expect("Failed to create test container");
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Read file, set hash bytes to zero, calculate hash
    let mut data = fs::read(file_path).expect("Failed to read file");
    let hash_offset = 36; // Hash starts at offset 36 in header

    // Set hash bytes to zero
    for i in 0..32 {
        data[hash_offset + i] = 0;
    }

    // Calculate hash of modified data
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let calculated_hash: [u8; 32] = hasher.finalize().into();

    // Read original container to get stored hash
    let original_container =
        CoelanoxContainer::read_binary_file(file_path).expect("Failed to read original container");

    // The stored hash should match the calculated hash of file with zero hash bytes
    assert_eq!(
        original_container.header.security_hash, calculated_hash,
        "Stored hash should match calculated hash of file with zero hash bytes"
    );
}

#[test]
fn test_hash_scope_entire_container() {
    // Create a container with known content
    let container = create_test_container().expect("Failed to create test container");
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Read entire file
    let file_data = fs::read(file_path).expect("Failed to read file");

    // Calculate hash manually (with hash bytes set to zero)
    let mut data_for_hashing = file_data.clone();
    let hash_offset = 36;
    let zero_hash = [0u8; 32];
    data_for_hashing[hash_offset..hash_offset + 32].copy_from_slice(&zero_hash);

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&data_for_hashing);
    let manual_hash: [u8; 32] = hasher.finalize().into();

    // Read container to get stored hash
    let read_container =
        CoelanoxContainer::read_binary_file(file_path).expect("Failed to read container");

    // Verify stored hash matches manual calculation
    assert_eq!(
        read_container.header.security_hash, manual_hash,
        "Stored hash should match manual calculation of entire container"
    );
}

#[test]
fn test_hash_verification_file_not_found() {
    // Try to verify non-existent file
    let non_existent = std::path::Path::new("/nonexistent/path/container.cnox");
    let result = verify_security_hash(non_existent);

    assert!(result.is_err(), "Should fail for non-existent file");
    match result {
        Err(CoelanoxError::FileSystem(_)) => {
            // Expected error type
        }
        Err(e) => panic!("Expected FileSystem error, got: {:?}", e),
        Ok(_) => panic!("Expected error, but verification succeeded"),
    }
}

#[test]
fn test_hash_verification_too_small_file() {
    // Create a file that's too small to contain a valid header
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    // Write only a few bytes (less than header size)
    fs::write(file_path, b"COEL").expect("Failed to write small file");

    let result = verify_security_hash(file_path);
    assert!(result.is_err(), "Should fail for file too small");
}
