//! Integration tests for container signing: sign → verify → run
//!
//! Verifies that:
//! - Package with --signing-key produces a signed container
//! - verify with --trusted-key succeeds for signed containers
//! - run with --trusted-key loads and executes signed containers
//! - Wrong trusted key fails signature verification
//! - Tampered signed container fails hash verification

use coelanox_core::format::{
    is_container_signed, verify_security_hash_with_trusted_key, CoelanoxContainer,
    CoelanoxManifest, CompressionType, ModelFormat,
};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_orchestrator::{CoelanoxRuntime, RuntimeConfig};
use std::fs;
use tempfile::NamedTempFile;

/// Create a minimal test container (unsigned)
fn create_test_container() -> CoelanoxResult<CoelanoxContainer> {
    let manifest = CoelanoxManifest::new(
        "test_model".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        0, // fallback-only: no machine code (model_size=0)
    );
    let machine_code = vec![];
    let compressed_weights = vec![0u8; 64];
    let ir_data = vec![0u8; 128];
    Ok(CoelanoxContainer::new(
        manifest,
        machine_code,
        compressed_weights,
        CompressionType::None,
        ir_data,
    ))
}

#[test]
fn test_sign_verify_run_flow() {
    // 1. Generate keypair
    let (secret, public) = coelanox_core::format::generate_signing_keypair();

    // 2. Create container and write with signing
    let container = create_test_container().expect("create container");
    let temp_file = NamedTempFile::new().expect("temp file");
    let path = temp_file.path();
    container
        .write_binary_file_with_signing(path, Some(&secret))
        .expect("write signed");

    // 3. Verify container is signed (flag set)
    assert!(
        is_container_signed(path).expect("is_container_signed"),
        "Container should be signed"
    );

    // 4. Verify hash + signature with trusted key
    verify_security_hash_with_trusted_key(path, Some(&public)).expect("verify with trusted key");

    // 5. Load with trusted key (runtime verifies hash + signature before load)
    let config = RuntimeConfig::default();
    let runtime = CoelanoxRuntime::with_config(config);
    runtime
        .set_trusted_public_key(public)
        .expect("set trusted key");
    runtime
        .load_container(path.to_str().unwrap())
        .expect("load container");
    // Load succeeds; execute may fail with minimal IR - we've verified the sign→verify→load flow
}

#[test]
fn test_signed_container_wrong_key_fails() {
    let (secret, _public) = coelanox_core::format::generate_signing_keypair();
    let (_, wrong_public) = coelanox_core::format::generate_signing_keypair();

    let container = create_test_container().expect("create container");
    let temp_file = NamedTempFile::new().expect("temp file");
    let path = temp_file.path();
    container
        .write_binary_file_with_signing(path, Some(&secret))
        .expect("write signed");

    let result = verify_security_hash_with_trusted_key(path, Some(&wrong_public));
    assert!(
        result.is_err(),
        "Wrong key should fail signature verification"
    );
    match result {
        Err(CoelanoxError::SecurityValidation(msg)) if msg.contains("Signature") => {}
        Err(e) => panic!("Expected SecurityValidation (signature), got: {:?}", e),
        Ok(_) => panic!("Expected failure"),
    }
}

#[test]
fn test_signed_container_tampered_fails_hash() {
    let (secret, public) = coelanox_core::format::generate_signing_keypair();

    let container = create_test_container().expect("create container");
    let temp_file = NamedTempFile::new().expect("temp file");
    let path = temp_file.path();
    container
        .write_binary_file_with_signing(path, Some(&secret))
        .expect("write signed");

    // Tamper: modify a byte after header
    let mut data = fs::read(path).expect("read");
    if data.len() > 100 {
        data[100] = data[100].wrapping_add(1);
        fs::write(path, data).expect("write tampered");
    }

    let result = verify_security_hash_with_trusted_key(path, Some(&public));
    assert!(
        result.is_err(),
        "Tampered container should fail hash verification"
    );
}

#[test]
fn test_unsigned_container_no_signature_check() {
    let container = create_test_container().expect("create container");
    let temp_file = NamedTempFile::new().expect("temp file");
    let path = temp_file.path();
    container.write_binary_file(path).expect("write unsigned");

    assert!(
        !is_container_signed(path).expect("is_container_signed"),
        "Container should not be signed"
    );

    // Verify hash only (no trusted key) should succeed
    verify_security_hash_with_trusted_key(path, None).expect("verify hash only");

    // Verify with some key should still pass hash; signature check is skipped for unsigned
    let (_, public) = coelanox_core::format::generate_signing_keypair();
    verify_security_hash_with_trusted_key(path, Some(&public))
        .expect("verify with key on unsigned");
}
