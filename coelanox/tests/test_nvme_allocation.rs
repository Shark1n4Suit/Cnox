//! Tests for NVMe Pre-Allocation
//!
//! These tests verify the NVMe pre-allocation functionality, including:
//! - NVMe allocator size calculation
//! - NVMe backing data creation
//! - Container integration with NVMe backing
//! - Offset calculation and storage
//! - Security hash includes NVMe backing

use coelanox_core::format::{
    verify_security_hash, CoelanoxContainer, CoelanoxManifest, CompressionType, ModelFormat,
};
use coelanox_core::ir::{DataType, Shape, UniversalIR};
use coelanox_core::optimization::{MaxShapeMetadata, MaxShapeReader};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_packager::memory::NvmeAllocator;
use tempfile::TempDir;

// Helper to create a dummy IR with weight and intermediate tensors
fn create_dummy_ir_with_shapes() -> UniversalIR {
    let mut ir = UniversalIR::new("test_nvme_model".to_string());
    ir.metadata
        .add_input_shape(Shape::new(vec![1, 3, 224, 224]));
    ir.metadata.add_output_shape(Shape::new(vec![1, 1000]));

    // Add weight tensor
    ir.add_weight_tensor_with_data(
        "conv_weights".to_string(),
        Shape::new(vec![64, 3, 3, 3]),
        DataType::Float32,
        vec![0; 64 * 3 * 3 * 3 * 4], // 4 bytes per float32
    );

    // Add intermediate tensor
    ir.add_intermediate_tensor(
        "layer_output".to_string(),
        Shape::new(vec![1, 64, 224, 224]),
        DataType::Float32,
    );

    ir
}

#[test]
fn test_nvme_allocator_calculate_size() {
    let ir = create_dummy_ir_with_shapes();
    let max_shapes = MaxShapeReader::read_max_shapes(&ir).unwrap();

    let size = NvmeAllocator::calculate_nvme_size(&max_shapes);

    // NVMe backing is workspace only (intermediates + input + output), not weights
    // Weight: 64 * 3 * 3 * 3 * 4 = 6912 (not included in backing)
    // Intermediate: 1 * 64 * 224 * 224 * 4 = 12,845,056
    // Input: 1 * 3 * 224 * 224 * 4 = 602,112
    // Output: 1 * 1000 * 4 = 4,000
    // Total workspace: 12,845,056 + 602,112 + 4,000 = 13,451,168
    let expected_intermediate = 64 * 224 * 224 * 4;
    let expected_input = 3 * 224 * 224 * 4;
    let expected_output = 1000 * 4;
    let expected_workspace = expected_intermediate + expected_input + expected_output;

    assert_eq!(size, expected_workspace);
}

#[test]
fn test_nvme_allocator_create_backing() {
    let ir = create_dummy_ir_with_shapes();
    let max_shapes = MaxShapeReader::read_max_shapes(&ir).unwrap();

    let backing = NvmeAllocator::create_nvme_backing(&max_shapes).unwrap();

    // Verify size matches calculation
    let expected_size = NvmeAllocator::calculate_nvme_size(&max_shapes);
    assert_eq!(backing.len(), expected_size);

    // Verify it's zero-initialized
    assert!(backing.iter().all(|&b| b == 0));
}

#[test]
fn test_nvme_allocator_zero_size_error() {
    let metadata = MaxShapeMetadata::new();

    let result = NvmeAllocator::create_nvme_backing(&metadata);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        CoelanoxError::InvalidParameter(_)
    ));
}

#[test]
fn test_nvme_allocator_size_for_types() {
    let mut metadata = MaxShapeMetadata::new();
    metadata.total_weight_bytes = 1000;
    metadata.total_intermediate_bytes = 500;

    // Test weights only
    assert_eq!(
        NvmeAllocator::calculate_size_for_types(&metadata, true, false),
        1000
    );

    // Test intermediates only
    assert_eq!(
        NvmeAllocator::calculate_size_for_types(&metadata, false, true),
        500
    );

    // Test both
    assert_eq!(
        NvmeAllocator::calculate_size_for_types(&metadata, true, true),
        1500
    );

    // Test neither
    assert_eq!(
        NvmeAllocator::calculate_size_for_types(&metadata, false, false),
        0
    );
}

#[test]
fn test_container_with_nvme_backing() -> CoelanoxResult<()> {
    let temp_dir = TempDir::new()?;
    let container_path = temp_dir.path().join("test_nvme.cnox");

    // Create dummy IR
    let ir = create_dummy_ir_with_shapes();
    let max_shapes = MaxShapeReader::read_max_shapes(&ir)?;
    let nvme_backing = NvmeAllocator::create_nvme_backing(&max_shapes)?;
    let nvme_backing_len = nvme_backing.len();

    // Create manifest
    let mut manifest = CoelanoxManifest::new(
        "test_nvme".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        1024, // machine_code_size
    );

    // Calculate offset (simplified - actual calculation is in packager)
    let manifest_size = serde_json::to_vec(&manifest).unwrap_or_default().len();
    let code_offset = coelanox_core::format::COELANOX_HEADER_SIZE as u64 + manifest_size as u64;
    let weights_offset = code_offset + 1024; // machine code size
    let ir_offset = weights_offset + 4; // no weights
    let nvme_backing_offset = ir_offset + 4 + 256; // dummy IR size

    manifest.nvme_backing_offset = Some(nvme_backing_offset);
    manifest.nvme_backing_size = Some(nvme_backing_len);

    // Create container with NVMe backing
    let container = CoelanoxContainer::new_with_nvme(
        manifest,
        vec![0u8; 1024], // machine code
        vec![],          // no compressed weights
        CompressionType::None,
        vec![0u8; 256], // IR data
        Some(nvme_backing),
    );

    // Write container
    container.write_binary_file(&container_path)?;

    // Verify container was written
    assert!(container_path.exists());

    // Read container back
    let loaded_container = CoelanoxContainer::read_binary_file(&container_path)?;

    // Verify NVMe backing data is present
    assert!(loaded_container.nvme_backing_data.is_some());
    assert_eq!(
        loaded_container.nvme_backing_data.as_ref().unwrap().len(),
        nvme_backing_len
    );

    // Verify manifest has NVMe fields
    assert!(loaded_container.manifest.nvme_backing_offset.is_some());
    assert!(loaded_container.manifest.nvme_backing_size.is_some());
    assert_eq!(
        loaded_container.manifest.nvme_backing_offset.unwrap(),
        nvme_backing_offset
    );
    assert_eq!(
        loaded_container.manifest.nvme_backing_size.unwrap(),
        nvme_backing_len
    );

    Ok(())
}

#[test]
fn test_container_without_nvme_backing() -> CoelanoxResult<()> {
    let temp_dir = TempDir::new()?;
    let container_path = temp_dir.path().join("test_no_nvme.cnox");

    // Create container without NVMe backing
    let manifest = CoelanoxManifest::new(
        "test_no_nvme".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        1024,
    );

    let container = CoelanoxContainer::new(
        manifest,
        vec![0u8; 1024],
        vec![],
        CompressionType::None,
        vec![0u8; 256],
    );

    // Write container
    container.write_binary_file(&container_path)?;

    // Read container back
    let loaded_container = CoelanoxContainer::read_binary_file(&container_path)?;

    // Verify NVMe backing data is None
    assert!(loaded_container.nvme_backing_data.is_none());
    assert!(loaded_container.manifest.nvme_backing_offset.is_none());
    assert!(loaded_container.manifest.nvme_backing_size.is_none());

    Ok(())
}

#[test]
fn test_security_hash_includes_nvme_backing() -> CoelanoxResult<()> {
    let temp_dir = TempDir::new()?;
    let container_path = temp_dir.path().join("test_hash_nvme.cnox");

    // Create container with NVMe backing
    let ir = create_dummy_ir_with_shapes();
    let max_shapes = MaxShapeReader::read_max_shapes(&ir)?;
    let nvme_backing = NvmeAllocator::create_nvme_backing(&max_shapes)?;
    let nvme_backing_len = nvme_backing.len();

    let mut manifest = CoelanoxManifest::new(
        "test_hash_nvme".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        1024,
    );

    let manifest_size = serde_json::to_vec(&manifest).unwrap_or_default().len();
    let code_offset = coelanox_core::format::COELANOX_HEADER_SIZE as u64 + manifest_size as u64;
    let nvme_backing_offset = (code_offset + 1024 + 4) + 4 + 256;

    manifest.nvme_backing_offset = Some(nvme_backing_offset);
    manifest.nvme_backing_size = Some(nvme_backing_len);

    let container = CoelanoxContainer::new_with_nvme(
        manifest,
        vec![0u8; 1024],
        vec![],
        CompressionType::None,
        vec![0u8; 256],
        Some(nvme_backing),
    );

    // Write container
    container.write_binary_file(&container_path)?;

    // Verify security hash (should include NVMe backing)
    verify_security_hash(&container_path)?;

    // Tamper with NVMe backing data in file
    let mut file_data = std::fs::read(&container_path)?;
    if let Some(offset) = container.manifest.nvme_backing_offset {
        let nvme_start = offset as usize + 8; // Skip length prefix
        if nvme_start < file_data.len() {
            file_data[nvme_start] ^= 0xFF; // Flip a bit
            std::fs::write(&container_path, &file_data)?;

            // Verify hash fails
            let result = verify_security_hash(&container_path);
            assert!(result.is_err());
        }
    }

    Ok(())
}

#[test]
fn test_nvme_backing_size_matches_calculation() -> CoelanoxResult<()> {
    let ir = create_dummy_ir_with_shapes();
    let max_shapes = MaxShapeReader::read_max_shapes(&ir)?;

    let calculated_size = NvmeAllocator::calculate_nvme_size(&max_shapes);
    let backing = NvmeAllocator::create_nvme_backing(&max_shapes)?;

    assert_eq!(backing.len(), calculated_size);

    Ok(())
}
