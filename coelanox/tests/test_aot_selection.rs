//! Tests for AOT binary selection and backend metadata.
//!
//! These tests verify that:
//! - Packaging with AOT selection flags works correctly
//! - BackendMetadata is correctly stored in container manifest
//! - Device ID mapping is correct
//! - Backward compatibility is maintained

use coelanox_core::format::{
    BackendMetadata, BackendSelectionMode, CoelanoxContainer, CoelanoxManifest, CompressionType,
    ModelFormat,
};
use tempfile::NamedTempFile;

/// Helper function to create a minimal test container with backend metadata
fn create_test_container_with_backend_metadata(
    backend_metadata: Option<BackendMetadata>,
) -> CoelanoxContainer {
    let mut manifest = CoelanoxManifest::new(
        "test_model".to_string(),
        ModelFormat::Onnx,
        vec![vec![1, 3, 224, 224]],
        vec![vec![1, 1000]],
        1024, // machine_code_size
    );

    manifest.backend_metadata = backend_metadata;

    let machine_code = vec![0u8; 1024];
    let compressed_weights = vec![0u8; 512];
    let ir_data = vec![0u8; 256];

    CoelanoxContainer::new(
        manifest,
        machine_code,
        compressed_weights,
        CompressionType::None,
        ir_data,
    )
}

#[test]
fn test_package_with_slim_selection() {
    // Test packaging with --target h100 (slim mode with hardware-specific target)
    let backend_metadata = BackendMetadata::new_slim("h100".to_string(), Vec::new());

    let container = create_test_container_with_backend_metadata(Some(backend_metadata.clone()));

    // Verify BackendMetadata is created
    assert!(
        container.manifest.backend_metadata.is_some(),
        "BackendMetadata should be present"
    );

    let metadata = container.manifest.backend_metadata.as_ref().unwrap();

    // Verify primary backend is "h100"
    assert_eq!(
        metadata.primary_backend, "h100",
        "Primary backend should be h100"
    );

    // Verify device ID 0 maps to "h100"
    assert_eq!(
        metadata.get_backend_for_device(0),
        Some(&"h100".to_string()),
        "Device ID 0 should map to h100"
    );

    // Verify selection mode is Slim
    match &metadata.selection_mode {
        BackendSelectionMode::Slim {
            primary,
            additional,
        } => {
            assert_eq!(primary, "h100");
            assert_eq!(additional.len(), 0);
        }
        _ => panic!("Selection mode should be Slim"),
    }
}

#[test]
fn test_package_with_backend_add() {
    // Test packaging with --target h100 --backend-add amd,intel
    let backend_metadata = BackendMetadata::new_slim(
        "h100".to_string(),
        vec!["amd".to_string(), "intel".to_string()],
    );

    let container = create_test_container_with_backend_metadata(Some(backend_metadata.clone()));

    // Verify BackendMetadata has primary + additional backends
    let metadata = container.manifest.backend_metadata.as_ref().unwrap();

    assert_eq!(metadata.primary_backend, "h100");
    assert_eq!(metadata.additional_backends.len(), 2);
    assert_eq!(metadata.additional_backends[0], "amd");
    assert_eq!(metadata.additional_backends[1], "intel");

    // Verify device ID mapping: 0=h100, 1=amd, 2=intel
    assert_eq!(
        metadata.get_backend_for_device(0),
        Some(&"h100".to_string())
    );
    assert_eq!(metadata.get_backend_for_device(1), Some(&"amd".to_string()));
    assert_eq!(
        metadata.get_backend_for_device(2),
        Some(&"intel".to_string())
    );
}

#[test]
fn test_package_with_fat_flag() {
    // Test packaging with --fat
    let all_backends = vec!["cpu".to_string(), "gpu".to_string(), "npu".to_string()];
    let backend_metadata = BackendMetadata::new_fat(all_backends.clone());

    let container = create_test_container_with_backend_metadata(Some(backend_metadata.clone()));

    // Verify BackendMetadata selection_mode is Fat
    let metadata = container.manifest.backend_metadata.as_ref().unwrap();

    match metadata.selection_mode {
        BackendSelectionMode::Fat => {
            // Correct
        }
        _ => panic!("Selection mode should be Fat"),
    }

    // Verify all available backends included
    assert_eq!(metadata.device_id_mapping.len(), 3);
    assert_eq!(metadata.get_backend_for_device(0), Some(&"cpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(1), Some(&"gpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(2), Some(&"npu".to_string()));
}

#[test]
fn test_package_with_auto_flag() {
    // Test packaging with --auto
    let detected_backends = vec!["cpu".to_string(), "gpu".to_string()];
    let backend_metadata = BackendMetadata::new_auto(detected_backends.clone());

    let container = create_test_container_with_backend_metadata(Some(backend_metadata.clone()));

    // Verify BackendMetadata selection_mode is Auto
    let metadata = container.manifest.backend_metadata.as_ref().unwrap();

    match metadata.selection_mode {
        BackendSelectionMode::Auto => {
            // Correct
        }
        _ => panic!("Selection mode should be Auto"),
    }

    // Verify detected backends included
    assert_eq!(metadata.device_id_mapping.len(), 2);
    assert_eq!(metadata.get_backend_for_device(0), Some(&"cpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(1), Some(&"gpu".to_string()));
}

#[test]
fn test_backend_metadata_in_manifest() {
    // Package a model with AOT selection
    let backend_metadata = BackendMetadata::new_slim("h100".to_string(), vec!["amd".to_string()]);

    let container = create_test_container_with_backend_metadata(Some(backend_metadata.clone()));

    // Write container to file
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    // Read container back
    let read_container =
        CoelanoxContainer::read_binary_file(file_path).expect("Failed to read container");

    // Verify manifest.backend_metadata is Some
    assert!(
        read_container.manifest.backend_metadata.is_some(),
        "BackendMetadata should be present in manifest"
    );

    // Verify metadata contains correct information
    let metadata = read_container.manifest.backend_metadata.as_ref().unwrap();
    assert_eq!(metadata.primary_backend, "h100");
    assert_eq!(metadata.additional_backends.len(), 1);
    assert_eq!(metadata.additional_backends[0], "amd");
    assert_eq!(
        metadata.get_backend_for_device(0),
        Some(&"h100".to_string())
    );
    assert_eq!(metadata.get_backend_for_device(1), Some(&"amd".to_string()));
}

#[test]
fn test_backward_compatibility_no_flags() {
    // Package a model with just --target cpu (no AOT flags)
    // This should result in BackendMetadata being None (backward compatible)
    let container = create_test_container_with_backend_metadata(None);

    // Verify BackendMetadata is None (backward compatible)
    assert!(
        container.manifest.backend_metadata.is_none(),
        "BackendMetadata should be None for backward compatibility"
    );

    // Verify container still works (can be written and read)
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let file_path = temp_file.path();

    container
        .write_binary_file(file_path)
        .expect("Failed to write container");

    let read_container =
        CoelanoxContainer::read_binary_file(file_path).expect("Failed to read container");

    assert!(
        read_container.manifest.backend_metadata.is_none(),
        "BackendMetadata should still be None after read"
    );
}

#[test]
fn test_backend_metadata_serialization() {
    // Test that BackendMetadata can be serialized and deserialized
    let metadata = BackendMetadata::new_slim(
        "h100".to_string(),
        vec!["amd".to_string(), "intel".to_string()],
    );

    // Serialize to JSON
    let json = serde_json::to_string(&metadata).expect("Failed to serialize BackendMetadata");

    // Deserialize from JSON
    let deserialized: BackendMetadata =
        serde_json::from_str(&json).expect("Failed to deserialize BackendMetadata");

    // Verify data is preserved
    assert_eq!(deserialized.primary_backend, "h100");
    assert_eq!(deserialized.additional_backends.len(), 2);
    assert_eq!(deserialized.additional_backends[0], "amd");
    assert_eq!(deserialized.additional_backends[1], "intel");
    assert_eq!(
        deserialized.get_backend_for_device(0),
        Some(&"h100".to_string())
    );
    assert_eq!(
        deserialized.get_backend_for_device(1),
        Some(&"amd".to_string())
    );
    assert_eq!(
        deserialized.get_backend_for_device(2),
        Some(&"intel".to_string())
    );
}

#[test]
fn test_backend_metadata_device_id_mapping_consistency() {
    // Test that device ID mapping is consistent across different creation methods
    let slim_metadata = BackendMetadata::new_slim("h100".to_string(), vec!["amd".to_string()]);

    // Verify device IDs are sequential starting from 0
    assert_eq!(
        slim_metadata.get_backend_for_device(0),
        Some(&"h100".to_string())
    );
    assert_eq!(
        slim_metadata.get_backend_for_device(1),
        Some(&"amd".to_string())
    );
    assert_eq!(slim_metadata.get_backend_for_device(2), None);

    // Verify device_id_mapping matches get_backend_for_device
    assert_eq!(
        slim_metadata.device_id_mapping.get(&0),
        Some(&"h100".to_string())
    );
    assert_eq!(
        slim_metadata.device_id_mapping.get(&1),
        Some(&"amd".to_string())
    );
    assert_eq!(slim_metadata.device_id_mapping.get(&2), None);
}

#[test]
fn test_backend_metadata_fat_vs_slim() {
    // Test that Fat and Slim modes produce different metadata
    let slim_metadata = BackendMetadata::new_slim("cpu".to_string(), vec!["gpu".to_string()]);

    let fat_metadata = BackendMetadata::new_fat(vec!["cpu".to_string(), "gpu".to_string()]);

    // Both should have same backends, but different selection modes
    assert_eq!(slim_metadata.primary_backend, fat_metadata.primary_backend);
    assert_eq!(
        slim_metadata.additional_backends,
        fat_metadata.additional_backends
    );

    // But selection modes should differ
    match (&slim_metadata.selection_mode, &fat_metadata.selection_mode) {
        (BackendSelectionMode::Slim { .. }, BackendSelectionMode::Fat) => {
            // Correct
        }
        _ => panic!("Selection modes should differ"),
    }
}
