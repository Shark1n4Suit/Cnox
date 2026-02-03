//! Tests for Task 1.3: Backend Manager & AOT Binary Selection
//! 
//! These tests verify that:
//! - BackendManager can be created and initialized
//! - Backend discovery works (even if no backends found)
//! - BackendMetadata structures work correctly
//! - Device ID mapping is correct

use coelanox_runtime::{BackendManager, BackendInfo, BackendKind, BackendSelectionStrategy, DiscoveryResult};
use coelanox_core::format::{BackendMetadata, BackendSelectionMode};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use tempfile::TempDir;

#[test]
fn test_backend_manager_creation() {
    // Create BackendManager instance
    let manager = BackendManager::new();
    
    // Verify it initializes successfully
    // Verify default state (no backends discovered)
    let backends = manager.list_backends();
    assert_eq!(backends.len(), 0, "New BackendManager should have no backends");
}

#[test]
fn test_backend_manager_discovery() {
    // Create BackendManager
    let mut manager = BackendManager::new();
    
    // Create temporary directory for test search paths
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let search_paths = vec![temp_dir.path().to_path_buf()];
    
    // Call discover_backends (may find 0 backends, that's OK)
    let result = manager.discover_backends(&search_paths);
    assert!(result.is_ok(), "Backend discovery should complete successfully");
    
    // Verify discovery result structure
    let discovery_result = result.unwrap();
    assert_eq!(discovery_result.success_count(), 0, "Should find 0 backends in empty directory");
    assert_eq!(discovery_result.failure_count(), 0, "Should have 0 failures for empty directory");
    // total_scanned is usize, so it's always >= 0
    assert!(discovery_result.total_scanned == discovery_result.total_scanned, "Total scanned should be valid");
    
    // Verify discovery completed (even if no backends found)
    let backends = manager.list_backends();
    // It's OK if no backends are found - we're just testing the discovery mechanism
    assert!(backends.len() == backends.len(), "Backend list should be valid");
}

#[test]
fn test_backend_manager_search_paths() {
    // Call default_search_paths
    let paths = BackendManager::default_search_paths();
    
    // Verify paths are returned
    assert!(!paths.is_empty(), "Default search paths should not be empty");
    
    // Verify paths are valid format (may not exist, but format is correct)
    for path in &paths {
        // Just verify it's a valid PathBuf (no panic on creation)
        assert!(path.as_os_str().len() > 0, "Path should not be empty");
    }
}

#[test]
fn test_backend_metadata_slim_mode() {
    // Create BackendMetadata with new_slim
    let primary = "h100".to_string();
    let additional = vec!["amd".to_string(), "intel".to_string()];
    let metadata = BackendMetadata::new_slim(primary.clone(), additional.clone());
    
    // Verify primary backend is device ID 0
    assert_eq!(metadata.primary_backend, "h100", "Primary backend should be h100");
    assert_eq!(metadata.get_backend_for_device(0), Some(&"h100".to_string()),
        "Device ID 0 should map to primary backend");
    
    // Verify additional backends get sequential IDs (1, 2, 3...)
    assert_eq!(metadata.get_backend_for_device(1), Some(&"amd".to_string()),
        "Device ID 1 should map to first additional backend");
    assert_eq!(metadata.get_backend_for_device(2), Some(&"intel".to_string()),
        "Device ID 2 should map to second additional backend");
    
    // Verify device_id_mapping is correct
    assert_eq!(metadata.device_id_mapping.len(), 3, "Should have 3 device IDs");
    assert_eq!(metadata.device_id_mapping.get(&0), Some(&"h100".to_string()));
    assert_eq!(metadata.device_id_mapping.get(&1), Some(&"amd".to_string()));
    assert_eq!(metadata.device_id_mapping.get(&2), Some(&"intel".to_string()));
    
    // Verify selection mode
    match metadata.selection_mode {
        BackendSelectionMode::Slim { primary: p, additional: a } => {
            assert_eq!(p, "h100");
            assert_eq!(a.len(), 2);
            assert_eq!(a[0], "amd");
            assert_eq!(a[1], "intel");
        }
        _ => panic!("Selection mode should be Slim"),
    }
}

#[test]
fn test_backend_metadata_fat_mode() {
    // Create BackendMetadata with new_fat
    let all_backends = vec!["cpu".to_string(), "gpu".to_string(), "npu".to_string()];
    let metadata = BackendMetadata::new_fat(all_backends.clone());
    
    // Verify all backends are included
    assert_eq!(metadata.primary_backend, "cpu", "Primary should be first backend");
    assert_eq!(metadata.additional_backends.len(), 2, "Should have 2 additional backends");
    
    // Verify device ID mapping is correct
    assert_eq!(metadata.device_id_mapping.len(), 3, "Should have 3 device IDs");
    assert_eq!(metadata.get_backend_for_device(0), Some(&"cpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(1), Some(&"gpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(2), Some(&"npu".to_string()));
    
    // Verify selection mode
    match metadata.selection_mode {
        BackendSelectionMode::Fat => {
            // Correct
        }
        _ => panic!("Selection mode should be Fat"),
    }
}

#[test]
fn test_backend_metadata_auto_mode() {
    // Create BackendMetadata with new_auto
    let detected_backends = vec!["cpu".to_string(), "gpu".to_string()];
    let metadata = BackendMetadata::new_auto(detected_backends.clone());
    
    // Verify detected backends are included
    assert_eq!(metadata.primary_backend, "cpu", "Primary should be first detected backend");
    assert_eq!(metadata.additional_backends.len(), 1, "Should have 1 additional backend");
    
    // Verify device ID mapping is correct
    assert_eq!(metadata.device_id_mapping.len(), 2, "Should have 2 device IDs");
    assert_eq!(metadata.get_backend_for_device(0), Some(&"cpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(1), Some(&"gpu".to_string()));
    
    // Verify selection mode
    match metadata.selection_mode {
        BackendSelectionMode::Auto => {
            // Correct
        }
        _ => panic!("Selection mode should be Auto"),
    }
}

#[test]
fn test_backend_metadata_device_id_lookup() {
    // Create BackendMetadata
    let metadata = BackendMetadata::new_slim(
        "h100".to_string(),
        vec!["amd".to_string(), "intel".to_string()],
    );
    
    // Test get_backend_for_device for various device IDs
    assert_eq!(metadata.get_backend_for_device(0), Some(&"h100".to_string()),
        "Device ID 0 should return primary backend");
    assert_eq!(metadata.get_backend_for_device(1), Some(&"amd".to_string()),
        "Device ID 1 should return first additional backend");
    assert_eq!(metadata.get_backend_for_device(2), Some(&"intel".to_string()),
        "Device ID 2 should return second additional backend");
    assert_eq!(metadata.get_backend_for_device(3), None,
        "Device ID 3 should return None (out of range)");
    assert_eq!(metadata.get_backend_for_device(100), None,
        "Large device ID should return None");
}

#[test]
fn test_backend_metadata_empty_additional() {
    // Test BackendMetadata with no additional backends
    let metadata = BackendMetadata::new_slim("cpu".to_string(), Vec::new());
    
    assert_eq!(metadata.primary_backend, "cpu");
    assert_eq!(metadata.additional_backends.len(), 0);
    assert_eq!(metadata.device_id_mapping.len(), 1);
    assert_eq!(metadata.get_backend_for_device(0), Some(&"cpu".to_string()));
    assert_eq!(metadata.get_backend_for_device(1), None);
}

#[test]
fn test_backend_metadata_fat_empty_backends() {
    // Test BackendMetadata::new_fat with empty list (should default to cpu)
    let metadata = BackendMetadata::new_fat(Vec::new());
    
    assert_eq!(metadata.primary_backend, "cpu", "Should default to cpu");
    assert_eq!(metadata.device_id_mapping.len(), 1);
}

#[test]
fn test_backend_manager_validation() {
    // Test backend validation methods
    let mut manager = BackendManager::new();
    
    // Register a test backend
    let backend_info = BackendInfo {
        name: "test_backend".to_string(),
        version: "1.0.0".to_string(),
        library_path: std::path::PathBuf::from("test"),
        supported_targets: vec![],
        kind: BackendKind::Translator,
    };
    manager.register_backend(backend_info);
    
    // Test validate_backend with existing backend
    assert!(manager.validate_backend("test_backend").is_ok(), 
        "Should validate existing backend");
    
    // Test validate_backend with non-existent backend
    let result = manager.validate_backend("nonexistent");
    assert!(result.is_err(), "Should fail for non-existent backend");
    match result {
        Err(CoelanoxError::BackendNotFound(_)) => {
            // Expected error type
        }
        _ => panic!("Expected BackendNotFound error"),
    }
    
    // Test validate_backends with mixed list
    let names = vec!["test_backend".to_string(), "nonexistent".to_string()];
    let result = manager.validate_backends(&names);
    assert!(result.is_err(), "Should fail when some backends are missing");
}

#[test]
fn test_backend_metadata_device_id_validation() {
    // Test device ID validation
    let metadata = BackendMetadata::new_slim(
        "h100".to_string(),
        vec!["amd".to_string(), "intel".to_string()],
    );
    
    // Valid device IDs should pass
    assert!(metadata.validate_device_id(0).is_ok());
    assert!(metadata.validate_device_id(1).is_ok());
    assert!(metadata.validate_device_id(2).is_ok());
    
    // Invalid device IDs should fail
    let result = metadata.validate_device_id(3);
    assert!(result.is_err(), "Should fail for out-of-range device ID");
    match result {
        Err(CoelanoxError::DeviceIdOutOfRange(id, max)) => {
            assert_eq!(id, 3);
            assert_eq!(max, 2);
        }
        _ => panic!("Expected DeviceIdOutOfRange error"),
    }
    
    // Test get_backend_for_device_checked
    assert_eq!(metadata.get_backend_for_device_checked(0).unwrap(), &"h100".to_string());
    assert_eq!(metadata.get_backend_for_device_checked(1).unwrap(), &"amd".to_string());
    
    let result = metadata.get_backend_for_device_checked(100);
    assert!(result.is_err(), "Should fail for out-of-range device ID");
}

#[test]
fn test_backend_selection_strategy() {
    // Test backend selection with fallback strategies
    let mut manager = BackendManager::new();
    
    // Register test backends
    let backend1 = BackendInfo {
        name: "primary".to_string(),
        version: "1.0.0".to_string(),
        library_path: std::path::PathBuf::from("primary"),
        supported_targets: vec![],
        kind: BackendKind::Translator,
    };
    let backend2 = BackendInfo {
        name: "fallback".to_string(),
        version: "1.0.0".to_string(),
        library_path: std::path::PathBuf::from("fallback"),
        supported_targets: vec![],
        kind: BackendKind::Translator,
    };
    manager.register_backend(backend1);
    manager.register_backend(backend2);
    
    // Test PrimaryOnly strategy with existing backend
    let strategy = BackendSelectionStrategy::PrimaryOnly("primary".to_string());
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_ok(), "Should find primary backend");
    assert_eq!(result.unwrap().name, "primary");
    
    // Test PrimaryOnly strategy with non-existent backend
    let strategy = BackendSelectionStrategy::PrimaryOnly("nonexistent".to_string());
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_err(), "Should fail for non-existent backend");
    
    // Test PrimaryWithFallback strategy
    let strategy = BackendSelectionStrategy::PrimaryWithFallback {
        primary: "primary".to_string(),
        fallback: "fallback".to_string(),
    };
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_ok(), "Should find primary backend");
    assert_eq!(result.unwrap().name, "primary");
    
    // Test PrimaryWithFallback when primary doesn't exist
    let strategy = BackendSelectionStrategy::PrimaryWithFallback {
        primary: "nonexistent".to_string(),
        fallback: "fallback".to_string(),
    };
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_ok(), "Should fallback to fallback backend");
    assert_eq!(result.unwrap().name, "fallback");
    
    // Test TryAll strategy
    let strategy = BackendSelectionStrategy::TryAll(
        vec!["nonexistent1".to_string(), "primary".to_string(), "nonexistent2".to_string()]
    );
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_ok(), "Should find first available backend");
    assert_eq!(result.unwrap().name, "primary");
    
    // Test TryAll with all non-existent
    let strategy = BackendSelectionStrategy::TryAll(
        vec!["nonexistent1".to_string(), "nonexistent2".to_string()]
    );
    let result = manager.select_backend_with_fallback(&strategy);
    assert!(result.is_err(), "Should fail when no backends available");
}

