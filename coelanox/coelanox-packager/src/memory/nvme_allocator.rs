//! NVMe Pre-Allocation Allocator for COELANOX
//! 
//! This module provides functionality to calculate NVMe backing size and create
//! pre-allocated memory layouts for container files.
//! 
//! **Purpose:**
//! - Calculate total memory size needed from max shapes
//! - Create pre-allocated memory layout (zero-initialized for now)
//! - Enable unified file approach (NVMe backing appended to container)

use coelanox_core::optimization::MaxShapeMetadata;
use coelanox_core::{CoelanoxError, CoelanoxResult};

/// NVMe Allocator for pre-allocation
/// 
/// Calculates NVMe backing size and creates pre-allocated memory layouts
/// based on maximum tensor shapes extracted from the IR.
pub struct NvmeAllocator;

impl NvmeAllocator {
    /// Calculate total NVMe size needed from max shapes
    ///
    /// Uses workspace only (intermediates + input/output). Weights are stored in the
    /// container weight section and loaded by the runtime separately; including them
    /// here would duplicate ~500MB+ for large models.
    pub fn calculate_nvme_size(metadata: &MaxShapeMetadata) -> usize {
        metadata.total_workspace_bytes()
    }
    
    /// Create NVMe backing data from max shapes
    /// 
    /// Creates a pre-allocated memory layout (zero-initialized) based on
    /// maximum tensor shapes. This backing data will be appended to the
    /// container file for unified file approach.
    /// 
    /// Creates pre-allocated memory layout from max shapes.
    /// Size = workspace only (intermediates + input/output), not weights.
    /// 
    /// **Future Enhancement:**
    /// - Structured layout with tensor offsets
    /// - Alignment handling
    /// - Memory layout optimization
    /// 
    /// # Arguments
    /// * `metadata` - Maximum shape metadata from MaxShapeReader
    /// 
    /// # Returns
    /// * `Ok(Vec<u8>)` - Pre-allocated backing data (zero-initialized)
    /// * `Err(CoelanoxError)` - If calculation fails
    pub fn create_nvme_backing(metadata: &MaxShapeMetadata) -> CoelanoxResult<Vec<u8>> {
        let total_size = Self::calculate_nvme_size(metadata);
        
        // Validate size is reasonable (not zero, not too large)
        if total_size == 0 {
            return Err(CoelanoxError::InvalidParameter(
                "NVMe backing size cannot be zero".to_string()
            ));
        }
        
        // Check for unreasonably large sizes (sanity check)
        // 1TB limit (1_000_000_000_000 bytes)
        const MAX_NVME_SIZE: usize = 1_000_000_000_000;
        if total_size > MAX_NVME_SIZE {
            return Err(CoelanoxError::InvalidParameter(
                format!("NVMe backing size too large: {} bytes (max: {} bytes)", total_size, MAX_NVME_SIZE)
            ));
        }
        
        // Create zero-initialized backing data
        // This represents pre-allocated memory that will be used at runtime
        // The actual data will be written by the runtime when tensors are loaded
        let backing_data = vec![0u8; total_size];
        
        Ok(backing_data)
    }
    
    /// Get NVMe backing size for a specific tensor type
    /// 
    /// # Arguments
    /// * `metadata` - Maximum shape metadata
    /// * `include_weights` - Include weight tensor sizes
    /// * `include_intermediates` - Include intermediate tensor sizes
    /// 
    /// # Returns
    /// * Total size in bytes for the specified tensor types
    pub fn calculate_size_for_types(
        metadata: &MaxShapeMetadata,
        include_weights: bool,
        include_intermediates: bool,
    ) -> usize {
        let mut total = 0;
        
        if include_weights {
            total += metadata.total_weight_bytes;
        }
        
        if include_intermediates {
            total += metadata.total_intermediate_bytes;
        }
        
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coelanox_core::optimization::{MaxShape, MaxShapeMetadata};
    
    #[test]
    fn test_calculate_nvme_size() {
        let mut metadata = MaxShapeMetadata::new();

        // Weights are not included in NVMe size (stored in container weight section)
        metadata.weight_shapes.push(MaxShape::new(
            "weight_0".to_string(),
            vec![1000, 1000],
            4, // Float32
        ));
        metadata.total_weight_bytes = 1000 * 1000 * 4;

        // Intermediates (workspace) are included
        metadata.intermediate_shapes.push(MaxShape::new(
            "intermediate_0".to_string(),
            vec![1, 1000],
            4, // Float32
        ));
        metadata.total_intermediate_bytes = 1 * 1000 * 4;

        let size = NvmeAllocator::calculate_nvme_size(&metadata);
        // Only workspace (intermediates + input/output), not weights
        assert_eq!(size, 1 * 1000 * 4);
    }
    
    #[test]
    fn test_create_nvme_backing() {
        let mut metadata = MaxShapeMetadata::new();

        metadata.intermediate_shapes.push(MaxShape::new(
            "intermediate_0".to_string(),
            vec![100, 100],
            4, // Float32
        ));
        metadata.total_intermediate_bytes = 100 * 100 * 4;

        let backing = NvmeAllocator::create_nvme_backing(&metadata).unwrap();

        assert_eq!(backing.len(), 100 * 100 * 4);
        // Verify it's zero-initialized
        assert!(backing.iter().all(|&b| b == 0));
    }
    
    #[test]
    fn test_create_nvme_backing_zero_size() {
        let metadata = MaxShapeMetadata::new();
        
        let result = NvmeAllocator::create_nvme_backing(&metadata);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_calculate_size_for_types() {
        let mut metadata = MaxShapeMetadata::new();
        
        metadata.total_weight_bytes = 1000;
        metadata.total_intermediate_bytes = 500;
        
        // Test weights only
        assert_eq!(NvmeAllocator::calculate_size_for_types(&metadata, true, false), 1000);
        
        // Test intermediates only
        assert_eq!(NvmeAllocator::calculate_size_for_types(&metadata, false, true), 500);
        
        // Test both
        assert_eq!(NvmeAllocator::calculate_size_for_types(&metadata, true, true), 1500);
        
        // Test neither
        assert_eq!(NvmeAllocator::calculate_size_for_types(&metadata, false, false), 0);
    }
}


