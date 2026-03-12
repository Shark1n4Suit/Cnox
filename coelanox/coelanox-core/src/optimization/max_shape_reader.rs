//! Max Shape Reader for COELANOX
//! 
//! This module extracts maximum tensor shapes from the Universal IR for pre-allocation.
//! Shapes are already in the IR (populated by the translator), so this module just
//! reads and extracts them.
//! 
//! **Key Features:**
//! - Reads shapes from IR (translator already resolved dynamic dimensions)
//! - Uses IR shapes as-is for pre-allocation
//! - Optional metadata overrides for explicit max bounds
//! - Generates max shape metadata for container pre-allocation
//! 
//! **Philosophy:**
//! - Trust the IR: Translator already handled dynamic shapes (ONNX -1, min/max bounds, etc.)
//! - Simple extraction: Just read what's in the IR
//! - Metadata is optional: Only for explicit user overrides

use crate::ir::{UniversalIR, Shape};
use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Maximum shape metadata for a tensor
/// 
/// Contains the maximum dimensions for a tensor, which can be used for
/// pre-allocation at package time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxShape {
    /// Tensor identifier (name or ID)
    pub tensor_id: String,
    /// Maximum dimensions (for pre-allocation)
    /// 
    /// These are the dimensions from the IR (translator already resolved dynamic shapes).
    /// Used for pre-allocation to reserve maximum possible memory.
    pub max_dimensions: Vec<usize>,
    /// Data type size in bytes (for memory calculation)
    pub data_type_size: usize,
}

impl MaxShape {
    /// Create a new max shape
    pub fn new(tensor_id: String, max_dimensions: Vec<usize>, data_type_size: usize) -> Self {
        Self {
            tensor_id,
            max_dimensions,
            data_type_size,
        }
    }
    
    /// Calculate total memory size in bytes for this max shape
    pub fn total_bytes(&self) -> usize {
        self.max_dimensions.iter().product::<usize>() * self.data_type_size
    }
    
    /// Get total number of elements
    pub fn total_elements(&self) -> usize {
        self.max_dimensions.iter().product()
    }
}

/// Max shape metadata for a container
/// 
/// Contains all maximum shapes extracted from the IR, organized by tensor type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxShapeMetadata {
    /// Maximum shapes for weight tensors
    pub weight_shapes: Vec<MaxShape>,
    /// Maximum shapes for intermediate tensors
    pub intermediate_shapes: Vec<MaxShape>,
    /// Maximum input shapes
    pub input_shapes: Vec<MaxShape>,
    /// Maximum output shapes
    pub output_shapes: Vec<MaxShape>,
    /// Total memory size in bytes for all weights
    pub total_weight_bytes: usize,
    /// Total memory size in bytes for all intermediates
    pub total_intermediate_bytes: usize,
}

impl MaxShapeMetadata {
    /// Create new max shape metadata
    pub fn new() -> Self {
        Self {
            weight_shapes: Vec::new(),
            intermediate_shapes: Vec::new(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            total_weight_bytes: 0,
            total_intermediate_bytes: 0,
        }
    }
    
    /// Calculate total memory size for all tensors (weights + intermediates).
    /// Use this when you need the full footprint for reporting.
    pub fn total_bytes(&self) -> usize {
        self.total_weight_bytes + self.total_intermediate_bytes
    }

    /// Total bytes for workspace only (intermediates + input + output).
    /// Use for NVMe backing: weights are stored in the container weight section and
    /// loaded separately at runtime; backing should not reserve space for weights again.
    pub fn total_workspace_bytes(&self) -> usize {
        let input_bytes: usize = self.input_shapes.iter().map(|s| s.total_bytes()).sum();
        let output_bytes: usize = self.output_shapes.iter().map(|s| s.total_bytes()).sum();
        self.total_intermediate_bytes + input_bytes + output_bytes
    }
}

impl Default for MaxShapeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Max Shape Reader
/// 
/// Extracts maximum tensor shapes from the Universal IR for pre-allocation.
/// Shapes are already in the IR (populated by translator), so this just reads them.
pub struct MaxShapeReader;

impl MaxShapeReader {
    /// Read maximum shapes from Universal IR
    /// 
    /// This method extracts all shapes from the IR and generates max shape metadata.
    /// The translator already resolved dynamic dimensions, so we just use the IR shapes as-is.
    /// Optional metadata overrides can provide explicit max bounds.
    /// 
    /// # Arguments
    /// * `ir` - The Universal IR containing all tensor shapes
    /// 
    /// # Returns
    /// * `Ok(MaxShapeMetadata)` with all extracted max shapes
    /// * `Err(CoelanoxError)` if extraction fails
    pub fn read_max_shapes(ir: &UniversalIR) -> CoelanoxResult<MaxShapeMetadata> {
        let mut metadata = MaxShapeMetadata::new();
        
        // Extract weight tensor shapes
        for (index, weight_tensor) in ir.weight_tensors.iter().enumerate() {
            let max_shape = Self::extract_tensor_shape(
                &format!("weight_{}", index),
                &weight_tensor.shape,
                weight_tensor.data_type.size(),
                &ir.metadata.metadata,
            )?;
            metadata.total_weight_bytes += max_shape.total_bytes();
            metadata.weight_shapes.push(max_shape);
        }
        
        // Extract intermediate tensor shapes
        for (index, intermediate_tensor) in ir.intermediate_tensors.iter().enumerate() {
            let max_shape = Self::extract_tensor_shape(
                &format!("intermediate_{}", index),
                &intermediate_tensor.shape,
                intermediate_tensor.data_type.size(),
                &ir.metadata.metadata,
            )?;
            metadata.total_intermediate_bytes += max_shape.total_bytes();
            metadata.intermediate_shapes.push(max_shape);
        }
        
        // Extract input shapes from model metadata
        for (index, input_shape) in ir.metadata.input_shapes.iter().enumerate() {
            let max_shape = Self::extract_tensor_shape(
                &format!("input_{}", index),
                input_shape,
                ir.metadata.data_type.size(),
                &ir.metadata.metadata,
            )?;
            metadata.input_shapes.push(max_shape);
        }
        
        // Extract output shapes from model metadata
        for (index, output_shape) in ir.metadata.output_shapes.iter().enumerate() {
            let max_shape = Self::extract_tensor_shape(
                &format!("output_{}", index),
                output_shape,
                ir.metadata.data_type.size(),
                &ir.metadata.metadata,
            )?;
            metadata.output_shapes.push(max_shape);
        }
        
        Ok(metadata)
    }
    
    /// Extract max shape for a single tensor
    /// 
    /// Uses the shape from IR as-is (translator already resolved dynamic dimensions).
    /// Optionally checks metadata for explicit max shape overrides.
    /// 
    /// # Arguments
    /// * `tensor_id` - Identifier for the tensor (name or ID)
    /// * `shape` - The shape from the IR (already resolved by translator)
    /// * `data_type_size` - Size of the data type in bytes
    /// * `metadata` - Model metadata (optional: for explicit max shape overrides)
    /// 
    /// # Returns
    /// * `Ok(MaxShape)` with the maximum dimensions
    /// * `Err(CoelanoxError)` if extraction fails
    fn extract_tensor_shape(
        tensor_id: &str,
        shape: &Shape,
        data_type_size: usize,
        metadata: &HashMap<String, String>,
    ) -> CoelanoxResult<MaxShape> {
        // Start with IR shape (translator already resolved dynamic dimensions)
        let mut max_dimensions = shape.dimensions.clone();
        
        // Check for explicit max shape override in metadata (optional)
        // Format: "tensor_id_max_shape" = "dim1,dim2,dim3"
        if let Some(max_shape_str) = metadata.get(&format!("{}_max_shape", tensor_id)) {
            // Parse max shape from metadata
            let parsed_dims: Result<Vec<usize>, _> = max_shape_str
                .split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect();
            
            if let Ok(dims) = parsed_dims {
                if dims.len() == max_dimensions.len() {
                    max_dimensions = dims;
                } else {
                    return Err(CoelanoxError::InvalidParameter(
                        format!("Max shape override for {} has wrong number of dimensions: expected {}, got {}", 
                            tensor_id, max_dimensions.len(), dims.len())
                    ));
                }
            } else {
                return Err(CoelanoxError::InvalidParameter(
                    format!("Invalid max shape format for {}: {}", tensor_id, max_shape_str)
                ));
            }
        }
        
        // Use IR shape as max shape (translator already resolved it)
        Ok(MaxShape::new(
            tensor_id.to_string(),
            max_dimensions,
            data_type_size,
        ))
    }
    
    /// Get max shape for a specific tensor by ID
    /// 
    /// # Arguments
    /// * `metadata` - The max shape metadata
    /// * `tensor_id` - The tensor ID to look up
    /// 
    /// # Returns
    /// * `Some(&MaxShape)` if found
    /// * `None` if not found
    pub fn get_max_shape<'a>(metadata: &'a MaxShapeMetadata, tensor_id: &str) -> Option<&'a MaxShape> {
        // Check weight tensors
        if let Some(shape) = metadata.weight_shapes.iter().find(|s| s.tensor_id == tensor_id) {
            return Some(shape);
        }
        
        // Check intermediate tensors
        if let Some(shape) = metadata.intermediate_shapes.iter().find(|s| s.tensor_id == tensor_id) {
            return Some(shape);
        }
        
        // Check input shapes
        if let Some(shape) = metadata.input_shapes.iter().find(|s| s.tensor_id == tensor_id) {
            return Some(shape);
        }
        
        // Check output shapes
        if let Some(shape) = metadata.output_shapes.iter().find(|s| s.tensor_id == tensor_id) {
            return Some(shape);
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_max_shape_creation() {
        let max_shape = MaxShape::new(
            "test_tensor".to_string(),
            vec![1, 3, 224, 224],
            4, // Float32
        );
        
        assert_eq!(max_shape.tensor_id, "test_tensor");
        assert_eq!(max_shape.max_dimensions, vec![1, 3, 224, 224]);
        assert_eq!(max_shape.total_elements(), 1 * 3 * 224 * 224);
        assert_eq!(max_shape.total_bytes(), 1 * 3 * 224 * 224 * 4);
    }
    
    #[test]
    fn test_max_shape_metadata_creation() {
        let metadata = MaxShapeMetadata::new();
        
        assert!(metadata.weight_shapes.is_empty());
        assert!(metadata.intermediate_shapes.is_empty());
        assert_eq!(metadata.total_weight_bytes, 0);
        assert_eq!(metadata.total_intermediate_bytes, 0);
    }
    
    #[test]
    fn test_extract_shape_from_ir() {
        let shape = Shape::new(vec![1, 3, 224, 224]);
        let metadata = HashMap::new();
        
        let max_shape = MaxShapeReader::extract_tensor_shape(
            "test_tensor",
            &shape,
            4, // Float32
            &metadata,
        ).unwrap();
        
        // Should use IR shape as-is
        assert_eq!(max_shape.max_dimensions, vec![1, 3, 224, 224]);
    }
    
    #[test]
    fn test_extract_shape_with_metadata_override() {
        let shape = Shape::new(vec![1, 3, 224, 224]);
        let mut metadata = HashMap::new();
        metadata.insert("test_tensor_max_shape".to_string(), "1,3,512,512".to_string());
        
        let max_shape = MaxShapeReader::extract_tensor_shape(
            "test_tensor",
            &shape,
            4, // Float32
            &metadata,
        ).unwrap();
        
        // Should use metadata override
        assert_eq!(max_shape.max_dimensions, vec![1, 3, 512, 512]);
    }
    
    #[test]
    fn test_extract_shape_with_invalid_metadata_override() {
        let shape = Shape::new(vec![1, 3, 224, 224]);
        let mut metadata = HashMap::new();
        metadata.insert("test_tensor_max_shape".to_string(), "1,3,512".to_string()); // Wrong number of dims
        
        let result = MaxShapeReader::extract_tensor_shape(
            "test_tensor",
            &shape,
            4, // Float32
            &metadata,
        );
        
        // Should return error for invalid override
        assert!(result.is_err());
    }
    
    #[test]
    fn test_read_max_shapes_from_ir() {
        use crate::ir::{UniversalIR, DataType};
        
        // Create a simple IR with some tensors
        let mut ir = UniversalIR::new("test_model".to_string());
        
        // Add input shape
        ir.metadata.add_input_shape(Shape::new(vec![1, 3, 224, 224]));
        
        // Add output shape
        ir.metadata.add_output_shape(Shape::new(vec![1, 1000]));
        
        // Add a weight tensor
        let _weight_id = ir.add_weight_tensor(
            "weight1".to_string(),
            Shape::new(vec![1000, 1000]),
            DataType::Float32,
        );
        
        // Add an intermediate tensor
        let _intermediate_id = ir.add_intermediate_tensor(
            "intermediate1".to_string(),
            Shape::new(vec![1, 1000]),
            DataType::Float32,
        );
        
        // Read max shapes
        let max_shapes = MaxShapeReader::read_max_shapes(&ir).unwrap();
        
        // Verify we got shapes for all tensors
        assert_eq!(max_shapes.input_shapes.len(), 1);
        assert_eq!(max_shapes.output_shapes.len(), 1);
        assert_eq!(max_shapes.weight_shapes.len(), 1);
        assert_eq!(max_shapes.intermediate_shapes.len(), 1);
        
        // Verify input shape
        assert_eq!(max_shapes.input_shapes[0].max_dimensions, vec![1, 3, 224, 224]);
        
        // Verify output shape
        assert_eq!(max_shapes.output_shapes[0].max_dimensions, vec![1, 1000]);
        
        // Verify weight shape
        assert_eq!(max_shapes.weight_shapes[0].max_dimensions, vec![1000, 1000]);
        
        // Verify intermediate shape
        assert_eq!(max_shapes.intermediate_shapes[0].max_dimensions, vec![1, 1000]);
        
        // Verify total bytes calculation
        assert!(max_shapes.total_weight_bytes > 0);
        assert!(max_shapes.total_intermediate_bytes > 0);
    }
}

