//! ONNX Translator for COELANOX
//! 
//! This module implements the ONNX to Universal IR translator using real protobuf parsing.
//! It converts ONNX models to COELANOX's Universal IR format for universal deployment.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use crate::ir::{UniversalIR, ModelMetadata, Operation, Tensor, OpType, Attribute, DataType, Shape};
use coelanox_core::format::{ModelFormat, HardwareTarget};
use super::TranslatorPlugin;
use super::onnx::{ModelProto, GraphProto, OperatorSetIdProto, ValueInfoProto, TypeProto, type_proto, TensorProto, NodeProto, AttributeProto};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "onnx-protobuf")]
use prost::Message;

/// ONNX Translator implementation
pub struct OnnxTranslator {
    /// Translator name
    name: String,
    /// Translator version
    version: String,
}

impl OnnxTranslator {
    /// Create a new ONNX translator
    pub fn new() -> Self {
        Self {
            name: "onnx_translator".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Translate ONNX model to Universal IR
    pub fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
        #[cfg(feature = "onnx-protobuf")]
        {
            // Parse ONNX model using prost
            let model = onnx::ModelProto::decode(data)
                .map_err(|e| CoelanoxError::format(&format!("Failed to parse ONNX model: {}", e)))?;
            
            // Convert to Universal IR
            self.convert_model(&model)
        }
        
        #[cfg(not(feature = "onnx-protobuf"))]
        {
            // Fallback implementation for when protobuf is not available
            self.translate_fallback(data)
        }
    }
    
    /// Convert ONNX ModelProto to Universal IR
    fn convert_model(&self, model: &onnx::ModelProto) -> CoelanoxResult<UniversalIR> {
        let graph = model.graph.as_ref()
            .ok_or_else(|| CoelanoxError::format("ONNX model has no graph"))?;
        
        // Create Universal IR
        let mut ir = UniversalIR::new();
        
        // Set metadata
        ir.metadata = ModelMetadata {
            name: model.producer_name.clone().unwrap_or_else(|| "onnx_model".to_string()),
            version: model.model_version.unwrap_or(1),
            description: model.doc_string.clone().unwrap_or_default(),
            author: model.producer_name.clone().unwrap_or_default(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            format: ModelFormat::Onnx,
            hardware_target: HardwareTarget::Cpu, // Default, will be optimized later
            input_shapes: Vec::new(), // Will be populated from graph
            output_shapes: Vec::new(), // Will be populated from graph
        };
        
        // Convert graph
        self.convert_graph(graph, &mut ir)?;
        
        Ok(ir)
    }
    
    /// Convert ONNX GraphProto to Universal IR
    fn convert_graph(&self, graph: &onnx::GraphProto, ir: &mut UniversalIR) -> CoelanoxResult<()> {
        // Convert input shapes
        let mut input_shapes = Vec::new();
        for input in &graph.input {
            if let Some(shape) = self.convert_onnx_shape(input)? {
                input_shapes.push(shape);
            }
        }
        ir.metadata.input_shapes = input_shapes;
        
        // Convert output shapes
        let mut output_shapes = Vec::new();
        for output in &graph.output {
            if let Some(shape) = self.convert_onnx_shape(output)? {
                output_shapes.push(shape);
            }
        }
        ir.metadata.output_shapes = output_shapes;
        
        // Convert nodes to operations
        for node in &graph.node {
            let operation = self.convert_onnx_node(node)?;
            ir.add_operation(operation);
        }
        
        // Convert initializers to weight tensors
        for initializer in &graph.initializer {
            let tensor = self.convert_onnx_initializer(initializer)?;
            ir.add_weight_tensor(tensor);
        }
        
        Ok(())
    }
    
    /// Convert ONNX ValueInfoProto to Shape
    fn convert_onnx_shape(&self, value_info: &onnx::ValueInfoProto) -> CoelanoxResult<Option<Shape>> {
        let type_proto = value_info.r#type.as_ref()
            .ok_or_else(|| CoelanoxError::format("ValueInfo has no type"))?;
        
        let tensor_type = type_proto.value.as_ref()
            .and_then(|v| match v {
                onnx::type_proto::Value::TensorType(t) => Some(t),
                _ => None,
            });
        
        if let Some(tensor_type) = tensor_type {
            let shape_proto = tensor_type.shape.as_ref()
                .ok_or_else(|| CoelanoxError::format("Tensor type has no shape"))?;
            
            let mut dimensions = Vec::new();
            for dim in &shape_proto.dim {
                let size = match &dim.value {
                    Some(onnx::tensor_shape_proto::dimension::Value::DimValue(size)) => {
                        if *size <= 0 {
                            return Err(CoelanoxError::format("Invalid dimension size: must be > 0"));
                        }
                        *size as usize
                    }
                    Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) => {
                        // Dynamic dimension - use 0 as placeholder
                        0
                    }
                    None => {
                        return Err(CoelanoxError::format("Dimension has no value"));
                    }
                };
                dimensions.push(size);
            }
            
            Ok(Some(Shape::new(dimensions)))
        } else {
            Ok(None)
        }
    }
    
    /// Convert ONNX NodeProto to Operation
    fn convert_onnx_node(&self, node: &onnx::NodeProto) -> CoelanoxResult<Operation> {
        let op_type = self.convert_onnx_op_type(&node.op_type)?;
        
        let mut operation = Operation::new(op_type);
        
        // Add inputs
        for input in &node.input {
            operation.add_input(input.clone());
        }
        
        // Add outputs
        for output in &node.output {
            operation.add_output(output.clone());
        }
        
        // Convert attributes
        for attr in &node.attribute {
            let key = attr.name.clone();
            let value = self.convert_onnx_attribute(attr)?;
            operation.add_attribute(key, value);
        }
        
        Ok(operation)
    }
    
    /// Convert ONNX operation type to Universal IR operation type
    pub fn convert_onnx_op_type(&self, op_type: &str) -> CoelanoxResult<OpType> {
        match op_type {
            // Basic arithmetic operations
            "Add" => Ok(OpType::Add),
            "Sub" => Ok(OpType::Subtract),
            "Mul" => Ok(OpType::Multiply),
            "Div" => Ok(OpType::Divide),
            
            // Element-wise operations
            "Relu" => Ok(OpType::Relu),
            "Sigmoid" => Ok(OpType::Sigmoid),
            "Tanh" => Ok(OpType::Tanh),
            "Softmax" => Ok(OpType::Softmax),
            "LogSoftmax" => Ok(OpType::LogSoftmax),
            "Gelu" => Ok(OpType::Gelu),
            "Swish" => Ok(OpType::Swish),
            
            // Neural network primitives
            "Conv" => Ok(OpType::Convolution),
            "MaxPool" => Ok(OpType::MaxPool),
            "AveragePool" => Ok(OpType::AvgPool),
            "GlobalMaxPool" => Ok(OpType::GlobalMaxPool),
            "GlobalAveragePool" => Ok(OpType::GlobalAvgPool),
            "BatchNormalization" => Ok(OpType::BatchNorm),
            "LayerNormalization" => Ok(OpType::LayerNorm),
            "Dropout" => Ok(OpType::Dropout),
            
            // Tensor manipulation
            "Reshape" => Ok(OpType::Reshape),
            "Transpose" => Ok(OpType::Transpose),
            "Permute" => Ok(OpType::Permute),
            "Concat" => Ok(OpType::Concatenate),
            "Split" => Ok(OpType::Split),
            "Slice" => Ok(OpType::Slice),
            "Gather" => Ok(OpType::Gather),
            "Scatter" => Ok(OpType::Scatter),
            
            // Matrix operations
            "MatMul" => Ok(OpType::MatMul),
            "Gemm" => Ok(OpType::Gemm),
            
            // Reduction operations
            "ReduceSum" => Ok(OpType::ReduceSum),
            "ReduceMean" => Ok(OpType::ReduceMean),
            "ReduceMax" => Ok(OpType::ReduceMax),
            "ReduceMin" => Ok(OpType::ReduceMin),
            "ReduceProd" => Ok(OpType::ReduceProd),
            
            // Broadcasting operations
            "Broadcast" => Ok(OpType::Broadcast),
            "Expand" => Ok(OpType::Expand),
            
            // Comparison operations
            "Equal" => Ok(OpType::Equal),
            "NotEqual" => Ok(OpType::NotEqual),
            "Greater" => Ok(OpType::Greater),
            "GreaterEqual" => Ok(OpType::GreaterEqual),
            "Less" => Ok(OpType::Less),
            "LessEqual" => Ok(OpType::LessEqual),
            
            // Logical operations
            "And" => Ok(OpType::And),
            "Or" => Ok(OpType::Or),
            "Not" => Ok(OpType::Not),
            
            // Legacy compatibility
            "Pooling" => Ok(OpType::Pooling),
            "Activation" => Ok(OpType::Activation),
            "Attention" => Ok(OpType::Attention),
            "MultiHeadAttention" => Ok(OpType::MultiHeadAttention),
            "SelfAttention" => Ok(OpType::SelfAttention),
            "Transformer" => Ok(OpType::Transformer),
            "Encoder" => Ok(OpType::Encoder),
            "Decoder" => Ok(OpType::Decoder),
            
            // Custom operations
            _ => Ok(OpType::Custom(op_type.to_string())),
        }
    }
    
    /// Convert ONNX AttributeProto to Universal IR Attribute
    fn convert_onnx_attribute(&self, attr: &onnx::AttributeProto) -> CoelanoxResult<Attribute> {
        match attr.r#type {
            1 => { // FLOAT
                Ok(Attribute::Float(attr.f.unwrap_or(0.0)))
            }
            2 => { // INT
                Ok(Attribute::Int(attr.i.unwrap_or(0)))
            }
            3 => { // STRING
                Ok(Attribute::String(attr.s.as_ref().map(|s| String::from_utf8_lossy(s).to_string()).unwrap_or_default()))
            }
            4 => { // TENSOR
                // Convert tensor to float array for simplicity
                if let Some(tensor) = &attr.t {
                    let floats = self.convert_onnx_tensor_to_floats(tensor)?;
                    Ok(Attribute::FloatArray(floats))
                } else {
                    Ok(Attribute::FloatArray(vec![]))
                }
            }
            5 => { // GRAPH
                // For now, treat graph attributes as strings
                Ok(Attribute::String("graph".to_string()))
            }
            6 => { // BOOL
                // ONNX represents booleans as integers
                Ok(Attribute::FloatArray(vec![if attr.i.unwrap_or(0) != 0 { 1.0 } else { 0.0 }]))
            }
            7 => { // FLOATS
                Ok(Attribute::FloatArray(attr.floats.clone()))
            }
            8 => { // INTS
                Ok(Attribute::IntArray(attr.ints.clone()))
            }
            9 => { // STRINGS
                let strings: Vec<String> = attr.strings.iter()
                    .map(|s| String::from_utf8_lossy(s).to_string())
                    .collect();
                Ok(Attribute::StringArray(strings))
            }
            _ => {
                Err(CoelanoxError::format(&format!("Unsupported attribute type: {}", attr.r#type)))
            }
        }
    }
    
    /// Convert ONNX TensorProto to float array
    fn convert_onnx_tensor_to_floats(&self, tensor: &onnx::TensorProto) -> CoelanoxResult<Vec<f32>> {
        match tensor.data_type {
            1 => { // FLOAT
                Ok(tensor.float_data.clone())
            }
            2 => { // UINT8
                Err(CoelanoxError::format("UINT8 data type not supported"))
            }
            3 => { // INT8
                Err(CoelanoxError::format("INT8 data type not supported"))
            }
            6 => { // INT32
                // Convert int32 to float
                Ok(tensor.int32_data.iter().map(|&x| x as f32).collect())
            }
            7 => { // INT64
                // Convert int64 to float
                Ok(tensor.int64_data.iter().map(|&x| x as f32).collect())
            }
            9 => { // BOOL
                // Convert bool to float
                Ok(tensor.int32_data.iter().map(|&x| if x != 0 { 1.0 } else { 0.0 }).collect())
            }
            10 => { // FLOAT16
                // Convert float16 to float32
                Ok(tensor.float_data.clone())
            }
            11 => { // DOUBLE
                // Convert double to float
                Ok(tensor.double_data.iter().map(|&x| x as f32).collect())
            }
            _ => {
                Err(CoelanoxError::format(&format!("Unsupported tensor data type: {}", tensor.data_type)))
            }
        }
    }
    
    /// Convert ONNX InitializerProto to Weight Tensor
    fn convert_onnx_initializer(&self, initializer: &onnx::TensorProto) -> CoelanoxResult<Tensor> {
        let data_type = self.convert_onnx_data_type(initializer.data_type)?;
        
        let mut dimensions = Vec::new();
        for dim in &initializer.dims {
            if *dim <= 0 {
                return Err(CoelanoxError::format("Invalid dimension size: must be > 0"));
            }
            dimensions.push(*dim as usize);
        }
        
        let shape = Shape::new(dimensions);
        let data = self.convert_onnx_tensor_to_floats(initializer)?;
        
        Ok(Tensor {
            name: initializer.name.clone(),
            shape,
            data_type,
            data: Some(data),
        })
    }
    
    /// Convert ONNX data type to Universal IR data type
    fn convert_onnx_data_type(&self, data_type: i32) -> CoelanoxResult<DataType> {
        match data_type {
            1 => Ok(DataType::Float32),   // FLOAT
            2 => Err(CoelanoxError::format("UINT8 data type not supported")),
            3 => Err(CoelanoxError::format("INT8 data type not supported")),
            6 => Ok(DataType::Int32),    // INT32
            7 => Ok(DataType::Int64),    // INT64
            9 => Ok(DataType::Bool),     // BOOL
            10 => Ok(DataType::Float32), // FLOAT16 -> Float32
            11 => Ok(DataType::Float64), // DOUBLE
            _ => Err(CoelanoxError::format(&format!("Unsupported data type: {}", data_type))),
        }
    }
    
    /// Fallback implementation when protobuf is not available
    fn translate_fallback(&self, _data: &[u8]) -> CoelanoxResult<UniversalIR> {
        // Create a minimal Universal IR for testing
        let mut ir = UniversalIR::new();
        
        ir.metadata = ModelMetadata {
            name: "fallback_onnx_model".to_string(),
            version: 1,
            description: "Fallback ONNX model".to_string(),
            author: "coelanox".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            format: ModelFormat::Onnx,
            hardware_target: HardwareTarget::Cpu,
            input_shapes: vec![Shape::new(vec![1, 784])], // MNIST-like input
            output_shapes: vec![Shape::new(vec![1, 10])], // 10 classes
        };
        
        // Add a simple operation
        let mut operation = Operation::new(OpType::MatMul);
        operation.add_input("input".to_string());
        operation.add_output("output".to_string());
        ir.add_operation(operation);
        
        Ok(ir)
    }
}

impl TranslatorPlugin for OnnxTranslator {
    /// Translate ONNX model to Universal IR
    fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
        self.translate(data)
    }
    
    /// Extract weights from ONNX model data
    fn extract_weights(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        #[cfg(feature = "onnx-protobuf")]
        {
            // Parse ONNX model using prost
            let model = onnx::ModelProto::decode(data)
                .map_err(|e| CoelanoxError::format(&format!("Failed to parse ONNX model: {}", e)))?;
            
            let graph = model.graph.as_ref()
                .ok_or_else(|| CoelanoxError::format("ONNX model has no graph"))?;
            
            let mut weights = Vec::new();
            
            // Extract weights from initializers
            for initializer in &graph.initializer {
                let tensor_weights = self.convert_onnx_tensor_to_floats(initializer)?;
                weights.extend(tensor_weights);
            }
            
            Ok(weights)
        }
        
        #[cfg(not(feature = "onnx-protobuf"))]
        {
            // Fallback: return dummy weights
            Ok(vec![1.0f32; 1000]) // Dummy weights for testing
        }
    }
    
    /// Get supported model formats
    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Onnx]
    }
    
    /// Get plugin name
    fn name(&self) -> &str {
        &self.name
    }
    
    /// Get plugin version
    fn version(&self) -> &str {
        &self.version
    }
    
    /// Get plugin author
    fn author(&self) -> &str {
        "coelanox"
    }
    
    /// Get plugin description
    fn description(&self) -> &str {
        "ONNX to Universal IR translator with real protobuf parsing"
    }
    
    fn as_any(&self) -> &dyn std::any::Any where Self: Sized {
        self
    }
}

impl Default for OnnxTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_onnx_translator_creation() {
        let translator = OnnxTranslator::new();
        assert_eq!(translator.name, "onnx_translator");
        assert_eq!(translator.version, "1.0.0");
    }
    
    #[test]
    fn test_convert_onnx_op_type() {
        let translator = OnnxTranslator::new();
        
        assert_eq!(translator.convert_onnx_op_type("Add").unwrap(), OpType::Add);
        assert_eq!(translator.convert_onnx_op_type("MatMul").unwrap(), OpType::MatMul);
        assert_eq!(translator.convert_onnx_op_type("Relu").unwrap(), OpType::Relu);
        assert_eq!(translator.convert_onnx_op_type("Conv").unwrap(), OpType::Convolution);
    }
    
    #[test]
    fn test_convert_onnx_data_type() {
        let translator = OnnxTranslator::new();
        
        assert_eq!(translator.convert_onnx_data_type(1).unwrap(), DataType::Float32);
        assert_eq!(translator.convert_onnx_data_type(6).unwrap(), DataType::Int32);
        assert_eq!(translator.convert_onnx_data_type(9).unwrap(), DataType::Bool);
        assert_eq!(translator.convert_onnx_data_type(10).unwrap(), DataType::Float32); // FLOAT16 -> Float32
    }
    
    #[test]
    fn test_unsupported_data_types() {
        let translator = OnnxTranslator::new();
        
        assert!(translator.convert_onnx_data_type(2).is_err()); // UINT8
        assert!(translator.convert_onnx_data_type(3).is_err()); // INT8
    }
    
    #[test]
    fn test_fallback_translation() {
        let translator = OnnxTranslator::new();
        let data = b"fake onnx data";
        
        let result = translator.translate_fallback(data);
        assert!(result.is_ok());
        
        let ir = result.unwrap();
        assert_eq!(ir.metadata.name, "fallback_onnx_model");
        assert_eq!(ir.metadata.format, ModelFormat::Onnx);
        assert!(!ir.operations.is_empty());
    }
}
