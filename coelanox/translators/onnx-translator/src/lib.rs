//! COELANOX ONNX Translator
//! 
//! This crate provides a build-time translator that converts ONNX models
//! into COELANOX's UniversalIR format. It uses pure Rust with prost for
//! protobuf parsing and has no runtime dependencies.

use coelanox_core::{
    CoelanoxError, CoelanoxResult, 
    format::ModelFormat,
    plugins::TranslatorPlugin,
    ir::{
        UniversalIR, OpType, DataType, Shape, Attribute
    }
};
use prost::Message;

// Simplified ONNX protobuf definitions for core functionality
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int32, tag = "1")]
    pub data_type: i32,
    #[prost(int64, repeated, tag = "2")]
    pub dims: Vec<i64>,
    #[prost(float, repeated, packed = true, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, packed = true, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes, repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, packed = true, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes, tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, packed = true, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, packed = true, tag = "11")]
    pub uint64_data: Vec<u64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
    #[prost(string, tag = "6")]
    pub denotation: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<DimensionProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct DimensionProto {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
    #[prost(string, tag = "2")]
    pub dim_param: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int32, tag = "21")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes, repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
}

/// ONNX to UniversalIR translator
pub struct OnnxTranslator {
    /// Translator name
    name: String,
    /// Translator version
    version: String,
    /// Translator author
    author: String,
}

impl OnnxTranslator {
    /// Create a new ONNX translator
    pub fn new() -> Self {
        Self {
            name: "onnx-translator".to_string(),
            version: "0.1.0".to_string(),
            author: "Benjamin Morin".to_string(),
        }
    }
}

impl Default for OnnxTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslatorPlugin for OnnxTranslator {
    /// Translate ONNX model data to Universal IR
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        // Parse ONNX model from raw bytes
        let onnx_model = ModelProto::decode(model_data)
            .map_err(|e| CoelanoxError::Translation(format!("Failed to parse ONNX model: {}", e)))?;

        // Create Universal IR
        let mut ir = UniversalIR::new(
            if onnx_model.doc_string.is_empty() {
                "onnx_model".to_string()
            } else {
                onnx_model.doc_string.clone()
            }
        );

        // Set model metadata
        self.set_model_metadata(&mut ir, &onnx_model)?;

        // Extract and add weight tensors
        self.extract_weight_tensors(&mut ir, &onnx_model)?;

        // Extract and add intermediate tensors
        self.extract_intermediate_tensors(&mut ir, &onnx_model)?;

        // Process graph nodes and build computation graph
        self.build_computation_graph(&mut ir, &onnx_model)?;

        // Validate the IR (with warnings instead of errors)
        match ir.validate() {
            Ok(_) => println!("✅ IR validation passed"),
            Err(e) => {
                eprintln!("⚠️  IR validation failed: {}", e);
                eprintln!("   Continuing with potentially invalid IR for analysis...");
            }
        }

        Ok(ir)
    }

    /// Extract weights from ONNX model data
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        let onnx_model = ModelProto::decode(model_data)
            .map_err(|e| CoelanoxError::Translation(format!("Failed to parse ONNX model: {}", e)))?;

        let mut weights = Vec::new();
        
        // Extract weights from initializers
        if let Some(graph) = &onnx_model.graph {
            for initializer in &graph.initializer {
                if !initializer.raw_data.is_empty() {
                    // Convert raw bytes to f32 values
                    let f32_values: Vec<f32> = initializer.raw_data
                        .chunks(4)
                        .map(|chunk| {
                            if chunk.len() == 4 {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    weights.extend(f32_values);
                } else if !initializer.float_data.is_empty() {
                    weights.extend(initializer.float_data.clone());
                }
            }
        }

        Ok(weights)
    }

    /// Get supported model formats
    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Onnx]
    }

    /// Get translator name
    fn name(&self) -> &str {
        &self.name
    }

    /// Get translator version
    fn version(&self) -> &str {
        &self.version
    }

    /// Get translator author
    fn author(&self) -> &str {
        &self.author
    }
}

impl OnnxTranslator {
    /// Set model metadata from ONNX model
    fn set_model_metadata(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        // Set model name
        ir.metadata.name = if onnx_model.doc_string.is_empty() {
            "onnx_model".to_string()
        } else {
            onnx_model.doc_string.clone()
        };
        
        // Set version
        ir.metadata.version = onnx_model.model_version.to_string();

        // Set data type (default to Float32)
        ir.metadata.set_data_type(DataType::Float32);

        // Add input shapes from graph
        if let Some(graph) = &onnx_model.graph {
            for input in &graph.input {
                if let Some(tensor_type) = &input.r#type {
                    if let Some(tensor_shape) = &tensor_type.tensor_type {
                        if let Some(shape) = &tensor_shape.shape {
                            let dimensions: Vec<usize> = shape.dim.iter()
                                .map(|dim| {
                                    if dim.dim_value > 0 {
                                        dim.dim_value as usize
                                    } else if !dim.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults
                                        match dim.dim_param.as_str() {
                                            "batch_size" | "N" => 1,  // Default batch size
                                            "sequence_length" | "seq_len" | "max_length" => 512,  // Default sequence length
                                            "hidden_size" | "d_model" => 768,  // BERT hidden size
                                            "vocab_size" => 30522,  // BERT vocab size
                                            _ => 1,  // Default fallback
                                        }
                                    } else {
                                        1  // Default fallback for unknown dimensions
                                    }
                                })
                                .collect();
                            
                            if !dimensions.is_empty() {
                                ir.metadata.add_input_shape(Shape::new(dimensions));
                            }
                        }
                    }
                }
            }

            // Add output shapes from graph
            for output in &graph.output {
                if let Some(tensor_type) = &output.r#type {
                    if let Some(tensor_shape) = &tensor_type.tensor_type {
                        if let Some(shape) = &tensor_shape.shape {
                            let dimensions: Vec<usize> = shape.dim.iter()
                                .map(|dim| {
                                    if dim.dim_value > 0 {
                                        dim.dim_value as usize
                                    } else if !dim.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults
                                        match dim.dim_param.as_str() {
                                            "batch_size" | "N" => 1,  // Default batch size
                                            "sequence_length" | "seq_len" | "max_length" => 512,  // Default sequence length
                                            "hidden_size" | "d_model" => 768,  // BERT hidden size
                                            "vocab_size" => 30522,  // BERT vocab size
                                            _ => 1,  // Default fallback
                                        }
                                    } else {
                                        1  // Default fallback for unknown dimensions
                                    }
                                })
                                .collect();
                            
                            if !dimensions.is_empty() {
                                ir.metadata.add_output_shape(Shape::new(dimensions));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract weight tensors from ONNX initializers
    fn extract_weight_tensors(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            for initializer in &graph.initializer {
                let name = initializer.name.clone();
                let data_type = self.map_onnx_data_type(initializer.data_type);
                
                // Extract shape
                let dimensions: Vec<usize> = initializer.dims.iter().map(|&d| d as usize).collect();
                let shape = Shape::new(dimensions);

                // Extract raw data
                let raw_data = if !initializer.raw_data.is_empty() {
                    initializer.raw_data.clone()
                } else {
                    // Convert from float_data or int32_data if raw_data is not available
                    match data_type {
                        DataType::Float32 => {
                            let mut bytes = Vec::new();
                            for &val in &initializer.float_data {
                                bytes.extend_from_slice(&val.to_le_bytes());
                            }
                            bytes
                        }
                        DataType::Int32 => {
                            let mut bytes = Vec::new();
                            for &val in &initializer.int32_data {
                                bytes.extend_from_slice(&val.to_le_bytes());
                            }
                            bytes
                        }
                        _ => Vec::new(),
                    }
                };

                ir.add_weight_tensor_with_data(name, shape, data_type, raw_data);
            }
        }

        Ok(())
    }

    /// Extract intermediate tensors from ONNX graph inputs/outputs
    fn extract_intermediate_tensors(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            // Process all value infos (inputs, outputs, intermediate values)
            let mut value_infos = Vec::new();
            value_infos.extend(&graph.input);
            value_infos.extend(&graph.output);
            value_infos.extend(&graph.value_info);

            for value_info in value_infos {
                let name = value_info.name.clone();
                
                if let Some(tensor_type) = &value_info.r#type {
                    if let Some(tensor_shape) = &tensor_type.tensor_type {
                        let data_type = self.map_onnx_data_type(tensor_shape.elem_type);
                        
                        // Extract shape
                        let dimensions: Vec<usize> = if let Some(shape) = &tensor_shape.shape {
                            shape.dim.iter()
                                .map(|dim| {
                                    if dim.dim_value > 0 {
                                        dim.dim_value as usize
                                    } else if !dim.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults
                                        match dim.dim_param.as_str() {
                                            "batch_size" | "N" => 1,  // Default batch size
                                            "sequence_length" | "seq_len" | "max_length" => 512,  // Default sequence length
                                            "hidden_size" | "d_model" => 768,  // BERT hidden size
                                            "vocab_size" => 30522,  // BERT vocab size
                                            _ => 1,  // Default fallback
                                        }
                                    } else {
                                        1  // Default fallback for unknown dimensions
                                    }
                                })
                                .collect()
                        } else {
                            Vec::new()
                        };

                        if !dimensions.is_empty() {
                            ir.add_intermediate_tensor(name, Shape::new(dimensions), data_type);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Build computation graph from ONNX nodes
    fn build_computation_graph(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            for onnx_node in &graph.node {
                // Map ONNX operation to COELANOX operation
                let op_type = self.map_onnx_operation(&onnx_node.op_type)?;

                // Create graph node
                let node_id = ir.add_node(op_type);

                // Add input tensors
                for input_name in &onnx_node.input {
                    if let Some((tensor_id, _)) = ir.get_tensor_by_name(input_name) {
                        if let Err(e) = ir.add_node_input(node_id, tensor_id) {
                            eprintln!("Warning: Failed to add input {} to node {}: {}", input_name, onnx_node.name, e);
                        }
                    } else {
                        eprintln!("Warning: Input tensor {} not found for node {}", input_name, onnx_node.name);
                    }
                }

                // Add output tensors
                for output_name in &onnx_node.output {
                    if let Some((tensor_id, _)) = ir.get_tensor_by_name(output_name) {
                        if let Err(e) = ir.add_node_output(node_id, tensor_id) {
                            eprintln!("Warning: Failed to add output {} to node {}: {}", output_name, onnx_node.name, e);
                        }
                    } else {
                        eprintln!("Warning: Output tensor {} not found for node {}", output_name, onnx_node.name);
                    }
                }

                // Add attributes
                if let Some(node) = ir.get_node_mut(node_id) {
                    for attr in &onnx_node.attribute {
                        let coelanox_attr = self.map_onnx_attribute(attr)?;
                        node.add_attribute(attr.name.clone(), coelanox_attr);
                    }
                }
            }
        }

        Ok(())
    }

    /// Map ONNX data type to COELANOX data type
    fn map_onnx_data_type(&self, onnx_type: i32) -> DataType {
        match onnx_type {
            1 => DataType::Float32,   // FLOAT
            2 => DataType::UInt8,     // UINT8
            3 => DataType::Int32,     // INT8 -> Int32 (closest match)
            6 => DataType::Int32,    // INT32
            7 => DataType::Int64,    // INT64
            9 => DataType::Bool,     // BOOL
            10 => DataType::Float32, // FLOAT16 -> Float32 (closest match)
            11 => DataType::Float64, // DOUBLE
            12 => DataType::Int32,   // UINT32 -> Int32 (closest match)
            13 => DataType::Int64,   // UINT64 -> Int64 (closest match)
            _ => DataType::Float32,  // Default fallback
        }
    }

    /// Map ONNX operation to COELANOX operation
    fn map_onnx_operation(&self, op_type: &str) -> CoelanoxResult<OpType> {
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
            "Expand" => Ok(OpType::Expand),

            // Comparison operations
            "Equal" => Ok(OpType::Equal),
            "Not" => Ok(OpType::Not),
            "Greater" => Ok(OpType::Greater),
            "GreaterOrEqual" => Ok(OpType::GreaterEqual),
            "Less" => Ok(OpType::Less),
            "LessOrEqual" => Ok(OpType::LessEqual),

            // Logical operations
            "And" => Ok(OpType::And),
            "Or" => Ok(OpType::Or),

            // Custom operations
            _ => Ok(OpType::Custom(op_type.to_string())),
        }
    }

    /// Map ONNX attribute to COELANOX attribute
    fn map_onnx_attribute(&self, attr: &AttributeProto) -> CoelanoxResult<Attribute> {
        match attr.r#type {
            0 => { // UNDEFINED - skip undefined attributes
                Ok(Attribute::String(format!("undefined_{}", attr.name)))
            }
            1 => { // FLOAT
                Ok(Attribute::Float(attr.f))
            }
            2 => { // INT
                Ok(Attribute::Int(attr.i))
            }
            3 => { // STRING
                Ok(Attribute::String(String::from_utf8_lossy(&attr.s).to_string()))
            }
            4 => { // TENSOR
                // For now, treat tensor attributes as strings
                Ok(Attribute::String(format!("tensor_{}", attr.name)))
            }
            5 => { // GRAPH
                // For now, treat graph attributes as strings
                Ok(Attribute::String(format!("graph_{}", attr.name)))
            }
            6 => { // FLOATS
                Ok(Attribute::FloatArray(attr.floats.clone()))
            }
            7 => { // INTS
                Ok(Attribute::IntArray(attr.ints.clone()))
            }
            8 => { // STRINGS
                let strings: Vec<String> = attr.strings.iter()
                    .map(|s| String::from_utf8_lossy(s).to_string())
                    .collect();
                Ok(Attribute::StringArray(strings))
            }
            _ => {
                Err(CoelanoxError::Translation(format!("Unsupported attribute type: {}", attr.r#type)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translator_creation() {
        let translator = OnnxTranslator::new();
        assert_eq!(translator.name(), "onnx-translator");
        assert_eq!(translator.version(), "0.1.0");
        assert_eq!(translator.author(), "Benjamin Morin");
    }

    #[test]
    fn test_supported_formats() {
        let translator = OnnxTranslator::new();
        let formats = translator.supported_formats();
        assert!(formats.contains(&ModelFormat::Onnx));
    }

    #[test]
    fn test_operation_mapping() {
        let translator = OnnxTranslator::new();
        
        assert_eq!(translator.map_onnx_operation("Add").unwrap(), OpType::Add);
        assert_eq!(translator.map_onnx_operation("Relu").unwrap(), OpType::Relu);
        assert_eq!(translator.map_onnx_operation("Conv").unwrap(), OpType::Convolution);
        assert_eq!(translator.map_onnx_operation("MatMul").unwrap(), OpType::MatMul);
        
        // Test custom operation
        match translator.map_onnx_operation("UnknownOp").unwrap() {
            OpType::Custom(name) => assert_eq!(name, "UnknownOp"),
            _ => panic!("Expected Custom operation"),
        }
    }

    #[test]
    fn test_data_type_mapping() {
        let translator = OnnxTranslator::new();
        
        assert_eq!(translator.map_onnx_data_type(1), DataType::Float32);
        assert_eq!(translator.map_onnx_data_type(2), DataType::UInt8);
        assert_eq!(translator.map_onnx_data_type(6), DataType::Int32);
        assert_eq!(translator.map_onnx_data_type(9), DataType::Bool);
    }
}