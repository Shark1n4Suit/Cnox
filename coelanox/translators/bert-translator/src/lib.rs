//! COELANOX BERT Translator
//! 
//! This crate provides a specialized build-time translator that converts BERT ONNX models
//! into COELANOX's UniversalIR format. It's optimized for BERT-specific operations and patterns,
//! making it faster and simpler than the general ONNX translator.
//! 
//! BERT-Specific Optimizations:
//! - Pattern recognition for attention blocks (Query, Key, Value MatMuls)
//! - Special handling for embedding layers (Gather operations)
//! - Recognition of BERT layer structure (12/24 layers)
//! - Efficient weight extraction for BERT weight naming conventions

use coelanox_core::{
    CoelanoxError, CoelanoxResult, 
    format::ModelFormat,
    plugins::TranslatorPlugin,
    ir::{
        UniversalIR, OpType, DataType, Shape, Attribute
    }
};
use prost::Message;
use std::collections::HashMap;

// Reuse ONNX protobuf definitions (same structure)
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
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(string, tag = "7")]
    pub domain: String,
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

#[derive(Debug, Clone, PartialEq, prost::Enumeration)]
#[repr(i32)]
pub enum AttributeProto_AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int32, repeated, tag = "1")]
    pub dims: Vec<i32>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(message, optional, tag = "3")]
    pub segment: Option<TensorProto_Segment>,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes, repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes, tag = "9")]
    pub doc_string: Vec<u8>,
    #[prost(bytes, tag = "10")]
    pub raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "11")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(enumeration = "TensorProto_DataLocation", tag = "12")]
    pub data_location: i32,
    #[prost(double, repeated, tag = "13")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "14")]
    pub uint64_data: Vec<u64>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto_Segment {
    #[prost(int64, tag = "1")]
    pub begin: i64,
    #[prost(int64, tag = "2")]
    pub end: i64,
}

#[derive(Debug, Clone, PartialEq, prost::Enumeration)]
#[repr(i32)]
pub enum TensorProto_DataLocation {
    Default = 0,
    External = 1,
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
    pub tensor_type: Option<TypeProto_Tensor>,
    #[prost(string, tag = "4")]
    pub denotation: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto_Tensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeProto_Dimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto_Dimension {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
    #[prost(string, tag = "2")]
    pub dim_param: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

/// BERT-specific translator optimized for BERT ONNX models
pub struct BertTranslator {
    name: String,
    version: String,
    author: String,
}

impl BertTranslator {
    /// Create a new BERT translator
    pub fn new() -> Self {
        Self {
            name: "bert_translator".to_string(),
            version: "0.1.0".to_string(),
            author: "Benjamin Morin".to_string(),
        }
    }
    
    /// Detect if a model is BERT by checking operations and structure
    fn is_bert_model(&self, onnx_model: &ModelProto) -> bool {
        if let Some(graph) = &onnx_model.graph {
            // Check for BERT-specific patterns:
            // 1. Has embedding operations (Gather for embeddings)
            // 2. Has attention-like patterns (MatMul operations)
            // 3. Has LayerNorm operations
            // 4. Has GELU activations
            let has_embedding = graph.node.iter().any(|n| {
                n.op_type == "Gather" && n.name.contains("embedding")
            });
            
            let has_attention = graph.node.iter().any(|n| {
                n.op_type == "MatMul" && (n.name.contains("attention") || n.name.contains("qkv"))
            });
            
            let has_layernorm = graph.node.iter().any(|n| {
                n.op_type == "LayerNormalization" || n.name.contains("layernorm")
            });
            
            let has_gelu = graph.node.iter().any(|n| {
                n.op_type == "Gelu" || n.name.contains("gelu")
            });
            
            // BERT typically has all of these
            (has_embedding || has_attention) && (has_layernorm || has_gelu)
        } else {
            false
        }
    }
    
    /// Map BERT-specific ONNX operations to UniversalIR
    fn map_bert_operation(&self, op_type: &str, node: &NodeProto) -> OpType {
        match op_type {
            // Embedding operations
            "Gather" => {
                // Check if it's an embedding lookup
                if node.name.contains("embedding") || node.name.contains("gather") {
                    OpType::Custom("Embedding".to_string())
                } else {
                    OpType::Gather
                }
            }
            
            // Attention operations
            "MatMul" => {
                // Check for attention patterns (Query, Key, Value)
                if node.name.contains("query") || node.name.contains("key") || node.name.contains("value") {
                    OpType::Custom("AttentionMatMul".to_string())
                } else {
                    OpType::MatMul
                }
            }
            
            // Layer normalization
            "LayerNormalization" => OpType::LayerNorm,
            
            // GELU activation (BERT-specific)
            "Gelu" | "GELU" => OpType::Gelu,
            
            // Standard operations
            "Add" => OpType::Add,
            "Sub" => OpType::Subtract,
            "Mul" => OpType::Multiply,
            "Div" => OpType::Divide,
            "Relu" => OpType::Relu,
            "Sigmoid" => OpType::Sigmoid,
            "Tanh" => OpType::Tanh,
            "Softmax" => OpType::Softmax,
            "Reshape" => OpType::Reshape,
            "Transpose" => OpType::Transpose,
            "Concat" => OpType::Concatenate,
            "Split" => OpType::Split,
            "Slice" => OpType::Slice,
            "ReduceSum" => OpType::ReduceSum,
            "ReduceMean" => OpType::ReduceMean,
            "ReduceMax" => OpType::ReduceMax,
            "ReduceMin" => OpType::ReduceMin,
            
            // Fallback for unknown operations
            _ => OpType::Custom(op_type.to_string()),
        }
    }
    
    /// Convert ONNX attribute to UniversalIR attribute
    fn convert_attribute(&self, attr: &AttributeProto) -> Attribute {
        match attr.r#type {
            1 => Attribute::Float(attr.f), // FLOAT
            2 => Attribute::Int(attr.i), // INT
            3 => Attribute::String(String::from_utf8_lossy(&attr.s).to_string()), // STRING
            6 => Attribute::FloatArray(attr.floats.clone()), // FLOATS
            7 => Attribute::IntArray(attr.ints.clone()), // INTS
            _ => Attribute::String(format!("{:?}", attr)),
        }
    }
    
    /// Extract weight tensors from BERT model
    fn extract_weight_tensors(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            println!("📊 Extracting {} weight tensors...", graph.initializer.len());
            let mut weight_count = 0;
            for initializer in &graph.initializer {
                weight_count += 1;
                if weight_count % 100 == 0 {
                    println!("   Processed {}/{} weight tensors...", weight_count, graph.initializer.len());
                }
                // Get tensor shape
                let dims: Vec<usize> = initializer.dims.iter()
                    .map(|&d| if d > 0 { d as usize } else { 1 })
                    .collect();
                
                let shape = Shape::new(dims);
                let data_type = match initializer.data_type {
                    1 => DataType::Float32, // FLOAT
                    2 => DataType::UInt8, // UINT8
                    6 => DataType::Int32, // INT32
                    7 => DataType::Int64, // INT64
                    9 => DataType::Bool, // BOOL
                    10 => DataType::Float32, // FLOAT16 -> Float32 (closest match)
                    11 => DataType::Float64, // DOUBLE
                    _ => DataType::Float32, // Default to Float32
                };
                
                // Use initializer name or generate one
                let name = if initializer.name.is_empty() {
                    format!("bert_weight_{}", ir.weight_tensors.len())
                } else {
                    initializer.name.clone()
                };
                
                // Extract raw data for weights
                // Safety check: limit individual tensor size to prevent memory issues
                let raw_data = if !initializer.raw_data.is_empty() {
                    if initializer.raw_data.len() > 500 * 1024 * 1024 { // 500MB per tensor max
                        eprintln!("⚠️  Warning: Skipping large weight tensor {} ({}MB)", 
                                 initializer.name, initializer.raw_data.len() / (1024 * 1024));
                        continue;
                    }
                    initializer.raw_data.clone()
                } else {
                    // Convert from float_data if raw_data is not available
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
                                bytes.extend_from_slice(&(val as i32).to_le_bytes());
                            }
                            bytes
                        }
                        _ => Vec::new(),
                    }
                };
                
                // Add weight tensor to IR (with data)
                ir.add_weight_tensor_with_data(name, shape, data_type, raw_data);
            }
            println!("✅ Extracted {} weight tensors", weight_count);
        }
        
        Ok(())
    }
    
    /// Extract intermediate tensors (from value_info, inputs, outputs)
    fn extract_intermediate_tensors(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            // Process all value infos (inputs, outputs, intermediate values)
            let mut value_infos = Vec::new();
            value_infos.extend(&graph.input);
            value_infos.extend(&graph.output);
            value_infos.extend(&graph.value_info);
            
            for value_info in value_infos {
                // Skip if this is a weight tensor (from initializers)
                let is_weight = graph.initializer.iter().any(|init| init.name == value_info.name);
                if is_weight {
                    continue;
                }
                
                if let Some(type_proto) = &value_info.r#type {
                    if let Some(tensor_type) = &type_proto.tensor_type {
                        let dims: Vec<usize> = if let Some(shape) = &tensor_type.shape {
                            shape.dim.iter()
                                .map(|d| {
                                    if d.dim_value > 0 {
                                        d.dim_value as usize
                                    } else if !d.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults for BERT
                                        match d.dim_param.as_str() {
                                            "batch_size" | "N" => 1,
                                            "sequence_length" | "seq_len" | "max_length" => 512,
                                            "hidden_size" | "d_model" => 768, // BERT hidden size
                                            "vocab_size" => 30522, // BERT vocab size
                                            _ => 1,
                                        }
                                    } else {
                                        1
                                    }
                                })
                                .collect()
                        } else {
                            vec![1] // Default shape
                        };
                        
                        let shape = Shape::new(dims);
                        let data_type = match tensor_type.elem_type {
                            1 => DataType::Float32,
                            2 => DataType::UInt8,
                            6 => DataType::Int32,
                            7 => DataType::Int64,
                            9 => DataType::Bool,
                            10 => DataType::Float32, // FLOAT16 -> Float32 (closest match)
                            11 => DataType::Float64,
                            _ => DataType::Float32,
                        };
                        
                        ir.add_intermediate_tensor(value_info.name.clone(), shape, data_type);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Build computation graph from BERT ONNX nodes
    fn build_computation_graph(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        if let Some(graph) = &onnx_model.graph {
            // Create tensor name to ID mapping
            let mut tensor_map: HashMap<String, coelanox_core::ir::TensorId> = HashMap::new();
            
            // Map input tensors (from graph.input)
            for input in &graph.input {
                if let Some((tensor_id, _)) = ir.get_tensor_by_name(&input.name) {
                    tensor_map.insert(input.name.clone(), tensor_id);
                }
            }
            
            // Map weight tensors (from initializers)
            for initializer in &graph.initializer {
                let name = if initializer.name.is_empty() {
                    format!("bert_weight_{}", tensor_map.len())
                } else {
                    initializer.name.clone()
                };
                
                if let Some((tensor_id, _)) = ir.get_tensor_by_name(&name) {
                    tensor_map.insert(name.clone(), tensor_id);
                }
            }
            
            // Process nodes and build graph
            for node in &graph.node {
                // Map operation type
                let op_type = self.map_bert_operation(&node.op_type, node);
                
                // Create graph node
                let node_id = ir.add_node(op_type);
                
                // Add operation name and attributes
                if let Some(graph_node) = ir.get_node_mut(node_id) {
                    // Add operation name
                    graph_node.attributes.insert("operation_name".to_string(), 
                        Attribute::String(node.name.clone()));
                    
                    // Convert attributes
                    for attr in &node.attribute {
                        let coelanox_attr = self.convert_attribute(attr);
                        graph_node.attributes.insert(attr.name.clone(), coelanox_attr);
                    }
                }
                
                // Map input tensors
                for input_name in &node.input {
                    if let Some(&tensor_id) = tensor_map.get(input_name) {
                        if let Err(e) = ir.add_node_input(node_id, tensor_id) {
                            eprintln!("Warning: Failed to add input {} to node {}: {}", input_name, node.name, e);
                        }
                    } else {
                        eprintln!("Warning: Input tensor {} not found for node {}", input_name, node.name);
                    }
                }
                
                // Map output tensors (create intermediate tensors if needed)
                for output_name in &node.output {
                    let tensor_id = if let Some(&tid) = tensor_map.get(output_name) {
                        tid
                    } else {
                        // Create intermediate tensor for output
                        let shape = Shape::new(vec![1]); // Default shape, will be updated
                        let tensor_id = ir.add_intermediate_tensor(
                            output_name.clone(),
                            shape,
                            DataType::Float32
                        );
                        tensor_map.insert(output_name.clone(), tensor_id);
                        tensor_id
                    };
                    
                    if let Err(e) = ir.add_node_output(node_id, tensor_id) {
                        eprintln!("Warning: Failed to add output {} to node {}: {}", output_name, node.name, e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Set model metadata
    fn set_model_metadata(&self, ir: &mut UniversalIR, onnx_model: &ModelProto) -> CoelanoxResult<()> {
        // Set model name
        ir.metadata.name = if onnx_model.doc_string.is_empty() {
            "bert_model".to_string()
        } else {
            onnx_model.doc_string.clone()
        };
        
        // Set version
        ir.metadata.version = onnx_model.model_version.to_string();
        
        // Extract input/output shapes from graph
        if let Some(graph) = &onnx_model.graph {
            // Input shapes
            for input in &graph.input {
                // Skip inputs that are weight tensors (from initializers)
                let is_weight = graph.initializer.iter().any(|init| init.name == input.name);
                if is_weight {
                    continue;
                }
                
                if let Some(type_proto) = &input.r#type {
                    if let Some(tensor_type) = &type_proto.tensor_type {
                        let dims: Vec<usize> = if let Some(shape) = &tensor_type.shape {
                            shape.dim.iter()
                                .map(|d| {
                                    if d.dim_value > 0 {
                                        d.dim_value as usize
                                    } else if !d.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults for BERT
                                        match d.dim_param.as_str() {
                                            "batch_size" | "N" => 1,
                                            "sequence_length" | "seq_len" | "max_length" => 512,
                                            "hidden_size" | "d_model" => 768, // BERT hidden size
                                            "vocab_size" => 30522, // BERT vocab size
                                            _ => 1,
                                        }
                                    } else {
                                        1
                                    }
                                })
                                .collect()
                        } else {
                            vec![1]
                        };
                        
                        if !dims.is_empty() {
                            ir.metadata.add_input_shape(Shape::new(dims));
                        }
                    }
                }
            }
            
            // Output shapes
            for output in &graph.output {
                if let Some(type_proto) = &output.r#type {
                    if let Some(tensor_type) = &type_proto.tensor_type {
                        let dims: Vec<usize> = if let Some(shape) = &tensor_type.shape {
                            shape.dim.iter()
                                .map(|d| {
                                    if d.dim_value > 0 {
                                        d.dim_value as usize
                                    } else if !d.dim_param.is_empty() {
                                        // Handle dynamic dimensions with reasonable defaults for BERT
                                        match d.dim_param.as_str() {
                                            "batch_size" | "N" => 1,
                                            "sequence_length" | "seq_len" | "max_length" => 512,
                                            "hidden_size" | "d_model" => 768, // BERT hidden size
                                            "vocab_size" => 30522, // BERT vocab size
                                            _ => 1,
                                        }
                                    } else {
                                        1
                                    }
                                })
                                .collect()
                        } else {
                            vec![1]
                        };
                        
                        if !dims.is_empty() {
                            ir.metadata.add_output_shape(Shape::new(dims));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl Default for BertTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslatorPlugin for BertTranslator {
    /// Translate BERT ONNX model to Universal IR
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        // Check model size to prevent excessive memory usage
        const MAX_MODEL_SIZE: usize = 2 * 1024 * 1024 * 1024; // 2GB max
        if model_data.len() > MAX_MODEL_SIZE {
            return Err(CoelanoxError::Translation(
                format!("Model size ({}) exceeds maximum allowed size ({}MB)", 
                        model_data.len(), MAX_MODEL_SIZE / (1024 * 1024))
            ));
        }
        
        println!("📦 Parsing BERT ONNX model ({}MB)...", model_data.len() / (1024 * 1024));
        
        // Parse ONNX model from raw bytes
        let onnx_model = ModelProto::decode(model_data)
            .map_err(|e| CoelanoxError::Translation(format!("Failed to parse BERT ONNX model: {}", e)))?;
        
        println!("✅ BERT ONNX model parsed successfully");
        
        // Verify it's a BERT model (optional check)
        if !self.is_bert_model(&onnx_model) {
            eprintln!("⚠️  Warning: Model may not be BERT-specific, but continuing translation...");
        }
        
        // Create Universal IR
        let mut ir = UniversalIR::new(
            if onnx_model.doc_string.is_empty() {
                "bert_model".to_string()
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
        if let Some(graph) = &onnx_model.graph {
            println!("🔄 Processing {} graph nodes...", graph.node.len());
        }
        self.build_computation_graph(&mut ir, &onnx_model)?;
        println!("✅ Computation graph built successfully");
        
        // Validate the IR
        match ir.validate() {
            Ok(_) => println!("✅ BERT IR validation passed"),
            Err(e) => {
                eprintln!("⚠️  BERT IR validation failed: {}", e);
                eprintln!("   Continuing with potentially invalid IR...");
            }
        }
        
        Ok(ir)
    }
    
    /// Extract weights from BERT ONNX model
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        let onnx_model = ModelProto::decode(model_data)
            .map_err(|e| CoelanoxError::Translation(format!("Failed to parse BERT ONNX model: {}", e)))?;
        
        let mut weights = Vec::new();
        
        // Extract weights from initializers
        if let Some(graph) = &onnx_model.graph {
            for initializer in &graph.initializer {
                if !initializer.raw_data.is_empty() {
                    // Convert raw bytes to f32 values
                    let f32_values: Vec<f32> = initializer.raw_data
                        .chunks(4)
                        .filter_map(|chunk| {
                            if chunk.len() == 4 {
                                Some(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            } else {
                                None
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
        vec![ModelFormat::Onnx] // Uses ONNX format but specialized for BERT
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

