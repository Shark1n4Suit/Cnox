//! Test the ONNX translator with a simple model
//! 
//! This test creates a minimal ONNX model and tests the translator functionality.

use onnx_translator::OnnxTranslator;
use coelanox_core::plugins::TranslatorPlugin;
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing COELANOX ONNX Translator");
    println!("=====================================");
    
    // Create the translator
    let translator = OnnxTranslator::new();
    
    println!("✅ Translator created: {} v{} by {}", 
             translator.name(), 
             translator.version(), 
             translator.author());
    
    // Check supported formats
    let formats = translator.supported_formats();
    println!("✅ Supported formats: {:?}", formats);
    
    // Create a minimal ONNX model for testing
    let test_model = create_minimal_onnx_model();
    
    println!("✅ Created minimal ONNX model ({} bytes)", test_model.len());
    
    // Test translation
    println!("\n🔄 Testing translation...");
    match translator.translate(&test_model) {
        Ok(ir) => {
            println!("✅ Translation successful!");
            println!("   Model name: {}", ir.metadata.name);
            println!("   Model version: {}", ir.metadata.version);
            println!("   Number of nodes: {}", ir.nodes.len());
            println!("   Number of weight tensors: {}", ir.weight_tensors.len());
            println!("   Number of intermediate tensors: {}", ir.intermediate_tensors.len());
            
            // Show some details about the IR
            if !ir.nodes.is_empty() {
                println!("   First node: {:?}", ir.nodes[0].op_type);
            }
            if !ir.weight_tensors.is_empty() {
                println!("   First weight tensor: {} {:?}", 
                         ir.weight_tensors[0].name, 
                         ir.weight_tensors[0].shape);
            }
        }
        Err(e) => {
            println!("❌ Translation failed: {}", e);
            return Err(e.into());
        }
    }
    
    // Test weight extraction
    println!("\n🔄 Testing weight extraction...");
    match translator.extract_weights(&test_model) {
        Ok(weights) => {
            println!("✅ Weight extraction successful!");
            println!("   Extracted {} weight values", weights.len());
            if !weights.is_empty() {
                println!("   First few weights: {:?}", &weights[..weights.len().min(5)]);
            }
        }
        Err(e) => {
            println!("❌ Weight extraction failed: {}", e);
            return Err(e.into());
        }
    }
    
    println!("\n🎉 All tests passed! The ONNX translator is working correctly.");
    Ok(())
}

/// Create a minimal ONNX model for testing
fn create_minimal_onnx_model() -> Vec<u8> {
    use onnx_translator::*;
    
    // Create a simple model with one Add operation
    let model = ModelProto {
        ir_version: 7,
        producer_name: "COELANOX Test".to_string(),
        producer_version: "1.0.0".to_string(),
        domain: "ai.coelanox".to_string(),
        model_version: 1,
        doc_string: "Test BERT-like model".to_string(),
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            doc_string: "Test computation graph".to_string(),
            node: vec![
                NodeProto {
                    name: "add_node".to_string(),
                    op_type: "Add".to_string(),
                    domain: "".to_string(),
                    doc_string: "Addition operation".to_string(),
                    input: vec!["input1".to_string(), "input2".to_string()],
                    output: vec!["output".to_string()],
                    attribute: vec![],
                }
            ],
            initializer: vec![
                TensorProto {
                    name: "weight1".to_string(),
                    data_type: 1, // FLOAT
                    dims: vec![128, 256],
                    float_data: vec![1.0; 128 * 256],
                    int32_data: vec![],
                    string_data: vec![],
                    int64_data: vec![],
                    raw_data: vec![],
                    double_data: vec![],
                    uint64_data: vec![],
                }
            ],
            input: vec![
                ValueInfoProto {
                    name: "input1".to_string(),
                    doc_string: "First input".to_string(),
                    r#type: Some(TypeProto {
                        denotation: "".to_string(),
                        tensor_type: Some(TensorTypeProto {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    DimensionProto {
                                        dim_value: 1,
                                        dim_param: "".to_string(),
                                    },
                                    DimensionProto {
                                        dim_value: 128,
                                        dim_param: "".to_string(),
                                    }
                                ],
                            }),
                        }),
                    }),
                },
                ValueInfoProto {
                    name: "input2".to_string(),
                    doc_string: "Second input".to_string(),
                    r#type: Some(TypeProto {
                        denotation: "".to_string(),
                        tensor_type: Some(TensorTypeProto {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    DimensionProto {
                                        dim_value: 1,
                                        dim_param: "".to_string(),
                                    },
                                    DimensionProto {
                                        dim_value: 128,
                                        dim_param: "".to_string(),
                                    }
                                ],
                            }),
                        }),
                    }),
                }
            ],
            output: vec![
                ValueInfoProto {
                    name: "output".to_string(),
                    doc_string: "Output tensor".to_string(),
                    r#type: Some(TypeProto {
                        denotation: "".to_string(),
                        tensor_type: Some(TensorTypeProto {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    DimensionProto {
                                        dim_value: 1,
                                        dim_param: "".to_string(),
                                    },
                                    DimensionProto {
                                        dim_value: 128,
                                        dim_param: "".to_string(),
                                    }
                                ],
                            }),
                        }),
                    }),
                }
            ],
            value_info: vec![],
        }),
    };
    
    // Encode the model to bytes
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("Failed to encode model");
    buf
}
