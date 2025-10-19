//! Tests for COELANOX weight extraction, compression, and inference
//! 
//! This module tests the complete weight system from ONNX extraction to runtime inference.

use coelanox::{CoelanoxPackager, CoelanoxRuntime, format::HardwareTarget};
use std::path::Path;
use tempfile::TempDir;

#[cfg(test)]
mod weight_tests {
    use super::*;

    #[test]
    fn test_weight_extraction_from_onnx() {
        println!("🧪 Testing weight extraction from ONNX...");
        
        // Create a simple ONNX model with weights
        let onnx_model = create_test_onnx_model();
        
        // Create packager
        let packager = CoelanoxPackager::new().expect("Failed to create packager");
        
        // Write ONNX model to temporary file
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = temp_dir.path().join("test_model.onnx");
        std::fs::write(&model_path, &onnx_model).expect("Failed to write ONNX model");
        
        // Package the model
        let container = packager.package(&model_path, HardwareTarget::Cpu)
            .expect("Failed to package model");
        
        // Verify weights were extracted and compressed
        assert!(!container.compressed_weights.is_empty(), "No weights extracted");
        println!("✅ Extracted {} bytes of compressed weights", container.compressed_weights.len());
        
        // Test decompression
        let decompressed_weights = container.decompress_weights()
            .expect("Failed to decompress weights");
        
        assert!(!decompressed_weights.is_empty(), "No weights after decompression");
        println!("✅ Decompressed {} weights", decompressed_weights.len());
        
        // Verify weights are reasonable (not all zeros)
        let has_non_zero = decompressed_weights.iter().any(|&w| w != 0.0);
        assert!(has_non_zero, "All weights are zero");
        println!("✅ Weights contain non-zero values");
    }

    #[test]
    fn test_compression_efficiency() {
        println!("🧪 Testing compression efficiency...");
        
        // Create test weights
        let test_weights = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original_size = test_weights.len() * 4; // 4 bytes per f32
        
        // Create packager
        let packager = CoelanoxPackager::new().expect("Failed to create packager");
        
        // Compress weights
        let (compressed_data, compression_type) = packager.compress_weights(&test_weights)
            .expect("Failed to compress weights");
        
        println!("✅ Compression type: {:?}", compression_type);
        println!("✅ Original size: {} bytes", original_size);
        println!("✅ Compressed size: {} bytes", compressed_data.len());
        
        // Verify compression worked
        assert!(!compressed_data.is_empty(), "Compression failed");
        
        // Test decompression
        let mut container = create_test_container_with_weights(compressed_data, compression_type);
        let decompressed = container.decompress_weights().expect("Failed to decompress");
        
        assert_eq!(decompressed.len(), test_weights.len(), "Decompressed size mismatch");
        
        // Verify values match (within floating point precision)
        for (i, (&original, &decompressed)) in test_weights.iter().zip(decompressed.iter()).enumerate() {
            assert!((original - decompressed).abs() < 0.001, 
                "Weight {} mismatch: {} vs {}", i, original, decompressed);
        }
        
        println!("✅ Compression and decompression successful");
    }

    #[test]
    fn test_real_inference_with_weights() {
        println!("🧪 Testing real inference with weights...");
        
        // Create a simple ONNX model
        let onnx_model = create_test_onnx_model();
        
        // Package the model
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_path = temp_dir.path().join("test_model.onnx");
        std::fs::write(&model_path, &onnx_model).expect("Failed to write ONNX model");
        
        let packager = CoelanoxPackager::new().expect("Failed to create packager");
        let container = packager.package(&model_path, HardwareTarget::Cpu)
            .expect("Failed to package model");
        
        // Create runtime
        let mut runtime = CoelanoxRuntime::new();
        runtime.load_container("test_model", container).expect("Failed to load container");
        
        // Test input data
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        // Execute inference
        let output = runtime.execute("test_model", &input_data)
            .expect("Failed to execute inference");
        
        // Verify output
        assert!(!output.is_empty(), "No output from inference");
        println!("✅ Inference output: {:?}", output);
        
        // Verify output is not all zeros (real computation)
        let has_non_zero = output.iter().any(|&x| x != 0.0);
        assert!(has_non_zero, "All output values are zero - inference not working");
        println!("✅ Inference produces non-zero results");
        
        // Verify output size matches expected
        let expected_output_size: usize = runtime.get_container("test_model")
            .unwrap()
            .manifest
            .output_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        
        assert_eq!(output.len(), expected_output_size, "Output size mismatch");
        println!("✅ Output size matches expected: {}", output.len());
    }

    #[test]
    fn test_operation_execution_with_weights() {
        println!("🧪 Testing operation execution with weights...");
        
        // Create runtime
        let runtime = CoelanoxRuntime::new();
        
        // Test Add operation
        let add_op = create_test_add_operation();
        let input_data = vec![1.0f32, 2.0, 3.0];
        let weights = vec![0.5f32, 1.0, 1.5];
        
        let add_output = runtime.execute_operation_with_weights(&add_op, &input_data, &weights)
            .expect("Failed to execute Add operation");
        
        // Verify Add: input[i] + weights[i]
        let expected_add = vec![1.5f32, 3.0, 4.5];
        assert_eq!(add_output, expected_add, "Add operation failed");
        println!("✅ Add operation: {:?} + {:?} = {:?}", input_data, weights, add_output);
        
        // Test Multiply operation
        let mul_op = create_test_multiply_operation();
        let mul_output = runtime.execute_operation_with_weights(&mul_op, &input_data, &weights)
            .expect("Failed to execute Multiply operation");
        
        // Verify Multiply: input[i] * weights[i]
        let expected_mul = vec![0.5f32, 2.0, 4.5];
        assert_eq!(mul_output, expected_mul, "Multiply operation failed");
        println!("✅ Multiply operation: {:?} * {:?} = {:?}", input_data, weights, mul_output);
        
        // Test Activation operation
        let activation_op = create_test_activation_operation();
        let activation_input = vec![-1.0f32, 0.0, 1.0, 2.0];
        let activation_output = runtime.execute_operation_with_weights(&activation_op, &activation_input, &weights)
            .expect("Failed to execute Activation operation");
        
        // Verify ReLU: max(0, x)
        let expected_activation = vec![0.0f32, 0.0, 1.0, 2.0];
        assert_eq!(activation_output, expected_activation, "Activation operation failed");
        println!("✅ Activation operation: ReLU({:?}) = {:?}", activation_input, activation_output);
    }

    #[test]
    fn test_convolution_with_weights() {
        println!("🧪 Testing convolution with weights...");
        
        let runtime = CoelanoxRuntime::new();
        
        // Create convolution operation
        let mut conv_op = create_test_convolution_operation();
        conv_op.attributes.insert("kernel_size".to_string(), coelanox::packager::ir::Attribute::Int(3));
        
        // Test data
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel_weights = vec![0.5f32, 1.0, 0.5]; // Kernel of size 3
        
        let conv_output = runtime.execute_convolution_with_weights(&conv_op, &input_data, &kernel_weights)
            .expect("Failed to execute convolution");
        
        // Verify convolution output size
        let expected_output_size = input_data.len() - 3 + 1; // input_len - kernel_size + 1
        assert_eq!(conv_output.len(), expected_output_size, "Convolution output size mismatch");
        
        // Verify convolution computation
        // For input [1,2,3,4,5,6,7,8] and kernel [0.5,1.0,0.5]:
        // Position 0: 1*0.5 + 2*1.0 + 3*0.5 = 0.5 + 2.0 + 1.5 = 4.0
        // Position 1: 2*0.5 + 3*1.0 + 4*0.5 = 1.0 + 3.0 + 2.0 = 6.0
        // etc.
        let expected_conv = vec![4.0f32, 6.0, 8.0, 10.0, 12.0, 14.0];
        assert_eq!(conv_output, expected_conv, "Convolution computation failed");
        println!("✅ Convolution: {:?} * {:?} = {:?}", input_data, kernel_weights, conv_output);
    }

    #[test]
    fn test_matmul_with_weights() {
        println!("🧪 Testing matrix multiplication with weights...");
        
        let runtime = CoelanoxRuntime::new();
        
        // Create matrix multiplication operation
        let mut matmul_op = create_test_matmul_operation();
        matmul_op.attributes.insert("dimensions".to_string(), 
            coelanox::packager::ir::Attribute::Ints(vec![2, 3])); // 2x3 matrix
        
        // Test data: 2x3 input matrix
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_data = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0]; // 2x3 weight matrix
        
        let matmul_output = runtime.execute_matmul_with_weights(&matmul_op, &input_data, &weight_data)
            .expect("Failed to execute matrix multiplication");
        
        // Verify matrix multiplication
        // For 2x3 matrices, output should be 2x1 (rows x 1)
        assert_eq!(matmul_output.len(), 2, "Matrix multiplication output size mismatch");
        
        // Verify computation:
        // Row 0: 1*0.5 + 2*1.0 + 3*1.5 = 0.5 + 2.0 + 4.5 = 7.0
        // Row 1: 4*2.0 + 5*2.5 + 6*3.0 = 8.0 + 12.5 + 18.0 = 38.5
        let expected_matmul = vec![7.0f32, 38.5];
        assert_eq!(matmul_output, expected_matmul, "Matrix multiplication computation failed");
        println!("✅ Matrix multiplication: {:?} * {:?} = {:?}", input_data, weight_data, matmul_output);
    }
}

// Helper functions for creating test data

fn create_test_onnx_model() -> Vec<u8> {
    // Create a simple ONNX model with weights
    let mut model = coelanox::onnx::ModelProto::default();
    
    // Set basic model info
    model.ir_version = 7;
    model.producer_name = "COELANOX Test".to_string();
    model.producer_version = "1.0".to_string();
    model.domain = "test".to_string();
    model.model_version = 1;
    
    // Create graph
    let mut graph = coelanox::onnx::GraphProto::default();
    graph.name = "test_graph".to_string();
    
    // Add input
    let mut input = coelanox::onnx::ValueInfoProto::default();
    input.name = "input".to_string();
    let mut input_type = coelanox::onnx::TypeProto::default();
    let mut tensor_type = coelanox::onnx::TypeProto_Tensor::default();
    tensor_type.elem_type = coelanox::onnx::TensorProto_DataType::FLOAT as i32;
    let mut shape = coelanox::onnx::TensorShapeProto::default();
    let mut dim = coelanox::onnx::TensorShapeProto_Dimension::default();
    dim.dim_value = 5;
    shape.dim.push(dim);
    tensor_type.shape = Some(shape);
    input_type.tensor_type = Some(tensor_type);
    input.r#type = Some(input_type);
    graph.input.push(input);
    
    // Add output
    let mut output = coelanox::onnx::ValueInfoProto::default();
    output.name = "output".to_string();
    let mut output_type = coelanox::onnx::TypeProto::default();
    let mut tensor_type = coelanox::onnx::TypeProto_Tensor::default();
    tensor_type.elem_type = coelanox::onnx::TensorProto_DataType::FLOAT as i32;
    let mut shape = coelanox::onnx::TensorShapeProto::default();
    let mut dim = coelanox::onnx::TensorShapeProto_Dimension::default();
    dim.dim_value = 3;
    shape.dim.push(dim);
    tensor_type.shape = Some(shape);
    output_type.tensor_type = Some(tensor_type);
    output.r#type = Some(output_type);
    graph.output.push(output);
    
    // Add weights (initializers)
    let mut weight1 = coelanox::onnx::TensorProto::default();
    weight1.name = "weight1".to_string();
    weight1.data_type = coelanox::onnx::TensorProto_DataType::FLOAT as i32;
    weight1.float_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    graph.initializer.push(weight1);
    
    let mut weight2 = coelanox::onnx::TensorProto::default();
    weight2.name = "weight2".to_string();
    weight2.data_type = coelanox::onnx::TensorProto_DataType::FLOAT as i32;
    weight2.float_data = vec![0.5, 1.0, 1.5];
    graph.initializer.push(weight2);
    
    // Add operation
    let mut add_node = coelanox::onnx::NodeProto::default();
    add_node.name = "add_node".to_string();
    add_node.op_type = "Add".to_string();
    add_node.input.push("input".to_string());
    add_node.input.push("weight1".to_string());
    add_node.output.push("output".to_string());
    graph.node.push(add_node);
    
    model.graph = Some(graph);
    
    // Serialize to bytes
    model.encode_to_vec()
}

fn create_test_container_with_weights(compressed_weights: Vec<u8>, compression_type: coelanox::format::CompressionType) -> coelanox::format::CoelanoxContainer {
    // Create a simple manifest
    let mut manifest = coelanox::format::CoelanoxManifest::default();
    manifest.model_name = "test_model".to_string();
    manifest.model_version = "1.0".to_string();
    manifest.input_shapes = vec![vec![5]];
    manifest.output_shapes = vec![vec![3]];
    
    // Create container
    coelanox::format::CoelanoxContainer::new(
        manifest,
        vec![0x90, 0x90, 0x90, 0x90], // Simple machine code
        compressed_weights,
        compression_type,
    )
}

fn create_test_add_operation() -> coelanox::packager::ir::Operation {
    let mut op = coelanox::packager::ir::Operation::new(coelanox::packager::ir::OpType::Add);
    op.inputs.push(coelanox::packager::ir::Tensor::new("input".to_string(), vec![3]));
    op.outputs.push(coelanox::packager::ir::Tensor::new("output".to_string(), vec![3]));
    op
}

fn create_test_multiply_operation() -> coelanox::packager::ir::Operation {
    let mut op = coelanox::packager::ir::Operation::new(coelanox::packager::ir::OpType::Multiply);
    op.inputs.push(coelanox::packager::ir::Tensor::new("input".to_string(), vec![3]));
    op.outputs.push(coelanox::packager::ir::Tensor::new("output".to_string(), vec![3]));
    op
}

fn create_test_activation_operation() -> coelanox::packager::ir::Operation {
    let mut op = coelanox::packager::ir::Operation::new(coelanox::packager::ir::OpType::Activation);
    op.inputs.push(coelanox::packager::ir::Tensor::new("input".to_string(), vec![4]));
    op.outputs.push(coelanox::packager::ir::Tensor::new("output".to_string(), vec![4]));
    op
}

fn create_test_convolution_operation() -> coelanox::packager::ir::Operation {
    let mut op = coelanox::packager::ir::Operation::new(coelanox::packager::ir::OpType::Convolution);
    op.inputs.push(coelanox::packager::ir::Tensor::new("input".to_string(), vec![8]));
    op.outputs.push(coelanox::packager::ir::Tensor::new("output".to_string(), vec![6]));
    op
}

fn create_test_matmul_operation() -> coelanox::packager::ir::Operation {
    let mut op = coelanox::packager::ir::Operation::new(coelanox::packager::ir::OpType::MatMul);
    op.inputs.push(coelanox::packager::ir::Tensor::new("input".to_string(), vec![2, 3]));
    op.outputs.push(coelanox::packager::ir::Tensor::new("output".to_string(), vec![2]));
    op
}
