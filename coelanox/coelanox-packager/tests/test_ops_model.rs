//! Test model that exercises each operation once for debugging
//! 
//! This creates a simple model with Add and Subtract operations
//! to test the full pipeline: IR → Package → Execute

use coelanox_core::ir::{UniversalIR, OpType, Shape, DataType};
use coelanox_packager::CoelanoxPackager;
use coelanox_core::format::{HardwareTarget, CoelanoxContainer, CoelanoxManifest, CompressionType};
use std::path::PathBuf;

/// Helper function to package a UniversalIR directly (without going through file I/O)
fn package_ir(packager: &CoelanoxPackager, ir: &UniversalIR, target: HardwareTarget) -> Result<CoelanoxContainer, coelanox_core::CoelanoxError> {
    use coelanox_packager::optimizers::HardwareOptimizer;
    use coelanox_packager::generators::CodeGenerator;
    
    // Step 1: Optimize IR
    let optimizer = match target {
        HardwareTarget::Cpu => coelanox_packager::optimizers::CpuOptimizer::new(),
        _ => return Err(coelanox_core::CoelanoxError::Runtime("Only CPU target supported for now".to_string())),
    };
    
    let optimized_ir = HardwareOptimizer::optimize(&optimizer, ir, target.clone())?;
    
    // Step 2: Generate machine code
    let generator = match target {
        HardwareTarget::Cpu => {
            Box::new(coelanox_packager::generators::cpu_codegen::CpuCodeGenerator::new()) as Box<dyn CodeGenerator>
        },
        _ => return Err(coelanox_core::CoelanoxError::Runtime("Only CPU target supported for now".to_string())),
    };
    
    let machine_code = generator.generate(&optimized_ir)?;
    
    // Step 3: Extract weights (empty for our test models)
    let weights: Vec<f32> = Vec::new();
    let compressed_weights = if weights.is_empty() {
        Vec::new()
    } else {
        // Simple compression: just convert to bytes
        let mut weight_bytes = Vec::new();
        for &weight in &weights {
            weight_bytes.extend_from_slice(&weight.to_le_bytes());
        }
        weight_bytes
    };
    
    // Step 4: Create manifest
    let manifest = CoelanoxManifest::new(
        ir.metadata.name.clone(),
        coelanox_core::format::ModelFormat::Custom("test".to_string()),
        ir.metadata.input_shapes.iter().map(|s| s.dimensions.clone()).collect(),
        ir.metadata.output_shapes.iter().map(|s| s.dimensions.clone()).collect(),
        machine_code.len(),
    );
    
    // Step 5: Serialize IR
    use coelanox_core::ir::{GraphNode, IntermediateTensor, WeightTensor};
    use serde::{Serialize, Deserialize};
    use std::collections::HashMap;
    
    #[derive(Serialize, Deserialize)]
    struct SerializableIR {
        target: HardwareTarget,
        optimized_nodes: Vec<GraphNode>,
        intermediate_tensors: Vec<IntermediateTensor>,
        optimization_metadata: HashMap<String, String>,
        original_model_name: String,
        weight_tensors: Vec<WeightTensor>,
    }
    
    // Convert shared_weights (Vec<Arc<WeightTensor>>) to Vec<WeightTensor>
    let weight_tensors: Vec<WeightTensor> = optimized_ir.shared_weights.iter().map(|wt| (*wt).clone()).collect();
    
    let serializable_ir = SerializableIR {
        target: optimized_ir.target,
        optimized_nodes: optimized_ir.optimized_nodes.clone(),
        intermediate_tensors: optimized_ir.intermediate_tensors.clone(),
        optimization_metadata: optimized_ir.optimization_metadata.clone(),
        original_model_name: optimized_ir.original_model_name.clone(),
        weight_tensors,
    };
    
    let ir_data = bincode::serialize(&serializable_ir)
        .map_err(|e| coelanox_core::CoelanoxError::Runtime(format!("Failed to serialize IR: {}", e)))?;
    
    // Step 6: Create container
    Ok(CoelanoxContainer::new(
        manifest,
        machine_code,
        compressed_weights,
        CompressionType::None,
        ir_data,
    ))
}

/// Create a simple test model with Add operation only
fn create_add_model() -> UniversalIR {
    let mut ir = UniversalIR::new("test_add_only".to_string());
    
    // Set input/output shapes
    ir.metadata.input_shapes.push(Shape::new(vec![4]));
    ir.metadata.input_shapes.push(Shape::new(vec![4]));
    ir.metadata.output_shapes.push(Shape::new(vec![4]));
    
    // Create input tensors
    let input_a = ir.add_intermediate_tensor(
        "input_a".to_string(),
        Shape::new(vec![4]),
        DataType::Float32,
    );
    
    let input_b = ir.add_intermediate_tensor(
        "input_b".to_string(),
        Shape::new(vec![4]),
        DataType::Float32,
    );
    
    // Create output tensor
    let output = ir.add_intermediate_tensor(
        "output".to_string(),
        Shape::new(vec![4]),
        DataType::Float32,
    );
    
    // Add Add operation
    let add_node = ir.add_node(OpType::Add);
    ir.add_node_input(add_node, input_a).unwrap();
    ir.add_node_input(add_node, input_b).unwrap();
    ir.add_node_output(add_node, output).unwrap();
    
    // Validate
    ir.validate().expect("IR validation failed");
    ir
}

#[test]
fn test_add_operation_end_to_end() {
    println!("=== Creating Add-only test model ===");
    let ir = create_add_model();
    
    println!("=== Packaging Add-only model ===");
    let packager = CoelanoxPackager::new().expect("Failed to create packager");
    let output_path = PathBuf::from("/tmp/test_add_only.cnox");
    
    let container = package_ir(&packager, &ir, HardwareTarget::Cpu)
        .expect("Packaging failed");
    container.write_binary_file(&output_path).expect("Failed to write container");
    println!("✅ Container saved to {:?}", output_path);
    
    println!("=== Model packaged successfully! ===");
    println!("You can now load and execute this container with:");
    println!("  coelanox-cli execute /tmp/test_add_only.cnox");
    println!("");
    println!("Or manually test with the runtime:");
    println!("  use coelanox_runtime::{{CoelanoxRuntime, CoelanoxRuntimeConfig}};");
    println!("  let mut runtime = CoelanoxRuntime::new(CoelanoxRuntimeConfig::default());");
    println!("  runtime.load_container(&output_path)?;");
    println!("  let output = runtime.execute(\"test_add_only\", &input_data)?;");
}

#[test]
fn test_add_operation_load_and_execute() {
    println!("=== Creating Add-only test model ===");
    let ir = create_add_model();
    
    println!("=== Packaging Add-only model ===");
    let packager = CoelanoxPackager::new().expect("Failed to create packager");
    let output_path = PathBuf::from("/tmp/test_add_only_exec.cnox");
    
    let container = package_ir(&packager, &ir, HardwareTarget::Cpu)
        .expect("Packaging failed");
    
    // Debug: Check manifest output shapes
    println!("=== Manifest Debug ===");
    println!("Output shapes: {:?}", container.manifest.output_shapes);
    println!("Output size calc: {}", container.manifest.output_shapes.iter().map(|s| s.iter().product::<usize>()).sum::<usize>());
    println!("Input shapes: {:?}", container.manifest.input_shapes);
    println!("Machine code size: {} bytes", container.optimized_code.len());
    
    container.write_binary_file(&output_path).expect("Failed to write container");
    println!("✅ Container saved to {:?}", output_path);
    
    println!("=== Loading and executing model ===");
    use coelanox_runtime::CoelanoxRuntime;
    
    let runtime = CoelanoxRuntime::new();
    
    // Load the container (takes &str, not PathBuf)
    let container_path = output_path.to_str().expect("Invalid path");
    match runtime.load_container(container_path) {
        Ok(_) => println!("✅ Container loaded successfully"),
        Err(e) => panic!("❌ Failed to load container: {:?}", e),
    }
    
    // Prepare input data
    // input_a = [1.0, 2.0, 3.0, 4.0]
    // input_b = [5.0, 6.0, 7.0, 8.0]
    // Expected output: [6.0, 8.0, 10.0, 12.0]
    let mut input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    input_data.extend_from_slice(&[5.0f32, 6.0f32, 7.0f32, 8.0f32]);
    
    println!("Input data: {:?}", input_data);
    
    // Execute the model (container_name is the path string)
    match runtime.execute(container_path, &input_data) {
        Ok(output) => {
            println!("✅ Execution successful!");
            println!("Output: {:?}", output);
            println!("Output len: {}", output.len());
            
            // Check if output is all zeros (machine code might not be writing)
            let all_zeros = output.iter().all(|&x| x == 0.0);
            if all_zeros {
                println!("⚠️  WARNING: Output is all zeros - machine code might not be writing correctly!");
            }
            
            // Verify correctness
            let expected = vec![6.0f32, 8.0f32, 10.0f32, 12.0f32];
            assert_eq!(output.len(), expected.len(), "Output size mismatch");
            
            for (i, (actual, expected_val)) in output.iter().zip(expected.iter()).enumerate() {
                let diff = (actual - expected_val).abs();
                assert!(diff < 0.0001, 
                    "Mismatch at index {}: expected {}, got {} (diff: {})", 
                    i, expected_val, actual, diff);
            }
            
            println!("✅ All outputs match expected values!");
        },
        Err(e) => {
            eprintln!("❌ Execution failed: {:?}", e);
            panic!("Execution failed: {:?}", e);
        }
    }
}

