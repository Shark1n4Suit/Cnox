//! Test runner for all 46 operations
//! 
//! This test verifies that all operations are registered and can be tested
//! (even if they return errors or are stubs)

use super::test_helpers::OperationTest;
use coelanox_core::ir::OpType;
use coelanox_packager::generators::CodeGenerator;

/// Test that verifies all operations are registered
#[test]
fn test_all_operations_registered() {
    let operations = vec![
        // Basic arithmetic (4)
        OpType::Add,
        OpType::Subtract,
        OpType::Multiply,
        OpType::Divide,
        
        // Element-wise (7)
        OpType::Relu,
        OpType::Sigmoid,
        OpType::Tanh,
        OpType::Softmax,
        OpType::LogSoftmax,
        OpType::Gelu,
        OpType::Swish,
        
        // Neural network (8)
        OpType::Convolution,
        OpType::MaxPool,
        OpType::AvgPool,
        OpType::GlobalMaxPool,
        OpType::GlobalAvgPool,
        OpType::BatchNorm,
        OpType::LayerNorm,
        OpType::Dropout,
        
        // Tensor manipulation (8)
        OpType::Reshape,
        OpType::Transpose,
        OpType::Permute,
        OpType::Concatenate,
        OpType::Split,
        OpType::Slice,
        OpType::Gather,
        OpType::Scatter,
        
        // Matrix (2)
        OpType::MatMul,
        OpType::Gemm,
        
        // Reduction (5)
        OpType::ReduceSum,
        OpType::ReduceMean,
        OpType::ReduceMax,
        OpType::ReduceMin,
        OpType::ReduceProd,
        
        // Broadcasting (2)
        OpType::Broadcast,
        OpType::Expand,
        
        // Comparison (6)
        OpType::Equal,
        OpType::NotEqual,
        OpType::Greater,
        OpType::GreaterEqual,
        OpType::Less,
        OpType::LessEqual,
        
        // Logical (3)
        OpType::And,
        OpType::Or,
        OpType::Not,
    ];
    
    println!("Testing {} operations are registered...", operations.len());
    
    let mut registered = 0;
    let mut working = 0;
    let mut errors = 0;
    let mut stubs = 0;
    
    let total_ops = operations.len();
    
    for op_type in &operations {
        let generator = coelanox_packager::generators::cpu_codegen::CpuCodeGenerator::new();
        let node = coelanox_core::ir::GraphNode::new(
            coelanox_core::ir::NodeId(0),
            (*op_type).clone()
        );
        
        // Try to generate code - if it works, the operation is registered
        let mut ir = coelanox_core::ir::UniversalIR::new(format!("test_{:?}", op_type));
        ir.metadata.input_shapes.push(coelanox_core::ir::Shape::new(vec![4]));
        ir.metadata.output_shapes.push(coelanox_core::ir::Shape::new(vec![4]));
        
        let mut optimized_ir = coelanox_core::ir::OptimizedIR::new(&ir, coelanox_core::format::HardwareTarget::Cpu);
        optimized_ir.optimized_nodes.push(node);
        
        match generator.generate(&optimized_ir) {
            Ok(code) => {
                registered += 1;
                if code.is_empty() {
                    stubs += 1;
                    println!("  {:?}: registered (stub)", op_type);
                } else {
                    working += 1;
                    println!("  {:?}: ✅ working ({} bytes)", op_type, code.len());
                }
            }
            Err(e) => {
                // Check if it's a "not implemented" error or a real error
                let err_str = format!("{:?}", e);
                if err_str.contains("not yet implemented") || err_str.contains("not supported") || 
                   err_str.contains("requires") || err_str.contains("not implemented") {
                    registered += 1;
                    errors += 1;
                    println!("  {:?}: ⚠️  registered but not implemented - {}", op_type, e);
                } else {
                    println!("  {:?}: ❌ NOT REGISTERED - {}", op_type, e);
                }
            }
        }
    }
    
    println!("\nSummary:");
    println!("  Total operations: {}", total_ops);
    println!("  Registered: {}", registered);
    println!("  Working: {}", working);
    println!("  Errors: {}", errors);
    println!("  Stubs: {}", stubs);
    
    assert_eq!(registered, total_ops, "All operations should be registered");
}

