//! Simple test for ADD operation to debug segfault

use coelanox_core::ir::{UniversalIR, OpType, Shape, DataType, NodeId, GraphNode, OptimizedIR, TensorId};
use coelanox_core::format::HardwareTarget;
use coelanox_packager::generators::cpu_codegen::CpuCodeGenerator;
use coelanox_packager::generators::CodeGenerator;
use std::alloc::{alloc, Layout};
use std::ptr;

fn make_executable(code: &[u8]) -> Result<*mut u8, String> {
    let page_size = 4096;
    let aligned_size = ((code.len() + page_size - 1) / page_size) * page_size;
    
    let layout = Layout::from_size_align(aligned_size, page_size)
        .map_err(|e| format!("Invalid layout: {}", e))?;
    
    unsafe {
        let ptr = alloc(layout) as *mut u8;
        if ptr.is_null() {
            return Err("Failed to allocate memory".to_string());
        }
        
        ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        
        #[cfg(target_os = "linux")]
        {
            use libc::{mprotect, PROT_READ, PROT_EXEC};
            let result = mprotect(
                ptr as *mut libc::c_void,
                aligned_size,
                PROT_READ | PROT_EXEC
            );
            if result != 0 {
                return Err("Failed to make memory executable".to_string());
            }
        }
        
        Ok(ptr)
    }
}

#[test]
fn test_add_simple() {
    println!("🧪 Testing ADD operation (simple)...");
    
    // Simple test: 4 elements
    // input_a = [1.0, 2.0, 3.0, 4.0]
    // input_b = [1.0, 1.0, 1.0, 1.0]
    // expected = [2.0, 3.0, 4.0, 5.0]
    let input_a = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let input_b = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32];
    let combined_input = [input_a.as_slice(), input_b.as_slice()].concat();
    let expected = vec![2.0f32, 3.0f32, 4.0f32, 5.0f32];
    
    // Create IR
    let mut ir = UniversalIR::new("test_add_simple".to_string());
    ir.metadata.input_shapes.push(Shape::new(vec![4]));
    ir.metadata.input_shapes.push(Shape::new(vec![4]));
    ir.metadata.output_shapes.push(Shape::new(vec![4]));
    
    let mut optimized_ir = OptimizedIR::new(&ir, HardwareTarget::Cpu);
    let node = GraphNode::new(NodeId(0), OpType::Add);
    // For binary ADD, we need two input tensors
    // In the test, both inputs come from the same input buffer (concatenated)
    // So we'll just create a simple node
    optimized_ir.optimized_nodes.push(node);
    
    // Generate code
    let generator = CpuCodeGenerator::new();
    let code = match generator.generate(&optimized_ir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Code generation failed: {:?}", e);
            return;
        }
    };
    
    println!("✅ Generated {} bytes of code", code.len());
    
    // Execute
    let ptr = match make_executable(&code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ Failed to make executable: {}", e);
            return;
        }
    };
    
    let mut output = vec![0.0f32; 4];
    let weights: Vec<f32> = Vec::new();
    
    type CpuFunc = unsafe extern "sysv64" fn(
        *const f32, usize, // rdi, rsi
        *mut f32, usize,   // rdx, rcx
        *const f32, usize  // r8,  r9
    );
    
    let func: CpuFunc = unsafe { std::mem::transmute(ptr as usize) };
    
    println!("Calling generated function...");
    unsafe {
        func(
            combined_input.as_ptr(), combined_input.len(),
            output.as_mut_ptr(), output.len(),
            weights.as_ptr(), weights.len(),
        );
    }
    
    println!("Input: {:?}", combined_input);
    println!("Output: {:?}", output);
    println!("Expected: {:?}", expected);
    
    // Verify
    let tolerance = 0.001f32;
    for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - exp).abs();
        if diff > tolerance {
            panic!("Element {}: expected {}, got {}, diff: {}", i, exp, actual, diff);
        }
    }
    
    println!("✅ Test passed!");
}

