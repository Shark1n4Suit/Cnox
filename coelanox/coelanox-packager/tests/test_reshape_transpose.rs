//! Test Reshape and Transpose operations individually

use coelanox_core::ir::{UniversalIR, OpType, Shape, NodeId, GraphNode, OptimizedIR};
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

type CpuFunc = unsafe extern "sysv64" fn(
    *const f32, usize, // rdi, rsi
    *mut f32, usize,   // rdx, rcx
    *const f32, usize  // r8,  r9
);

fn test_operation(op_type: OpType, input: &[f32], output_len: usize, description: &str) -> bool {
    println!("\n=== Testing {} ===", description);
    
    // Create IR
    let mut ir = UniversalIR::new(format!("test_{:?}", op_type));
    ir.metadata.input_shapes.push(Shape::new(vec![input.len()]));
    ir.metadata.output_shapes.push(Shape::new(vec![output_len]));
    
    let mut optimized_ir = OptimizedIR::new(&ir, HardwareTarget::Cpu);
    let node = GraphNode::new(NodeId(0), op_type.clone());
    optimized_ir.optimized_nodes.push(node);
    
    // Generate code
    let generator = CpuCodeGenerator::new();
    let code = match generator.generate(&optimized_ir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Code generation failed: {:?}", e);
            return false;
        }
    };
    
    if code.is_empty() {
        eprintln!("❌ Generated code is empty");
        return false;
    }
    
    println!("✅ Generated {} bytes of code", code.len());
    
    // Make executable
    let ptr = match make_executable(&code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ Failed to make executable: {}", e);
            return false;
        }
    };
    
    let mut output = vec![0.0f32; output_len];
    let weights: Vec<f32> = Vec::new();
    
    let func: CpuFunc = unsafe { std::mem::transmute(ptr as usize) };
    
    // Execute with panic catching
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe {
            func(
                input.as_ptr(), input.len(),
                output.as_mut_ptr(), output.len(),
                weights.as_ptr(), weights.len(),
            );
        }
        output
    }));
    
    match result {
        Ok(output_result) => {
            println!("Input: {:?}", input);
            println!("Output: {:?}", output_result);
            
            // Check for non-zero output (basic sanity check)
            let all_zeros = output_result.iter().all(|&x| x == 0.0);
            if all_zeros {
                eprintln!("⚠️  Output is all zeros");
                return false;
            }
            
            // Check for NaN/Inf
            let has_nan_inf = output_result.iter().any(|&x| x.is_nan() || x.is_infinite());
            if has_nan_inf {
                eprintln!("⚠️  Output contains NaN or Inf");
                return false;
            }
            
            // For Reshape/Transpose, output should match input values (just reordered)
            let input_sorted: Vec<f32> = {
                let mut v = input.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            let output_sorted: Vec<f32> = {
                let mut v = output_result.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            
            if input_sorted != output_sorted {
                eprintln!("⚠️  Output values don't match input values (after sorting)");
                eprintln!("   Input sorted: {:?}", input_sorted);
                eprintln!("   Output sorted: {:?}", output_sorted);
                return false;
            }
            
            println!("✅ {} executed successfully!", description);
            true
        }
        Err(_) => {
            eprintln!("💥 SEGFAULT during execution");
            false
        }
    }
}

#[test]
fn test_reshape() {
    println!("🧪 Testing Reshape Operation\n");
    
    // Test 1: 1D to 1D (no change, just copy)
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    assert!(test_operation(OpType::Reshape, &input, 4, "Reshape (1D→1D)"));
    
    // Test 2: Larger input
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32];
    assert!(test_operation(OpType::Reshape, &input, 8, "Reshape (1D→1D, larger)"));
}

#[test]
fn test_transpose() {
    println!("🧪 Testing Transpose Operation\n");
    
    // Test 1: 1D transpose (should be no-op)
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    assert!(test_operation(OpType::Transpose, &input, 4, "Transpose (1D, no-op)"));
    
    // Test 2: Larger input
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32];
    assert!(test_operation(OpType::Transpose, &input, 8, "Transpose (1D, larger)"));
}

#[test]
fn test_reshape_and_transpose() {
    println!("🧪 Testing Reshape and Transpose Operations\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Test Reshape
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    if test_operation(OpType::Reshape, &input, 4, "Reshape") {
        passed += 1;
    } else {
        failed += 1;
    }
    
    // Test Transpose
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    if test_operation(OpType::Transpose, &input, 4, "Transpose") {
        passed += 1;
    } else {
        failed += 1;
    }
    
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              RESHAPE/TRANSPOSE TEST SUMMARY               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("✅ Passed: {}", passed);
    println!("❌ Failed: {}", failed);
    println!("Total: {}", passed + failed);
    
    if failed > 0 {
        panic!("Some operations failed!");
    }
}

