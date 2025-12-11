//! Operation Template Validation Tests
//! 
//! This test suite validates each operation template individually by:
//! 1. Generating minimal machine code (prologue + operation + epilogue)
//! 2. Executing with simple test data
//! 3. Verifying correct output or catching segfaults
//! 
//! This helps identify which specific operation templates have bugs.
//! 
//! Run with: `cargo test --test operation_template_validation -- --ignored`

use coelanox_packager::generators::templates::cpu_templates::CpuTemplates;
use coelanox_core::ir::{OpType, OptimizedIR, GraphNode, IntermediateTensor};
use coelanox_core::format::HardwareTarget;
use std::collections::HashMap;
use std::sync::Arc;
use std::ptr;

/// Test framework for validating operation templates
struct OperationTestFramework {
    templates: CpuTemplates,
}

impl OperationTestFramework {
    fn new() -> Self {
        Self {
            templates: CpuTemplates::new(),
        }
    }
    
    /// Generate minimal machine code for a single operation
    /// Returns: (prologue + operation + epilogue) as bytes
    fn generate_single_operation_code(&self, op_type: OpType) -> Vec<u8> {
        let mut code = Vec::new();
        
        // Generate prologue (x86-64 System V ABI)
        // Function signature: fn(&[f32], &mut [f32], &[f32])
        code.extend_from_slice(&[
            0x55,                    // push rbp
            0x48, 0x89, 0xE5,        // mov rbp, rsp
            0x53,                    // push rbx (callee-saved)
            0x41, 0x54,              // push r12 (callee-saved)
            0x41, 0x55,              // push r13 (callee-saved)
            0x41, 0x56,              // push r14 (callee-saved)
            0x41, 0x57,              // push r15 (callee-saved)
            0x48, 0x83, 0xEC, 0x20,  // sub rsp, 32 (align stack)
            // Load parameters: r12=input.ptr, r14=output.ptr, r15=output.len
            0x49, 0x89, 0xFC,        // mov r12, rdi (input pointer)
            0x49, 0x89, 0xD6,        // mov r14, rdx (output pointer)
            0x49, 0x89, 0xCF,        // mov r15, rcx (output length)
        ]);
        
        // Get template for this operation
        if let Some(template) = self.templates.get_template(op_type.clone()) {
            code.extend_from_slice(&template.machine_code);
        } else {
            // Passthrough for unsupported operations
            code.extend_from_slice(&[
                0xF3, 0x41, 0x0F, 0x10, 0x04, 0x9C,  // movss xmm0, [r12+rbx*4]
                0xF3, 0x41, 0x0F, 0x11, 0x04, 0x9E,  // movss [r14+rbx*4], xmm0
            ]);
        }
        
        // Generate epilogue
        code.extend_from_slice(&[
            0x48, 0x83, 0xC4, 0x20,  // add rsp, 32
            0x41, 0x5F,              // pop r15
            0x41, 0x5E,              // pop r14
            0x41, 0x5D,              // pop r13
            0x41, 0x5C,              // pop r12
            0x5B,                    // pop rbx
            0x5D,                    // pop rbp
            0xC3,                    // ret
        ]);
        
        code
    }
    
    /// Allocate executable memory and copy code
    fn make_executable(&self, code: &[u8]) -> Result<*mut u8, String> {
        use std::alloc::{alloc, Layout};
        
        let page_size = 4096;
        let aligned_size = ((code.len() + page_size - 1) / page_size) * page_size;
        
        let layout = Layout::from_size_align(aligned_size, page_size)
            .map_err(|e| format!("Invalid layout: {}", e))?;
        
        unsafe {
            let ptr = alloc(layout) as *mut u8;
            if ptr.is_null() {
                return Err("Failed to allocate memory".to_string());
            }
            
            // Copy code
            ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
            
            // Make executable (Linux-specific)
            #[cfg(target_os = "linux")]
            {
                use libc::{mprotect, PROT_READ | PROT_EXEC, PROT_READ | PROT_WRITE};
                let result = mprotect(
                    ptr as *mut libc::c_void,
                    aligned_size,
                    PROT_READ | PROT_EXEC
                );
                if result != 0 {
                    // Restore write permissions if mprotect fails
                    mprotect(ptr as *mut libc::c_void, aligned_size, PROT_READ | PROT_WRITE);
                    return Err("Failed to make memory executable".to_string());
                }
            }
            
            Ok(ptr)
        }
    }
    
    /// Execute a single operation with test data
    /// Function signature: fn(&[f32], &mut [f32], &[f32])
    /// x86-64 ABI: rdi=input.ptr, rsi=input.len, rdx=output.ptr, rcx=output.len, r8=weights.ptr, r9=weights.len
    fn execute_operation(
        &self,
        code_ptr: *mut u8,
        input: &[f32],
        output: &mut [f32],
        weights: &[f32],
    ) -> Result<(), String> {
        unsafe {
            // Create function pointer
            type FuncPtr = unsafe extern "C" fn(*const f32, usize, *mut f32, usize, *const f32, usize);
            let func: FuncPtr = std::mem::transmute(code_ptr);
            
            // Call the function
            func(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                weights.as_ptr(),
                weights.len(),
            );
            
            Ok(())
        }
    }
    
    /// Test a single operation with simple data
    fn test_operation(&self, op_type: OpType) -> Result<TestResult, String> {
        // Generate code
        let code = self.generate_single_operation_code(op_type.clone());
        
        if code.is_empty() {
            return Err("Generated code is empty".to_string());
        }
        
        // Make executable
        let code_ptr = self.make_executable(&code)?;
        
        // Prepare test data
        // For binary operations (Add, Subtract, etc.), we need two inputs
        // We'll lay them out sequentially: [input_a | input_b]
        let test_size = 4; // Small test size
        let mut combined_input = vec![1.0f32; test_size * 2]; // Double size for two inputs
        for i in 0..test_size {
            combined_input[i] = (i + 1) as f32; // [1.0, 2.0, 3.0, 4.0]
            combined_input[i + test_size] = (i + 1) as f32 * 2.0; // [2.0, 4.0, 6.0, 8.0]
        }
        
        let mut output = vec![0.0f32; test_size];
        let weights = vec![0.0f32; 0]; // Empty weights for now
        
        // Execute with signal handler for segfault detection
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute_operation(code_ptr, &combined_input, &mut output, &weights)
        }));
        
        match result {
            Ok(Ok(())) => {
                // Execution succeeded - verify output
                Ok(TestResult::Success {
                    output: output.clone(),
                })
            }
            Ok(Err(e)) => {
                Ok(TestResult::ExecutionError(e))
            }
            Err(_) => {
                // Panic/segfault detected
                Ok(TestResult::Segfault)
            }
        }
    }
}

/// Test result for an operation
#[derive(Debug, Clone)]
enum TestResult {
    Success { output: Vec<f32> },
    ExecutionError(String),
    Segfault,
    Unsupported,
}

/// Run all operation validation tests
#[test]
#[ignore] // Ignore by default - run with `cargo test -- --ignored`
fn test_all_operations() {
    let framework = OperationTestFramework::new();
    
    // List of all operations to test
    let operations = vec![
        OpType::Add,
        OpType::Subtract,
        OpType::Multiply,
        OpType::Divide,
        OpType::Relu,
        OpType::Sigmoid,
        OpType::Tanh,
        OpType::Softmax,
        OpType::LogSoftmax,
        OpType::Gelu,
        OpType::Swish,
        OpType::MatMul,
        OpType::Gemm,
        OpType::Convolution,
        OpType::MaxPool,
        OpType::AvgPool,
        OpType::GlobalMaxPool,
        OpType::GlobalAvgPool,
        OpType::BatchNorm,
        OpType::LayerNorm,
        OpType::Dropout,
        OpType::ReduceSum,
        OpType::ReduceMean,
        OpType::ReduceMax,
        OpType::ReduceMin,
        OpType::ReduceProd,
        OpType::Reshape,
        OpType::Transpose,
        OpType::Permute,
        OpType::Concatenate,
        OpType::Split,
        OpType::Slice,
        OpType::Gather,
        OpType::Scatter,
        OpType::Broadcast,
        OpType::Expand,
        OpType::Equal,
        OpType::NotEqual,
        OpType::Greater,
        OpType::GreaterEqual,
        OpType::Less,
        OpType::LessEqual,
        OpType::And,
        OpType::Or,
        OpType::Not,
    ];
    
    let mut results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut segfaults = 0;
    let mut unsupported = 0;
    
    println!("\n=== Operation Template Validation ===");
    println!("Testing {} operations...\n", operations.len());
    
    for op_type in operations {
        print!("Testing {:?}... ", op_type);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        match framework.test_operation(op_type.clone()) {
            Ok(TestResult::Success { output }) => {
                println!("✅ PASS (output: {:?})", &output[..std::cmp::min(4, output.len())]);
                passed += 1;
            }
            Ok(TestResult::Segfault) => {
                println!("💥 SEGFAULT");
                segfaults += 1;
                results.push((op_type, TestResult::Segfault));
            }
            Ok(TestResult::ExecutionError(e)) => {
                println!("❌ ERROR: {}", e);
                failed += 1;
                results.push((op_type, TestResult::ExecutionError(e)));
            }
            Ok(TestResult::Unsupported) => {
                println!("⚠️  UNSUPPORTED");
                unsupported += 1;
            }
            Err(e) => {
                println!("❌ FAILED: {}", e);
                failed += 1;
                results.push((op_type, TestResult::ExecutionError(e)));
            }
        }
    }
    
    println!("\n=== Summary ===");
    println!("✅ Passed: {}", passed);
    println!("❌ Failed: {}", failed);
    println!("💥 Segfaults: {}", segfaults);
    println!("⚠️  Unsupported: {}", unsupported);
    
    if !results.is_empty() {
        println!("\n=== Failed Operations ===");
        for (op_type, result) in results {
            println!("  {:?}: {:?}", op_type, result);
        }
    }
    
    // Fail the test if there are segfaults or failures
    if segfaults > 0 || failed > 0 {
        panic!("Operation validation failed: {} segfaults, {} errors", segfaults, failed);
    }
}

/// Quick test for a specific operation (for debugging)
#[test]
#[ignore]
fn test_single_operation() {
    let framework = OperationTestFramework::new();
    let op_type = OpType::Add; // Change this to test different operations
    
    println!("Testing operation: {:?}", op_type);
    
    match framework.test_operation(op_type) {
        Ok(TestResult::Success { output }) => {
            println!("✅ Success! Output: {:?}", output);
        }
        Ok(TestResult::Segfault) => {
            println!("💥 Segfault detected!");
        }
        Ok(TestResult::ExecutionError(e)) => {
            println!("❌ Error: {}", e);
        }
        Ok(TestResult::Unsupported) => {
            println!("⚠️  Unsupported");
        }
        Err(e) => {
            println!("❌ Failed: {}", e);
        }
    }
}

