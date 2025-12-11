//! Shared test helpers for operation tests

use coelanox_core::ir::{UniversalIR, OpType, Shape, NodeId, GraphNode, OptimizedIR};
use coelanox_core::format::HardwareTarget;
use coelanox_packager::generators::cpu_codegen::CpuCodeGenerator;
use coelanox_packager::generators::CodeGenerator;
use std::alloc::{alloc, Layout};
use std::ptr;

/// Make code executable (allocate and set execute permissions)
pub fn make_executable(code: &[u8]) -> Result<*mut u8, String> {
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

/// Function signature for generated CPU code
/// System V ABI: rdi=input.ptr, rsi=input.len, rdx=output.ptr, rcx=output.len, r8=weights.ptr, r9=weights.len
pub type CpuFunc = unsafe extern "sysv64" fn(
    *const f32, usize, // rdi, rsi - input buffer
    *mut f32, usize,   // rdx, rcx - output buffer
    *const f32, usize  // r8,  r9  - weights buffer
);

/// Test configuration for an operation
pub struct OperationTest {
    pub op_type: OpType,
    pub input_a: Vec<f32>,
    pub input_b: Option<Vec<f32>>, // None for unary operations
    pub expected_output: Vec<f32>,
    pub tolerance: f32,
}

impl OperationTest {
    /// Create a binary operation test (two inputs)
    pub fn binary(op_type: OpType, input_a: Vec<f32>, input_b: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            op_type,
            input_a,
            input_b: Some(input_b),
            expected_output: expected,
            tolerance: 0.001f32,
        }
    }
    
    /// Create a unary operation test (one input)
    pub fn unary(op_type: OpType, input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            op_type,
            input_a: input,
            input_b: None,
            expected_output: expected,
            tolerance: 0.001f32,
        }
    }
    
    /// Run the test
    pub fn run(&self) -> Result<(), String> {
        println!("🧪 Testing {:?} operation...", self.op_type);
        
        // Prepare input buffer
        let combined_input = if let Some(ref input_b) = self.input_b {
            // Binary operation: concatenate input_a and input_b
            [self.input_a.as_slice(), input_b.as_slice()].concat()
        } else {
            // Unary operation: just input_a
            self.input_a.clone()
        };
        
        // Create IR
        let mut ir = UniversalIR::new(format!("test_{:?}", self.op_type));
        ir.metadata.input_shapes.push(Shape::new(vec![self.input_a.len()]));
        if self.input_b.is_some() {
            ir.metadata.input_shapes.push(Shape::new(vec![self.input_b.as_ref().unwrap().len()]));
        }
        ir.metadata.output_shapes.push(Shape::new(vec![self.expected_output.len()]));
        
        let mut optimized_ir = OptimizedIR::new(&ir, HardwareTarget::Cpu);
        let node = GraphNode::new(NodeId(0), self.op_type.clone());
        optimized_ir.optimized_nodes.push(node);
        
        // Generate code
        let generator = CpuCodeGenerator::new();
        let code = generator.generate(&optimized_ir)
            .map_err(|e| format!("Code generation failed: {:?}", e))?;
        
        if code.is_empty() {
            return Err("Generated code is empty".to_string());
        }
        
        println!("✅ Generated {} bytes of code", code.len());
        
        // Make executable
        let ptr = make_executable(&code)
            .map_err(|e| format!("Failed to make executable: {}", e))?;
        
        // Prepare output buffer
        let mut output = vec![0.0f32; self.expected_output.len()];
        let weights: Vec<f32> = Vec::new();
        
        // Execute
        // Note: For binary operations, the code uses r15 (output_len) to calculate second input offset
        // So we pass output.len() as both rsi (input_len) and rcx (output_len)
        // The actual combined input length is passed as rsi, but the code uses r15 (from rcx)
        let func: CpuFunc = unsafe { std::mem::transmute(ptr as usize) };
        
        println!("Calling generated function...");
        unsafe {
            // For reductions, input length != output length
            // For binary ops, input length = combined_input.len() but loop uses output_len
            // For unary ops, input length = input_a.len()
            let input_len = if self.input_b.is_some() {
                // Binary operation: check if it's MatMul/Gemm (special case)
                match self.op_type {
                    coelanox_core::ir::OpType::MatMul | coelanox_core::ir::OpType::Gemm => {
                        // MatMul/Gemm: loop over input_a length, output is scalar
                        self.input_a.len()
                    }
                    _ => {
                        // Other binary ops: pass combined length, but code uses output_len for loop
                        self.expected_output.len() // For binary ops, loop uses output_len
                    }
                }
            } else {
                // Unary operation: check if it's a reduction
                match self.op_type {
                    coelanox_core::ir::OpType::ReduceSum |
                    coelanox_core::ir::OpType::ReduceMax |
                    coelanox_core::ir::OpType::ReduceMin |
                    coelanox_core::ir::OpType::ReduceMean |
                    coelanox_core::ir::OpType::ReduceProd |
                    coelanox_core::ir::OpType::GlobalMaxPool |
                    coelanox_core::ir::OpType::GlobalAvgPool => {
                        // Reductions: input length is input_a.len(), output is scalar
                        self.input_a.len()
                    }
                    coelanox_core::ir::OpType::MaxPool |
                    coelanox_core::ir::OpType::AvgPool => {
                        // Pooling: input length is input_a.len(), output is input_a.len() / pool_size
                        // For pool_size=2, input is 2x output
                        self.input_a.len()
                    }
                    _ => {
                        // Regular unary: input length = output length
                        self.expected_output.len()
                    }
                }
            };
            
            func(
                combined_input.as_ptr(),
                input_len, // rsi: input length
                output.as_mut_ptr(),
                self.expected_output.len(), // rcx: output length (saved to r15)
                weights.as_ptr(),
                weights.len(),
            );
        }
        
        println!("Input A: {:?}", self.input_a);
        if let Some(ref input_b) = self.input_b {
            println!("Input B: {:?}", input_b);
        }
        println!("Output: {:?}", output);
        println!("Expected: {:?}", self.expected_output);
        
        // Verify
        for (i, (&actual, &expected)) in output.iter().zip(self.expected_output.iter()).enumerate() {
            let diff = (actual - expected).abs();
            if diff > self.tolerance {
                return Err(format!(
                    "Element {}: expected {}, got {}, diff: {}",
                    i, expected, actual, diff
                ));
            }
        }
        
        println!("✅ Test passed!");
        Ok(())
    }
}

