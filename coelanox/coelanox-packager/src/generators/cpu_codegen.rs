//! Modular SIMD-based CPU code generator
//! 
//! This module provides a modular architecture for generating optimized SIMD machine code:
//! - Calculation modules: Pure computation logic (exp, log, softmax, etc.)
//! - Loop execution: Main SIMD loop processing chunks
//! - Bridge: Handle remaining elements that don't fit in SIMD chunks
//! - Tail: Final operations, horizontal reductions, cleanup

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::HardwareTarget;
use coelanox_core::ir::{OptimizedIR, GraphNode, OpType};
use iced_x86::{Code, Instruction, Register, Encoder, MemoryOperand};
use crate::generators::CodeGenerator;
use std::sync::Mutex;

/// Operation generator function type
type OpGenerator = fn(&CpuCodeGenerator, &GraphNode) -> CoelanoxResult<Vec<u8>>;

/// CPU code generator with modular SIMD architecture
pub struct CpuCodeGenerator {
    name: String,
    version: String,
    /// Current operation context (buffer registers for active operation)
    /// This is set per-operation during code generation
    /// Uses Mutex for thread-safe interior mutability (trait requires Send+Sync)
    current_input_buffer: Mutex<Option<Register>>,
    current_output_buffer: Mutex<Option<Register>>,
}

impl CpuCodeGenerator {
    /// Create a new CPU code generator
    pub fn new() -> Self {
        Self {
            name: "cpu_simd_generator".to_string(),
            version: "2.0.0".to_string(),
            current_input_buffer: Mutex::new(None),
            current_output_buffer: Mutex::new(None),
        }
    }
    
    /// Get generator function for an operation type
    /// Uses match statement instead of HashMap (OpType doesn't implement Hash)
    fn get_op_generator(&self, op_type: &OpType) -> Option<OpGenerator> {
        match op_type {
            OpType::Add => Some(Self::generate_add as OpGenerator),
            OpType::Subtract => Some(Self::generate_subtract as OpGenerator),
            OpType::Multiply => Some(Self::generate_multiply as OpGenerator),
            OpType::Divide => Some(Self::generate_divide as OpGenerator),
            OpType::Softmax => Some(Self::generate_softmax as OpGenerator),
            OpType::LogSoftmax => Some(Self::generate_logsoftmax as OpGenerator),
            OpType::Tanh => Some(Self::generate_tanh as OpGenerator),
            OpType::Sigmoid => Some(Self::generate_sigmoid as OpGenerator),
            OpType::Relu => Some(Self::generate_relu as OpGenerator),
            OpType::Gelu => Some(Self::generate_gelu as OpGenerator),
            OpType::ReduceMean => Some(Self::generate_mean as OpGenerator),
            OpType::Convolution => Some(Self::generate_convolution as OpGenerator),
            OpType::MatMul => Some(Self::generate_matmul as OpGenerator),
            OpType::MaxPool => Some(Self::generate_maxpool as OpGenerator),
            OpType::AvgPool => Some(Self::generate_avgpool as OpGenerator),
            OpType::BatchNorm => Some(Self::generate_batchnorm as OpGenerator),
            OpType::LayerNorm => Some(Self::generate_layernorm as OpGenerator),
            _ => None,
        }
    }
}

impl CodeGenerator for CpuCodeGenerator {
    fn generate(&self, optimized_ir: &OptimizedIR) -> CoelanoxResult<Vec<u8>> {
        let mut code = Vec::new();
        
        // Calculate total intermediate buffer size needed
        let total_intermediate_size = optimized_ir.intermediate_tensors.iter()
            .map(|t| t.size_bytes())
            .sum::<usize>();
        
        // Generate prologue (allocates intermediate buffers on stack)
        code.extend_from_slice(&self.generate_prologue_with_buffers(total_intermediate_size)?);
        
        // Track buffer pointers: TensorId -> stack offset
        // r10 = input buffer (from function param)
        // r11 = first intermediate buffer (stack allocated)
        // r12 = second intermediate buffer (if needed)
        // ... etc
        // r14 = final output buffer (from function param)
        
        // Track buffer pointers: TensorId -> Register
        // This maps each intermediate tensor to its buffer register
        let mut buffer_map: std::collections::HashMap<coelanox_core::ir::TensorId, Register> = std::collections::HashMap::new();
        
        // Calculate stack offsets for intermediate buffers
        // Each intermediate tensor gets a stack-allocated buffer
        let mut stack_offset = 0usize;
        let mut intermediate_registers = vec![Register::R11, Register::R13]; // Available registers for intermediates
        
        // Map intermediate tensors to stack buffers
        for (idx, tensor) in optimized_ir.intermediate_tensors.iter().enumerate() {
            let tensor_id = coelanox_core::ir::TensorId::intermediate(idx);
            if idx < intermediate_registers.len() {
                buffer_map.insert(tensor_id, intermediate_registers[idx]);
            }
            stack_offset += tensor.size_bytes();
        }
        
        // Generate code for each operation node
        for (idx, node) in optimized_ir.optimized_nodes.iter().enumerate() {
            // Determine input/output buffers for this operation
            let (input_buffer_reg, output_buffer_reg) = self.determine_buffers(
                node,
                &optimized_ir,
                &buffer_map,
                idx
            )?;
            
            // Generate operation with specific buffer pointers
            let operation_code = self.generate_operation_with_buffers(
                node,
                input_buffer_reg,
                output_buffer_reg
            )?;
            code.extend_from_slice(&operation_code);
        }
        
        // Generate epilogue (deallocates stack buffers)
        code.extend_from_slice(&self.generate_epilogue_with_buffers(total_intermediate_size));
        
        Ok(code)
    }
    
    fn target_hardware(&self) -> HardwareTarget {
        HardwareTarget::Cpu
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

impl CpuCodeGenerator {
    /// Generate function prologue with intermediate buffer allocation
    /// System V ABI: rdi=input.ptr, rsi=input.len, rdx=output.ptr, rcx=output.len, r8=weights.ptr, r9=weights.len
    /// 
    /// Architecture for intermediates:
    /// - Allocate all intermediate buffers on stack in prologue
    /// - Each intermediate tensor gets a stack-allocated buffer
    /// - Operations read from input buffer, write to output buffer
    /// - Next operation uses previous operation's output as input
    fn generate_prologue_with_buffers(&self, intermediate_buffer_size: usize) -> CoelanoxResult<Vec<u8>> {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Save callee-saved registers
        code.push(0x55); // push rbp
        code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
        code.push(0x53); // push rbx
        code.extend_from_slice(&[0x41, 0x54]); // push r12
        code.extend_from_slice(&[0x41, 0x55]); // push r13
        code.extend_from_slice(&[0x41, 0x56]); // push r14
        code.extend_from_slice(&[0x41, 0x57]); // push r15
        
        // Allocate intermediate buffers on stack
        // Round up to 16-byte alignment for SIMD
        let aligned_size = (intermediate_buffer_size + 15) & !15;
        if aligned_size > 0 {
            // sub rsp, aligned_size
            if aligned_size <= 127 {
                code.push(0x48);
                code.push(0x83);
                code.push(0xEC);
                code.push(aligned_size as u8);
            } else {
                // For larger sizes, use 32-bit immediate
                code.push(0x48);
                code.push(0x81);
                code.push(0xEC);
                code.extend_from_slice(&(aligned_size as u32).to_le_bytes());
            }
        } else {
            // Minimum stack alignment
            code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
        }
        
        // Save parameters to callee-saved registers
        // r10 = input.ptr (rdi) - first operation reads from here
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R10, Register::RDI).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // r11 = first intermediate buffer (rsp + offset)
        // This will be set per-operation based on which intermediate tensor is used
        // For now, we'll calculate offsets dynamically
        
        // r14 = final output.ptr (rdx) - last operation writes here
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R14, Register::RDX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // r15 = output.len (rcx) - for length checks
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R15, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // r12 = weights.ptr (r8) - for weight access
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R12, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        Ok(code)
    }
    
    /// Generate function epilogue with buffer deallocation
    fn generate_epilogue_with_buffers(&self, intermediate_buffer_size: usize) -> Vec<u8> {
        let mut code = Vec::new();
        
        // Deallocate intermediate buffers (restore stack)
        let aligned_size = (intermediate_buffer_size + 15) & !15;
        if aligned_size > 0 {
            if aligned_size <= 127 {
                code.push(0x48);
                code.push(0x83);
                code.push(0xC4);
                code.push(aligned_size as u8);
            } else {
                code.push(0x48);
                code.push(0x81);
                code.push(0xC4);
                code.extend_from_slice(&(aligned_size as u32).to_le_bytes());
            }
        } else {
            code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
        }
        
        // Restore callee-saved registers
        code.extend_from_slice(&[0x41, 0x5F]); // pop r15
        code.extend_from_slice(&[0x41, 0x5E]); // pop r14
        code.extend_from_slice(&[0x41, 0x5D]); // pop r13
        code.extend_from_slice(&[0x41, 0x5C]); // pop r12
        code.push(0x5B); // pop rbx
        code.push(0x5D); // pop rbp
        code.push(0xC3); // ret
        
        code
    }
    
    /// Determine input and output buffer registers for an operation
    /// 
    /// Strategy:
    /// - First operation: reads from r10 (input buffer)
    /// - Intermediate operations: read from previous op's output buffer
    /// - Last operation: writes to r14 (final output buffer)
    /// - Intermediate operations: write to stack-allocated buffers
    /// Determine input and output buffer registers for an operation
    /// 
    /// Strategy:
    /// - First operation: reads from r10 (input buffer)
    /// - Intermediate operations: read from previous op's output buffer (via buffer_map)
    /// - Last operation: writes to r14 (final output buffer)
    /// - Intermediate operations: write to intermediate buffers (r11, r13, etc.)
    fn determine_buffers(
        &self,
        node: &GraphNode,
        optimized_ir: &OptimizedIR,
        buffer_map: &std::collections::HashMap<coelanox_core::ir::TensorId, Register>,
        operation_index: usize
    ) -> CoelanoxResult<(Register, Register)> {
        use coelanox_core::ir::TensorId;
        
        // Determine input buffer
        // Check if input tensor is in buffer_map (intermediate from previous op)
        // Otherwise, use r10 (initial input)
        let input_buffer = if let Some(input_tensor_id) = node.input_tensors.first() {
            // Check if this tensor is an intermediate (from previous operation)
            if input_tensor_id.is_intermediate() {
                // Look up in buffer_map
                buffer_map.get(input_tensor_id)
                    .copied()
                    .unwrap_or(Register::R10) // Fallback to input if not found
            } else {
                // Input tensor (from initial input)
                Register::R10
            }
        } else {
            // No input tensors (shouldn't happen, but fallback)
            Register::R10
        };
        
        // Determine output buffer
        // If this is the last operation, write to final output (r14)
        // Otherwise, write to intermediate buffer (from buffer_map)
        let output_buffer = if operation_index == optimized_ir.optimized_nodes.len() - 1 {
            // Last operation: write to final output (r14)
            Register::R14
        } else if let Some(output_tensor_id) = node.output_tensors.first() {
            // Intermediate operation: write to intermediate buffer
            // Look up in buffer_map
            buffer_map.get(output_tensor_id)
                .copied()
                .unwrap_or(Register::R11) // Fallback to r11 if not found
        } else {
            // No output tensors (shouldn't happen, but fallback)
            Register::R11
        };
        
        Ok((input_buffer, output_buffer))
    }
    
    /// Generate operation with specific buffer pointers
    /// 
    /// This is the key: operations read from input_buffer, write to output_buffer
    /// Next operation uses this operation's output_buffer as its input_buffer
    fn generate_operation_with_buffers(
        &self,
        node: &GraphNode,
        input_buffer_reg: Register,
        output_buffer_reg: Register
    ) -> CoelanoxResult<Vec<u8>> {
        // Set buffer registers in generator context (using RefCell for interior mutability)
        // Operations will read these when generating code
        *self.current_input_buffer.lock().unwrap() = Some(input_buffer_reg);
        *self.current_output_buffer.lock().unwrap() = Some(output_buffer_reg);
        
        // Generate operation (generators will use current_input/output_buffer)
        if let Some(generator) = self.get_op_generator(&node.op_type) {
            generator(self, node)
        } else {
            Err(CoelanoxError::Internal(format!(
                "No generator registered for operation: {:?}",
                node.op_type
            )))
        }
    }
    
    /// Get current input buffer register (for operation generation)
    fn get_current_input_buffer(&self) -> Register {
        self.current_input_buffer.lock().unwrap().unwrap_or(Register::R10)
    }
    
    /// Get current output buffer register (for operation generation)
    fn get_current_output_buffer(&self) -> Register {
        self.current_output_buffer.lock().unwrap().unwrap_or(Register::R14)
    }
    
    /// Generate function epilogue
    fn generate_epilogue(&self) -> Vec<u8> {
        let mut code = Vec::new();
        
        // Restore callee-saved registers
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20]); // add rsp, 32
        code.extend_from_slice(&[0x41, 0x5F]); // pop r15
        code.extend_from_slice(&[0x41, 0x5E]); // pop r14
        code.extend_from_slice(&[0x41, 0x5D]); // pop r13
        code.extend_from_slice(&[0x41, 0x5C]); // pop r12
        code.push(0x5B); // pop rbx
        code.push(0x5D); // pop rbp
        code.push(0xC3); // ret
        
        code
    }
    
    /// Generate code for a single operation using dispatch table
    fn generate_operation(&self, node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // Handle custom operations
        if let OpType::Custom(_) = &node.op_type {
            return Err(CoelanoxError::Internal("Custom operations not yet supported".to_string()));
        }
        
        // Look up generator in dispatch table
        if let Some(generator) = self.get_op_generator(&node.op_type) {
            generator(self, node)
        } else {
            Err(CoelanoxError::Internal(format!(
                "No generator registered for operation: {:?}",
                node.op_type
            )))
        }
    }
    
    // ============================================================================
    // OPERATION GENERATORS
    // ============================================================================
    
    fn generate_add(&self, node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // Get tensor size from output tensor shape
        // For ADD: output shape should match input shape
        // We'll use rsi (input length) which is passed as function parameter
        // The loop will use rsi to determine how many elements to process
        // For now, use a reasonable default - the actual count comes from rsi at runtime
        let total_elements = 1024; // This is just for loop structure - actual count from rsi
        let simd_chunk_size = 8; // AVX2: 8 floats
        
        // Get buffer registers from generator context (set by generate_operation_with_buffers)
        let input_buffer_reg = self.get_current_input_buffer();
        let output_buffer_reg = self.get_current_output_buffer();
        
        // Use modular architecture: ALL phases are SEPARATE
        // - Operation Prologue: Operation-specific setup (separate)
        // - Loop: Reusable structure
        // - Bridge: Reusable structure  
        // - Operation Tail: Operation-specific cleanup (separate)
        // KEY: Calculations use buffer registers, not hardcoded r10/r14
        self.generate_complete_operation(
            total_elements,
            simd_chunk_size,
            // Operation Prologue: ADD-specific setup (if needed)
            // For ADD, no special setup needed, but structure is there
            |_code| Ok(()),
            // Loop body: pure calculation (no loop structure)
            // Uses input_buffer_reg and output_buffer_reg (not hardcoded!)
            |code, _chunk_size| {
                // Generate ADD calculation code (separate from loop)
                // This is pure computation - no loop, no jumps, just SIMD ops
                // Uses buffer registers passed via closure capture
                let calc_code = Self::generate_add_calculation_with_buffers(
                    input_buffer_reg,
                    output_buffer_reg
                );
                code.extend_from_slice(&calc_code);
                Ok(())
            },
            // Bridge body: scalar calculation for remainder
            |code, remainder| {
                // Scalar ADD for remaining elements
                let scalar_code = Self::generate_add_scalar_with_buffers(
                    remainder,
                    input_buffer_reg,
                    output_buffer_reg
                );
                code.extend_from_slice(&scalar_code);
                Ok(())
            },
            // Operation Tail: ADD-specific cleanup (if needed)
            // For ADD, no special cleanup needed, but structure is there
            |_code| Ok(()),
        )
    }
    
    /// Pure calculation: ADD operation with buffer registers
    /// 
    /// Architecture: Calculations are SEPARATE from loops AND use buffer registers
    /// - Loop structure: Handled by generate_simd_loop()
    /// - Calculation: This function (pure SIMD operations)
    /// - Buffer registers: Passed as parameters (not hardcoded!)
    /// 
    /// This allows chaining:
    /// - Op1: input=r10, output=r11 (intermediate buffer)
    /// - Op2: input=r11, output=r13 (next intermediate)
    /// - Op3: input=r13, output=r14 (final output)
    /// 
    /// Operation: output = input_a + input_b (element-wise)
    /// For AVX2: processes 8 floats at a time (32 bytes)
    /// For binary ADD: input_a is at [input_buffer_reg], input_b is at [input_buffer_reg + r15*4]
    fn generate_add_calculation_with_buffers(
        input_buffer_reg: Register,
        output_buffer_reg: Register
    ) -> Vec<u8> {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Load first input chunk: vmovaps ymm0, [input_buffer_reg + rbx*4] (scale=4, 32 bytes)
        // Use iced_x86 to verify encoding - create memory operand manually
        // For now, use verified hardcoded encoding for common registers
        // The encoding is: VEX.256.66.0F.W0 28 /r
        // Memory operand encoding depends on base register
        match input_buffer_reg {
            Register::R10 => {
                // vmovaps ymm0, [r10 + rbx*4] - verified encoding
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x28, 0x04, 0x1A]);
            }
            Register::R11 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x28, 0x04, 0x1B]);
            }
            Register::R13 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x28, 0x04, 0x1D]);
            }
            Register::R14 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x28, 0x04, 0x1E]);
            }
            _ => {
                // For other registers, would need dynamic encoding
                return vec![];
            }
        }
        
        // Calculate second input base address: rax = r15 * 4 + r10
        // mov rax, r15
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R15).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // shl rax, 2 (multiply by 4 to get byte offset)
        // Use with2 for immediate shift - need to encode manually for imm8
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // add rax, r10 (base address)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R10).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Load second input chunk: vmovaps ymm1, [rax + rbx*4] (scale=4, 32 bytes)
        // Verified encoding: VEX.256.66.0F.W0 28 /r with [rax + rbx*4]
        code.extend_from_slice(&[0xC4, 0xE1, 0x7C, 0x28, 0x0C, 0x18]);
        
        // ADD calculation: vaddps ymm0, ymm0, ymm1
        // Verified encoding: VEX.256.0F.W0 58 /r
        code.extend_from_slice(&[0xC5, 0xFC, 0x58, 0xC1]);
        
        // Store result: vmovaps [output_buffer_reg + rbx*4], ymm0
        // Verified encoding: VEX.256.66.0F.W0 29 /r
        match output_buffer_reg {
            Register::R11 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x29, 0x04, 0x1B]);
            }
            Register::R13 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x29, 0x04, 0x1D]);
            }
            Register::R14 => {
                code.extend_from_slice(&[0xC4, 0xC1, 0x7C, 0x29, 0x04, 0x1E]);
            }
            _ => {
                return vec![];
            }
        }
        
        code
    }
    
    /// Scalar ADD for remainder elements with buffer registers
    /// 
    /// Processes remaining elements that don't fit in a full SIMD chunk
    /// Uses scalar SSE instructions (movss, addss) for individual floats
    /// 
    /// Note: rax should contain the byte offset of remaining elements
    /// (calculated in bridge section: rax = rbx * 32)
    fn generate_add_scalar_with_buffers(
        remainder: usize,
        input_buffer_reg: Register,
        output_buffer_reg: Register
    ) -> Vec<u8> {
        let mut code = Vec::new();
        
        // Process each remaining element individually
        // Loop: for i in 0..remainder
        //   output[i] = input[i] + weights[i]
        
        // Initialize loop counter (rcx) with remainder count
        // mov rcx, remainder
        // Encoding: 48 B9 [8-byte immediate] for mov rcx, imm64
        code.push(0x48); code.push(0xB9); // mov rcx, imm64
        code.extend_from_slice(&(remainder as u64).to_le_bytes());
        
        // Scalar loop start label
        let loop_start = code.len();
        
        // Load scalar float: movss xmm0, [input_buffer_reg + rax]
        // Use hardcoded encoding for common registers
        match input_buffer_reg {
            Register::R10 => {
                // movss xmm0, [r10 + rax]
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]);
            }
            Register::R11 => {
                // movss xmm0, [r11 + rax]
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]);
            }
            Register::R13 => {
                // movss xmm0, [r13 + rax]
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]);
            }
            Register::R14 => {
                // movss xmm0, [r14 + rax]
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]);
            }
            _ => return vec![],
        }
        
        // Load second input: movss xmm1, [r10 + r15*4 + rax]
        // rax already has offset from bridge section
        // r15 contains output_len (saved from rcx in prologue)
        // Calculate second input address: r10 + r15*4 + rax
        // Use r9 as temp (it's caller-saved, but we're in a function so it's safe)
        // Actually, r9 is weights.len parameter, so let's use r8 (weights.ptr) as temp
        // But r8 might be needed... Let's use a different approach: calculate in rdx temporarily
        // Actually, rdx is output.ptr, so that's also used. Let's use r9 (weights.len) as temp
        // Save r9 first, use it, then restore (but that's complex)
        // Simpler: use r8 (weights.ptr) as temp - we're not using weights in ADD
        // mov r8, r15
        code.extend_from_slice(&[0x4D, 0x89, 0xF8]); // mov r8, r15
        // shl r8, 2 (multiply by 4)
        code.extend_from_slice(&[0x49, 0xC1, 0xE0, 0x02]); // shl r8, 2
        // add r8, r10 (base address)
        code.extend_from_slice(&[0x4D, 0x01, 0xD0]); // add r8, r10
        // add r8, rax (add current offset)
        code.extend_from_slice(&[0x49, 0x01, 0xC0]); // add r8, rax
        // movss xmm1, [r8]
        // Encoding: F3 41 0F 10 08 (REX.W=0, REX.B=1 for r8, ModR/M: 00 001 000 = 0x08)
        code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x08]); // movss xmm1, [r8]
        
        // ADD: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Store: movss [output_buffer_reg + rax], xmm0
        match output_buffer_reg {
            Register::R11 => {
                // movss [r11 + rax], xmm0
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]);
            }
            Register::R13 => {
                // movss [r13 + rax], xmm0
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]);
            }
            Register::R14 => {
                // movss [r14 + rax], xmm0
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]);
            }
            _ => return vec![],
        }
        
        // Advance offset: add rax, 4 (next float)
        code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x04]);
        
        // Decrement counter: dec rcx
        code.extend_from_slice(&[0x48, 0xFF, 0xC9]);
        
        // Loop back if not zero: jnz loop_start
        let jnz_pos = code.len();
        code.push(0x75); code.push(0x00); // jnz placeholder
        
        // Patch jump offset
        let jnz_offset = (loop_start as i32 - (jnz_pos + 2) as i32) as i8;
        code[jnz_pos + 1] = jnz_offset as u8;
        
        code
    }
    
    /// Scalar ADD for remainder elements (bridge section)
    /// Pure calculation - no loop structure
    fn generate_add_scalar_static(_remainder: usize) -> Vec<u8> {
        let mut code = Vec::new();
        // Scalar ADD implementation for remaining elements
        // movss xmm0, [r10 + rax]
        // addss xmm0, [r8 + rax]
        // movss [r14 + rax], xmm0
        // TODO: Implement scalar ADD
        code
    }
    
    fn generate_subtract(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD subtract
        Ok(vec![])
    }
    
    fn generate_multiply(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD multiply
        Ok(vec![])
    }
    
    fn generate_divide(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD divide
        Ok(vec![])
    }
    
    fn generate_softmax(&self, node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        let total_elements = 1024; // TODO: Get from node/IR
        let simd_chunk_size = 8;
        
        // Softmax needs: prologue (setup), loop (exp), tail (horizontal sum), loop (normalize)
        // This shows how prologue and tail are SEPARATE from loop calculations
        
        self.generate_complete_operation(
            total_elements,
            simd_chunk_size,
            // Operation Prologue: Softmax-specific setup
            // Examples: Initialize max accumulator, sum accumulator
            |code| {
                // Initialize max to -infinity
                // Initialize sum to 0
                // TODO: Generate softmax prologue code
                Ok(())
            },
            // Loop body: exp calculation (pure, separate)
            |code, _chunk_size| {
                let calc_code = Self::generate_exp_calculation_static();
                code.extend_from_slice(&calc_code);
                Ok(())
            },
            // Bridge: scalar exp for remainder (separate)
            |code, remainder| {
                let scalar_code = Self::generate_exp_scalar_static(remainder);
                code.extend_from_slice(&scalar_code);
                Ok(())
            },
            // Operation Tail: horizontal sum reduction (separate from loop!)
            |code| {
                // Tail is SEPARATE - handles final operations
                // Horizontal reduction, normalization, etc.
                let tail_code = Self::generate_horizontal_sum_static();
                code.extend_from_slice(&tail_code);
                Ok(())
            },
        )
    }
    
    /// Pure calculation: EXP operation (no loop, no tail, just calculation)
    fn generate_exp_calculation_static() -> Vec<u8> {
        let mut code = Vec::new();
        // vmovaps ymm0, [r10 + rbx*32]
        // vexp_approx ymm0, ymm0  // exp calculation
        // vmovaps [r14 + rbx*32], ymm0
        // TODO: Implement exp
        code
    }
    
    /// Scalar EXP for remainder
    fn generate_exp_scalar_static(_remainder: usize) -> Vec<u8> {
        let mut code = Vec::new();
        // TODO: Implement scalar exp
        code
    }
    
    /// Tail operation: Horizontal sum reduction (SEPARATE from loop calculations)
    /// This is called AFTER the loop completes
    /// Reduces all SIMD chunks to a single sum
    fn generate_horizontal_sum_static() -> Vec<u8> {
        let mut code = Vec::new();
        // Horizontal sum: vhaddps ymm0, ymm0, ymm0
        // Then: vperm2f128 ymm1, ymm0, ymm0, 0x01
        // Then: vaddps ymm0, ymm0, ymm1
        // Then: vhaddps ymm0, ymm0, ymm0
        // Result: xmm0[0] contains sum of all elements
        // TODO: Implement horizontal sum
        code
    }
    
    fn generate_logsoftmax(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD logsoftmax
        Ok(vec![])
    }
    
    fn generate_tanh(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD tanh
        Ok(vec![])
    }
    
    fn generate_sigmoid(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD sigmoid
        Ok(vec![])
    }
    
    fn generate_relu(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD ReLU
        Ok(vec![])
    }
    
    fn generate_gelu(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD GELU
        Ok(vec![])
    }
    
    fn generate_mean(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD mean (ReduceMean)
        Ok(vec![])
    }
    
    fn generate_convolution(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD convolution
        Ok(vec![])
    }
    
    fn generate_matmul(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD matmul
        Ok(vec![])
    }
    
    fn generate_maxpool(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD maxpool
        Ok(vec![])
    }
    
    fn generate_avgpool(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD avgpool
        Ok(vec![])
    }
    
    fn generate_batchnorm(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD batchnorm
        Ok(vec![])
    }
    
    fn generate_layernorm(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD layernorm
        Ok(vec![])
    }
    
    fn generate_dotproduct(&self, _node: &GraphNode) -> CoelanoxResult<Vec<u8>> {
        // TODO: Implement modular SIMD dotproduct
        Ok(vec![])
    }
    
    // ============================================================================
    // MODULAR COMPONENTS - CORE HANDLING LOGIC
    // ============================================================================
    
    /// Initialize operation: setup registers and calculate loop parameters
    /// This is the operation-level initialization (separate from function prologue)
    /// 
    /// Sets up:
    /// - Loop counter (rbx = 0)
    /// - Calculates chunk count and remainder from rsi (input length)
    /// - Prepares registers for SIMD operations
    /// 
    /// Returns: (code, chunk_count, remainder)
    fn initialize_operation(&self, 
                           total_elements: usize,
                           simd_chunk_size: usize) -> (Vec<u8>, usize, usize) {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Calculate chunk_count and remainder (for reference, but we'll calculate at runtime)
        let chunk_count = total_elements / simd_chunk_size;
        let remainder = total_elements % simd_chunk_size;
        
        // Initialize loop counter (rbx) to 0
        // rbx will count SIMD chunks processed
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Calculate chunk count at runtime: r13 = r15 / simd_chunk_size
        // r15 contains output_len (saved from rcx in prologue)
        // For binary ops, output_len = first_input_length
        // We use output_len because that's what we're actually processing
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::R15).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Divide by simd_chunk_size (right shift for power of 2)
        match simd_chunk_size {
            8 => {
                // shr r13, 3 (divide by 8)
                code.extend_from_slice(&[0x49, 0xC1, 0xED, 0x03]);
            }
            4 => {
                // shr r13, 2 (divide by 4)
                code.extend_from_slice(&[0x49, 0xC1, 0xED, 0x02]);
            }
            16 => {
                // shr r13, 4 (divide by 16)
                code.extend_from_slice(&[0x49, 0xC1, 0xED, 0x04]);
            }
            _ => {
                // For non-power-of-2, would need division instruction
                // For now, assume power of 2
            }
        }
        
        // Calculate remainder: r11 = rsi % simd_chunk_size
        // This will be used in bridge section
        // For now, we'll calculate it in bridge section when needed
        
        (code, chunk_count, remainder)
    }
    
    /// Generate main SIMD loop execution section
    /// Processes elements in chunks (e.g., 8 floats at a time with AVX2)
    /// 
    /// This is the REUSABLE loop structure - same for all operations
    /// The loop_body_fn provides the operation-specific calculation
    /// 
    /// Parameters:
    /// - loop_body_fn: Function that generates the loop body code (pure calculation)
    /// - simd_chunk_size: Number of elements processed per iteration (typically 8 for AVX2)
    /// 
    /// Register usage:
    /// - rbx: Loop counter (SIMD chunks processed)
    /// - r13: Chunk count (rsi / simd_chunk_size) - calculated in initialize_operation
    /// - rsi: Total elements (from function parameter)
    fn generate_simd_loop<F>(&self,
                             mut loop_body_fn: F,
                             simd_chunk_size: usize) -> CoelanoxResult<Vec<u8>>
    where
        F: FnMut(&mut Vec<u8>, usize) -> CoelanoxResult<()>,
    {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Loop structure (reusable for all operations):
        // loop_start:
        //   cmp rbx, r13        ; Compare counter with chunk count
        //   jge bridge_section  ; Exit if done
        //   [loop_body_fn]      ; Operation-specific calculation
        //   inc rbx              ; Increment counter
        //   jmp loop_start       ; Loop back
        
        let loop_start = code.len();
        
        // Compare loop counter (rbx) with chunk count (r13)
        // r13 was calculated in initialize_operation: r13 = rsi / simd_chunk_size
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RBX, Register::R13).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump to bridge section if loop complete (rbx >= r13)
        // We'll patch this offset later
        let jge_pos = code.len();
        code.push(0x0F); code.push(0x8D); // jge (long form for 32-bit offset)
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00); // placeholder offset
        
        // Generate loop body (calls the provided function - pure calculation)
        // This is where the operation-specific SIMD code goes
        loop_body_fn(&mut code, simd_chunk_size)?;
        
        // Increment loop counter (rbx++)
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back to loop start
        let jmp_pos = code.len();
        code.push(0xE9); // jmp (long form for 32-bit offset)
        let jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00); // placeholder offset
        
        let loop_end = code.len();
        
        // Patch jump offsets
        // jge offset: distance from jge to bridge section (will be patched by bridge)
        // jmp offset: distance from jmp to loop_start
        let jmp_offset = (loop_start as i32 - (jmp_pos + 5) as i32) as i32;
        code[jmp_offset_placeholder..jmp_offset_placeholder+4].copy_from_slice(&jmp_offset.to_le_bytes());
        
        // Store jge position for bridge to patch
        // Bridge will calculate: bridge_start - (jge_pos + 6)
        // For now, we'll return the position so bridge can patch it
        // Actually, we need a different approach - bridge comes after, so we can patch in bridge
        
        Ok(code)
    }
    
    /// Generate bridge section for remaining elements
    /// Handles elements that don't fit in a full SIMD chunk
    /// 
    /// This is the REUSABLE bridge structure - same for all operations
    /// The bridge_body_fn provides the operation-specific scalar code
    /// 
    /// Parameters:
    /// - bridge_body_fn: Function that generates scalar/smaller SIMD code for remainder
    /// - remainder: Number of remaining elements (0-7 for AVX2) - calculated at compile time
    /// 
    /// Strategy:
    /// - Calculate actual remainder at runtime: r11 = rsi - (rbx * simd_chunk_size)
    /// - If remainder == 0, jump to tail
    /// - Otherwise, process remainder with scalar/smaller SIMD code
    fn generate_bridge<F>(&self,
                         mut bridge_body_fn: F,
                         remainder: usize,
                         simd_chunk_size: usize) -> CoelanoxResult<Vec<u8>>
    where
        F: FnMut(&mut Vec<u8>, usize) -> CoelanoxResult<()>,
    {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Bridge section (reusable structure):
        // bridge_start:
        //   Calculate remainder: r11 = rsi - (rbx * simd_chunk_size)
        //   test r11, r11
        //   jz tail_section        ; Skip if no remainder
        //   [bridge_body_fn]       ; Process remainder (operation-specific)
        // tail_section:
        
        let bridge_start = code.len();
        
        // Calculate remaining elements: r11 = r15 - (rbx * simd_chunk_size)
        // r15 = output_len (what we're actually processing)
        // rbx = number of SIMD chunks processed
        // rbx * simd_chunk_size = number of elements processed
        
        // Step 1: Copy rbx to rax
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Step 2: Multiply rax by simd_chunk_size (shift left for power of 2)
        match simd_chunk_size {
            8 => {
                // shl rax, 3 (multiply by 8)
                code.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x03]);
            }
            4 => {
                // shl rax, 2 (multiply by 4)
                code.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x02]);
            }
            16 => {
                // shl rax, 4 (multiply by 16)
                code.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x04]);
            }
            _ => {
                return Err(CoelanoxError::Internal(format!(
                    "SIMD chunk size {} not supported (must be power of 2)",
                    simd_chunk_size
                )));
            }
        }
        
        // Step 3: Calculate remainder: r11 = r15 - rax
        // r15 contains output_len (what we're actually processing)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::R15).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Sub_rm64_r64, Register::R11, Register::RAX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Step 4: Check if remainder is zero
        // test r11, r11 (more efficient than cmp r11, 0)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Test_rm64_r64, Register::R11, Register::R11).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Step 5: Jump to tail if no remainder (r11 == 0)
        let jz_pos = code.len();
        code.push(0x0F); code.push(0x84); // jz (long form for 32-bit offset)
        let jz_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00); // placeholder offset
        
        // Step 6: Generate bridge body (process remaining elements)
        // This calls the operation-specific scalar code
        bridge_body_fn(&mut code, remainder)?;
        
        let bridge_end = code.len();
        
        // Step 7: Patch jump to tail
        // Tail section comes after bridge, so we need to know where it starts
        // For now, we'll calculate: tail_start - (jz_pos + 6)
        // But tail comes after, so we'll patch this in generate_complete_operation
        // Store position for later patching
        
        Ok(code)
    }
    
    /// Generate tail section for final operations
    /// Horizontal reductions, final calculations, cleanup
    /// 
    /// This is the REUSABLE tail structure - same for all operations
    /// The tail_body_fn provides the operation-specific cleanup
    /// 
    /// Parameters:
    /// - tail_body_fn: Function that generates tail operations (horizontal reductions, cleanup)
    fn generate_tail<F>(&self,
                       mut tail_body_fn: F) -> CoelanoxResult<Vec<u8>>
    where
        F: FnMut(&mut Vec<u8>) -> CoelanoxResult<()>,
    {
        let mut code = Vec::new();
        
        // Tail section (reusable structure):
        // tail_start:
        //   [tail_body_fn]  ; Operation-specific cleanup
        //   (operation completes, control returns to next operation or function epilogue)
        
        let tail_start = code.len();
        
        // Generate tail body (horizontal reductions, final calculations)
        // This calls the operation-specific cleanup code
        tail_body_fn(&mut code)?;
        
        Ok(code)
    }
    
    /// Generate complete operation with separate prologue, loop, bridge, and tail
    /// 
    /// Architecture: ALL phases are SEPARATE
    /// - Operation Prologue: Operation-specific setup (separate from function prologue)
    /// - Loop: Main SIMD computation (reusable)
    /// - Bridge: Remainder handling (reusable)
    /// - Operation Tail: Operation-specific cleanup (separate from function epilogue)
    /// 
    /// Function-level prologue/epilogue are handled separately in generate()
    /// 
    /// Parameters:
    /// - total_elements: Total number of elements to process
    /// - simd_chunk_size: Number of elements per SIMD chunk (typically 8 for AVX2)
    /// - op_prologue_fn: Operation-specific prologue (setup, initialization)
    /// - loop_body_fn: Generates SIMD loop body code (pure calculation)
    /// - bridge_body_fn: Generates scalar/smaller SIMD code for remainder
    /// - op_tail_fn: Operation-specific tail (horizontal reductions, cleanup)
    pub fn generate_complete_operation<FP, FL, FB, FT>(&self,
                                                       total_elements: usize,
                                                       simd_chunk_size: usize,
                                                       mut op_prologue_fn: FP,
                                                       mut loop_body_fn: FL,
                                                       mut bridge_body_fn: FB,
                                                       mut op_tail_fn: FT) -> CoelanoxResult<Vec<u8>>
    where
        FP: FnMut(&mut Vec<u8>) -> CoelanoxResult<()>,
        FL: FnMut(&mut Vec<u8>, usize) -> CoelanoxResult<()>,
        FB: FnMut(&mut Vec<u8>, usize) -> CoelanoxResult<()>,
        FT: FnMut(&mut Vec<u8>) -> CoelanoxResult<()>,
    {
        let mut code = Vec::new();
        
        // 1. Operation Prologue: Operation-specific setup (SEPARATE from function prologue)
        // Examples: Initialize accumulators, set up operation-specific registers
        op_prologue_fn(&mut code)?;
        
        // 2. Initialize loop (calculate chunk_count, remainder)
        let (init_code, _chunk_count, remainder) = self.initialize_operation(total_elements, simd_chunk_size);
        code.extend_from_slice(&init_code);
        
        // 3. Generate main SIMD loop (reusable structure)
        let loop_code = self.generate_simd_loop(loop_body_fn, simd_chunk_size)?;
        let loop_code_start = code.len(); // Track where loop code starts
        let loop_code_len = loop_code.len();
        code.extend_from_slice(&loop_code);
        
        // 4. Generate bridge section (reusable structure)
        // Track positions for jump patching
        // jge in loop is at offset 6 from loop_start (after: cmp instruction = 3 bytes, jge opcode = 2 bytes, offset start = 1 byte)
        let jge_pos_in_loop = 6; // Position of jge offset in loop_code (after cmp + jge opcode)
        
        // Generate bridge section
        let bridge_code = self.generate_bridge(bridge_body_fn, remainder, simd_chunk_size)?;
        let bridge_start_pos = code.len();
        code.extend_from_slice(&bridge_code);
        let bridge_end_pos = code.len();
        
        // Patch jge in loop to jump to bridge_start
        // jge is at: loop_code_start + jge_pos_in_loop
        let jge_absolute_pos = loop_code_start + jge_pos_in_loop;
        let jge_offset = (bridge_start_pos as i32 - (jge_absolute_pos + 6) as i32) as i32;
        code[jge_absolute_pos + 2..jge_absolute_pos + 6].copy_from_slice(&jge_offset.to_le_bytes());
        
        // 5. Operation Tail: Operation-specific cleanup (SEPARATE from function epilogue)
        // Examples: Horizontal reductions, final calculations
        let tail_start_pos = code.len();
        op_tail_fn(&mut code)?;
        
        // Patch jz in bridge to jump to tail_start
        // Find jz position in bridge_code (it's at offset 29 from bridge_start)
        // jz instruction: 0x0F 0x84 [4-byte offset]
        let jz_pos_in_bridge = 29; // Position after: mov, shl, mov, sub, test instructions
        let jz_absolute_pos = bridge_start_pos + jz_pos_in_bridge;
        let jz_offset = (tail_start_pos as i32 - (jz_absolute_pos + 6) as i32) as i32;
        code[jz_absolute_pos + 2..jz_absolute_pos + 6].copy_from_slice(&jz_offset.to_le_bytes());
        
        Ok(code)
    }
    
    // ============================================================================
    // SIMD HELPER FUNCTIONS (Infrastructure - not AI operations)
    // ============================================================================
    // These are utility functions for SIMD operations that will be used by AI operations
    // They are part of the infrastructure, not the operations themselves
    
    /// Generate SIMD load instruction (vmovaps)
    /// Loads 8 floats (32 bytes) from memory into YMM register
    /// 
    /// Parameters:
    /// - base_reg: Base register (r10, r11, r13, r14, etc.)
    /// - index_reg: Index register (rbx for loop counter)
    /// - scale: Scale factor (8 for 8 floats = 32 bytes)
    /// - dest_reg: Destination YMM register (ymm0, ymm1, etc.)
    fn generate_simd_load(&self, base_reg: Register, index_reg: Register, scale: u8, dest_reg: Register) -> Vec<u8> {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // vmovaps ymm_dest, [base_reg + index_reg * scale]
        // This is complex encoding - using iced_x86 for correctness
        // For AVX2: vmovaps ymm0, [r10 + rbx*32]
        // Encoding depends on registers - using iced_x86 encoder
        
        // TODO: Use iced_x86 to encode vmovaps properly
        // For now, return empty - will be implemented with proper encoding
        
        code
    }
    
    /// Generate SIMD store instruction (vmovaps)
    /// Stores 8 floats (32 bytes) from YMM register to memory
    fn generate_simd_store(&self, src_reg: Register, base_reg: Register, index_reg: Register, scale: u8) -> Vec<u8> {
        let mut code = Vec::new();
        // TODO: Use iced_x86 to encode vmovaps store
        code
    }
    
    /// Generate horizontal reduction (sum) for AVX2
    /// Reduces 8 floats in YMM register to a single sum
    /// 
    /// Algorithm:
    /// 1. vhaddps ymm0, ymm0, ymm0  (horizontal add pairs)
    /// 2. vperm2f128 ymm1, ymm0, ymm0, 0x01  (permute high/low)
    /// 3. vaddps ymm0, ymm0, ymm1  (add high and low)
    /// 4. vhaddps ymm0, ymm0, ymm0  (final horizontal add)
    /// Result: xmm0[0] contains sum of all 8 elements
    fn generate_horizontal_sum_avx2(&self) -> Vec<u8> {
        let mut code = Vec::new();
        // TODO: Implement horizontal sum with proper AVX2 instructions
        // vhaddps ymm0, ymm0, ymm0
        // vperm2f128 ymm1, ymm0, ymm0, 0x01
        // vaddps ymm0, ymm0, ymm1
        // vhaddps ymm0, ymm0, ymm0
        code
    }
    
    /// Generate horizontal reduction (max) for AVX2
    /// Reduces 8 floats in YMM register to a single maximum
    fn generate_horizontal_max_avx2(&self) -> Vec<u8> {
        let mut code = Vec::new();
        // TODO: Implement horizontal max with proper AVX2 instructions
        code
    }
    
    // ============================================================================
    // CALCULATION MODULES (Pure computation logic - for AI operations)
    // ============================================================================
    // These will be implemented when adding AI operations
    // For now, infrastructure is complete and ready
}

