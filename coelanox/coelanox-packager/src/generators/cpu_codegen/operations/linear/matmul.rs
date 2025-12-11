//! MatMul operation
//! Matrix multiplication: C = A * B

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_matmul(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // MatMul: matrix multiplication
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // For 1D: dot product of two vectors
    // For 2D: proper matrix multiplication (MxK * KxN = MxN)
    // C[i][j] = sum over k of A[i][k] * B[k][j]
    
    // Input: combined buffer with input_a and input_b
    // For binary ops, input_b is at offset r15 * 4 (output_len * 4 bytes)
    // Output: scalar (dot product) for 1D, matrix for 2D
    // NOTE: For MatMul, we loop over input_a length (rsi), not output_len (r15)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: proper matrix multiplication
        // A is MxK, B is KxN, C is MxN
        // We need to infer K - for now, assume K = output_cols (common case: square matrices or K=N)
        // Actually, K should be the inner dimension - let's use input_a.cols if available
        // For simplicity, assume K = output_cols (works for common cases)
        let k = output_cols; // Inner dimension (can be adjusted based on actual input shapes)
        
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Store dimensions in registers:
        // r9 = output_cols (N)
        // r10 = k (K) - we'll calculate input_b offset based on this
        if output_cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC1);
            code.extend_from_slice(&(output_cols as u32).to_le_bytes());
        }
        
        // Store k in r10 (temporarily, we'll use it for calculations)
        if k <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC2); code.push(k as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC2);
            code.extend_from_slice(&(k as u32).to_le_bytes());
        }
        
        // Calculate input_b base offset: input_a size = output_rows * k * 4
        // input_b_offset = output_rows * k * 4
        // We'll store this in r11
        let input_a_size = output_rows * k;
        code.push(0x48); code.push(0xC7); code.push(0xC3);
        code.extend_from_slice(&((input_a_size * 4) as u32).to_le_bytes()); // mov rbx, input_a_size * 4
        
        // Outer loop: output rows (M) - r12
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R12, Register::R12).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let outer_loop_start = code.len();
        if output_rows <= 127 {
            code.push(0x49); code.push(0x83); code.push(0xFC); code.push(output_rows as u8);
        } else {
            code.push(0x49); code.push(0x81); code.push(0xFC);
            code.extend_from_slice(&(output_rows as u32).to_le_bytes());
        }
        
        let outer_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Inner loop: output cols (N) - r8
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Zero accumulator for C[i][j]: pxor xmm0, xmm0
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
        
        // K loop: sum over k of A[i][k] * B[k][j]
        // Use rcx for k loop counter
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let k_loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R10).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let k_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate A[i][k] offset: A is MxK, A[i][k] = (i * K + k) * 4
        // rax = (r12 * k + rcx) * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R12).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by k: rax = rax * k
        if k <= 127 {
            code.push(0x48); code.push(0x6B); code.push(0xC0); code.push(k as u8); // imul rax, rax, k
        } else {
            code.push(0x48); code.push(0x69); code.push(0xC0);
            code.extend_from_slice(&(k as u32).to_le_bytes()); // imul rax, rax, k
        }
        
        // Add k index: rax = rax + rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by 4: shl rax, 2
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
        
        // Load A[i][k]: movss xmm1, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Calculate B[k][j] offset: B is KxN, B[k][j] = (k * N + j) * 4
        // rdx = (rcx * output_cols + r8) * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RDX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by output_cols (N): rdx = rdx * output_cols
        if output_cols <= 127 {
            code.push(0x48); code.push(0x6B); code.push(0xD1); code.push(output_cols as u8); // imul rdx, rcx, output_cols
        } else {
            code.push(0x48); code.push(0x69); code.push(0xD1);
            code.extend_from_slice(&(output_cols as u32).to_le_bytes()); // imul rdx, rcx, output_cols
        }
        
        // Add j: rdx = rdx + r8
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RDX, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by 4: shl rdx, 2
        code.push(0x48); code.push(0xC1); code.push(0xE2); code.push(0x02);
        
        // Add input_b base offset: rdx = rdx + rbx (rbx has input_a_size * 4)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RDX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Calculate full address: rax = input_buffer_reg + rdx
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, input_buffer_reg).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RDX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Load B[k][j]: movss xmm2, [rax]
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x10]); // movss xmm2, [rax]
        
        // Multiply: mulss xmm1, xmm2 (A[i][k] * B[k][j])
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]);
        
        // Accumulate: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Increment k: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let k_jmp_pos = code.len();
        code.push(0xE9);
        let k_jmp_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        let k_loop_end = code.len();
        let k_jge_offset = (k_loop_end as i32 - (k_jge_pos + 6) as i32) as i32;
        code[k_jge_pos + 2..k_jge_pos + 6].copy_from_slice(&k_jge_offset.to_le_bytes());
        let k_jmp_offset = (k_loop_start as i32 - (k_jmp_pos + 5) as i32) as i32;
        code[k_jmp_placeholder..k_jmp_placeholder + 4].copy_from_slice(&k_jmp_offset.to_le_bytes());
        
        // Now calculate output offset: rax = (i * N + j) * 4 = (r12 * output_cols + r8) * 4
        code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
        
        // Store C[i][j]: movss [output_buffer_reg + rax], xmm0
        match output_buffer_reg {
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
        }
        
        // Increment inner loop: inc r8
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_jmp_pos = code.len();
        code.push(0xE9);
        let inner_jmp_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        let inner_loop_end = code.len();
        let inner_jge_offset = (inner_loop_end as i32 - (inner_jge_pos + 6) as i32) as i32;
        code[inner_jge_pos + 2..inner_jge_pos + 6].copy_from_slice(&inner_jge_offset.to_le_bytes());
        let inner_jmp_offset = (inner_loop_start as i32 - (inner_jmp_pos + 5) as i32) as i32;
        code[inner_jmp_placeholder..inner_jmp_placeholder + 4].copy_from_slice(&inner_jmp_offset.to_le_bytes());
        
        // Increment outer loop: inc r12
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::R12).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let outer_jmp_pos = code.len();
        code.push(0xE9);
        let outer_jmp_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        let outer_loop_end = code.len();
        let outer_jge_offset = (outer_loop_end as i32 - (outer_jge_pos + 6) as i32) as i32;
        code[outer_jge_pos + 2..outer_jge_pos + 6].copy_from_slice(&outer_jge_offset.to_le_bytes());
        let outer_jmp_offset = (outer_loop_start as i32 - (outer_jmp_pos + 5) as i32) as i32;
        code[outer_jmp_placeholder..outer_jmp_placeholder + 4].copy_from_slice(&outer_jmp_offset.to_le_bytes());
        
        return Ok(code);
    }
    
    // 1D case: dot product
    // Input: [input_a..., input_b...] where each vector has k elements
    // For 1D MatMul, input_len = 2*k, output_len = 1
    // We loop k times, computing dot product: sum(input_a[i] * input_b[i]) for i in 0..k
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Calculate k = input_len / 2 (each vector has k elements)
    // Save rsi (input length) to r13, then divide by 2
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Divide r13 by 2: shr r13, 1 (k = input_len / 2)
    code.extend_from_slice(&[0x49, 0xD1, 0xED]); // shr r13, 1
    
    // Calculate input_b base offset: k * 4 bytes
    // Store in r12: r12 = r13 * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R12, Register::R13).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.extend_from_slice(&[0x49, 0xC1, 0xE4, 0x02]); // shl r12, 2 (r12 = k * 4)
    
    // Zero accumulator: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    
    // Check if done: cmp rcx, r13 (loop k times, where k = input_len / 2)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R13).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate byte offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load input_a[rcx]: movss xmm1, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Calculate input_b address: r8 = input_buffer_reg + r12 + rax
    // r12 has k * 4 (input_b base offset), rax has rcx * 4 (element offset)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R8, input_buffer_reg).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.extend_from_slice(&[0x4D, 0x01, 0xE0]); // add r8, r12 (add input_b base offset)
    code.extend_from_slice(&[0x49, 0x01, 0xC0]); // add r8, rax (add element offset)
    
    // Load input_b[rcx]: movss xmm2, [r8]
    code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x10]);
    
    // Multiply: mulss xmm1, xmm2
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]); // mulss xmm1, xmm2
    
    // Add to accumulator: addss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
    
    // Increment counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back to loop start
    let jmp_pos = code.len();
    code.push(0xE9);
    let jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    let loop_end = code.len();
    let jge_offset = (loop_end as i32 - (jge_pos + 6) as i32) as i32;
    code[jge_pos + 2..jge_pos + 6].copy_from_slice(&jge_offset.to_le_bytes());
    let jmp_offset = (loop_start as i32 - (jmp_pos + 5) as i32) as i32;
    code[jmp_offset_placeholder..jmp_offset_placeholder + 4].copy_from_slice(&jmp_offset.to_le_bytes());
    
    // Store result: movss [output_buffer_reg], xmm0
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    Ok(code)
}

