//! BatchNorm operation
//! (x - mean) / sqrt(var + epsilon) * gamma + beta
//! Computes mean and variance from input, then normalizes

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_batchnorm(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // BatchNorm: (x - mean) / sqrt(var + epsilon) * gamma + beta
    // Weights contain: [gamma (scale), beta (bias)] in r12 buffer
    // Algorithm:
    // 1. Compute mean from input
    // 2. Compute variance from input
    // 3. Normalize: output = gamma * (input - mean) / sqrt(var + epsilon) + beta
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: compute mean/variance over all elements, then normalize
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Load epsilon: mov eax, 0x3727C5AC; movd xmm6, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3727C5ACu32.to_le_bytes()); // 1e-5f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF0]); // movd xmm6, eax
        
        // Load scale (gamma) and bias (beta) from weights buffer (r12)
        // For safety, check if weights buffer is available (r9 >= 2), otherwise use defaults
        // Check: cmp r9, 2; jb use_defaults
        code.push(0x49); code.push(0x83); code.push(0xF9); code.push(0x02); // cmp r9, 2
        let jb_pos = code.len();
        code.push(0x0F); code.push(0x82);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Weights available: Load gamma and beta from r12
        // Load gamma: movss xmm2, [r12]
        code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x14, 0x24]);
        // Load beta: movss xmm3, [r12 + 4]
        code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x5C, 0x24, 0x04]);
        
        // Jump past defaults
        let jmp_past_defaults_pos = code.len();
        code.push(0xE9);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Use defaults: gamma = 1.0, beta = 0.0
        let defaults_start = code.len();
        let jb_offset = (defaults_start as i32 - (jb_pos + 6) as i32) as i32;
        code[jb_pos + 2..jb_pos + 6].copy_from_slice(&jb_offset.to_le_bytes());
        
        // Load gamma = 1.0: mov eax, 0x3F800000; movd xmm2, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
        
        // Load beta = 0.0: pxor xmm3, xmm3
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xDB]); // pxor xmm3, xmm3
        
        // Patch jump past defaults
        let after_defaults = code.len();
        let jmp_past_defaults_offset = (after_defaults as i32 - (jmp_past_defaults_pos + 5) as i32) as i32;
        code[jmp_past_defaults_pos + 1..jmp_past_defaults_pos + 5].copy_from_slice(&jmp_past_defaults_offset.to_le_bytes());
        
        // ============================================================================
        // PASS 1: Compute mean
        // ============================================================================
        // Save input length (rsi) to r11 for loop
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Zero accumulator: pxor xmm0, xmm0
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
        
        // Initialize loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass1_loop_start = code.len();
        
        // Check if done: cmp rcx, r11
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass1_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate byte offset: rax = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load input: movss xmm1, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Accumulate: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Increment counter: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back: jmp pass1_loop_start
        let pass1_jmp_pos = code.len();
        code.push(0xE9);
        let pass1_jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch pass1 jumps
        let pass1_loop_end = code.len();
        let pass1_jge_offset = (pass1_loop_end as i32 - (pass1_jge_pos + 6) as i32) as i32;
        code[pass1_jge_pos + 2..pass1_jge_pos + 6].copy_from_slice(&pass1_jge_offset.to_le_bytes());
        let pass1_jmp_offset = (pass1_loop_start as i32 - (pass1_jmp_pos + 5) as i32) as i32;
        code[pass1_jmp_offset_placeholder..pass1_jmp_offset_placeholder + 4].copy_from_slice(&pass1_jmp_offset.to_le_bytes());
        
        // Compute mean: divss xmm0, [count]
        // Convert count to float: cvtsi2ss xmm1, r11
        code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]); // cvtsi2ss xmm1, r11
        // Divide sum by count: divss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
        // xmm0 now contains mean - save to xmm4
        code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0
        
        // ============================================================================
        // PASS 2: Compute variance
        // ============================================================================
        // Zero accumulator: pxor xmm0, xmm0
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
        
        // Reset loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass2_loop_start = code.len();
        
        // Check if done: cmp rcx, r11
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass2_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate byte offset: rax = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load input: movss xmm1, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Compute (input - mean): subss xmm1, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xCC]);
        
        // Compute (input - mean)^2: mulss xmm1, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]);
        
        // Accumulate variance: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Increment counter: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back: jmp pass2_loop_start
        let pass2_jmp_pos = code.len();
        code.push(0xE9);
        let pass2_jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch pass2 jumps
        let pass2_loop_end = code.len();
        let pass2_jge_offset = (pass2_loop_end as i32 - (pass2_jge_pos + 6) as i32) as i32;
        code[pass2_jge_pos + 2..pass2_jge_pos + 6].copy_from_slice(&pass2_jge_offset.to_le_bytes());
        let pass2_jmp_offset = (pass2_loop_start as i32 - (pass2_jmp_pos + 5) as i32) as i32;
        code[pass2_jmp_offset_placeholder..pass2_jmp_offset_placeholder + 4].copy_from_slice(&pass2_jmp_offset.to_le_bytes());
        
        // Compute variance: divss xmm0, [count]
        // Convert count to float: cvtsi2ss xmm1, r11
        code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]); // cvtsi2ss xmm1, r11
        // Divide sum by count: divss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
        // xmm0 now contains variance - save to xmm5
        code.extend_from_slice(&[0x0F, 0x28, 0xE8]); // movaps xmm5, xmm0
        
        // Compute std = sqrt(variance + epsilon): addss xmm5, xmm6; sqrtss xmm5, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xEE]); // addss xmm5, xmm6
        code.extend_from_slice(&[0xF3, 0x0F, 0x51, 0xED]); // sqrtss xmm5, xmm5
        
        // ============================================================================
        // PASS 3: Normalize and apply scale/bias
        // ============================================================================
        // Store cols in r9 for 2D offset calculation
        if cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC1);
            code.extend_from_slice(&(cols as u32).to_le_bytes());
        }
        
        // Outer loop: rows (rbx)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let outer_loop_start = code.len();
        
        // Check if done: cmp rbx, rows
        if rows <= 127 {
            code.push(0x48); code.push(0x83); code.push(0xFB); code.push(rows as u8);
        } else {
            code.push(0x48); code.push(0x81); code.push(0xFB);
            code.extend_from_slice(&(rows as u32).to_le_bytes());
        }
        
        let outer_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Inner loop: cols (r8)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_loop_start = code.len();
        
        // Check if done: cmp r8, r9 (cols)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate 2D offset: rax = (rbx * cols + r8) * 4
        code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
        
        // Load input: movss xmm0, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Compute (input - mean): subss xmm0, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
        
        // Divide by std: divss xmm0, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC5]);
        
        // Multiply by gamma: mulss xmm0, xmm2
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC2]);
        
        // Add beta: addss xmm0, xmm3
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]);
        
        // Store: movss [output_buffer_reg + rax], xmm0
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
        
        // Jump back to inner loop start
        let inner_jmp_pos = code.len();
        code.push(0xE9);
        let inner_jmp_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch inner loop jumps
        let inner_loop_end = code.len();
        let inner_jge_offset = (inner_loop_end as i32 - (inner_jge_pos + 6) as i32) as i32;
        code[inner_jge_pos + 2..inner_jge_pos + 6].copy_from_slice(&inner_jge_offset.to_le_bytes());
        let inner_jmp_offset = (inner_loop_start as i32 - (inner_jmp_pos + 5) as i32) as i32;
        code[inner_jmp_placeholder..inner_jmp_placeholder + 4].copy_from_slice(&inner_jmp_offset.to_le_bytes());
        
        // Increment outer loop: inc rbx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back to outer loop start
        let outer_jmp_pos = code.len();
        code.push(0xE9);
        let outer_jmp_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch outer loop jumps
        let outer_loop_end = code.len();
        let outer_jge_offset = (outer_loop_end as i32 - (outer_jge_pos + 6) as i32) as i32;
        code[outer_jge_pos + 2..outer_jge_pos + 6].copy_from_slice(&outer_jge_offset.to_le_bytes());
        let outer_jmp_offset = (outer_loop_start as i32 - (outer_jmp_pos + 5) as i32) as i32;
        code[outer_jmp_placeholder..outer_jmp_placeholder + 4].copy_from_slice(&outer_jmp_offset.to_le_bytes());
        
        return Ok(code);
    }
    
    // 1D case: compute mean/variance, then normalize
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load epsilon: mov eax, 0x3727C5AC; movd xmm6, eax
    code.push(0xB8);
    code.extend_from_slice(&0x3727C5ACu32.to_le_bytes()); // 1e-5f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF0]); // movd xmm6, eax
    
    // Load scale (gamma) and bias (beta) from weights buffer (r12)
    // For safety, check if weights buffer is available (r9 >= 2), otherwise use defaults
    // Check: cmp r9, 2; jb use_defaults
    code.push(0x49); code.push(0x83); code.push(0xF9); code.push(0x02); // cmp r9, 2
    let jb_pos = code.len();
    code.push(0x0F); code.push(0x82);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Weights available: Load gamma and beta from r12
    // Load gamma: movss xmm2, [r12]
    code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x14, 0x24]);
    // Load beta: movss xmm3, [r12 + 4]
    code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x5C, 0x24, 0x04]);
    
    // Jump past defaults
    let jmp_past_defaults_pos = code.len();
    code.push(0xE9);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Use defaults: gamma = 1.0, beta = 0.0
    let defaults_start = code.len();
    let jb_offset = (defaults_start as i32 - (jb_pos + 6) as i32) as i32;
    code[jb_pos + 2..jb_pos + 6].copy_from_slice(&jb_offset.to_le_bytes());
    
    // Load gamma = 1.0: mov eax, 0x3F800000; movd xmm2, eax
    code.push(0xB8);
    code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
    
    // Load beta = 0.0: pxor xmm3, xmm3
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xDB]); // pxor xmm3, xmm3
    
    // Patch jump past defaults
    let after_defaults = code.len();
    let jmp_past_defaults_offset = (after_defaults as i32 - (jmp_past_defaults_pos + 5) as i32) as i32;
    code[jmp_past_defaults_pos + 1..jmp_past_defaults_pos + 5].copy_from_slice(&jmp_past_defaults_offset.to_le_bytes());
    
    // ============================================================================
    // PASS 1: Compute mean
    // ============================================================================
    // Save input length (rsi) to r11 for loop
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Zero accumulator: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass1_loop_start = code.len();
    
    // Check if done: cmp rcx, r11
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass1_jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate byte offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load input: movss xmm1, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Accumulate: addss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
    
    // Increment counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back: jmp pass1_loop_start
    let pass1_jmp_pos = code.len();
    code.push(0xE9);
    let pass1_jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch pass1 jumps
    let pass1_loop_end = code.len();
    let pass1_jge_offset = (pass1_loop_end as i32 - (pass1_jge_pos + 6) as i32) as i32;
    code[pass1_jge_pos + 2..pass1_jge_pos + 6].copy_from_slice(&pass1_jge_offset.to_le_bytes());
    let pass1_jmp_offset = (pass1_loop_start as i32 - (pass1_jmp_pos + 5) as i32) as i32;
    code[pass1_jmp_offset_placeholder..pass1_jmp_offset_placeholder + 4].copy_from_slice(&pass1_jmp_offset.to_le_bytes());
    
    // Compute mean: divss xmm0, [count]
    // Convert count to float: cvtsi2ss xmm1, r11
    code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]); // cvtsi2ss xmm1, r11
    // Divide sum by count: divss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
    // xmm0 now contains mean - save to xmm4
    code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0
    
    // ============================================================================
    // PASS 2: Compute variance
    // ============================================================================
    // Zero accumulator: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Reset loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass2_loop_start = code.len();
    
    // Check if done: cmp rcx, r11
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass2_jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate byte offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load input: movss xmm1, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Compute (input - mean): subss xmm1, xmm4
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xCC]);
    
    // Compute (input - mean)^2: mulss xmm1, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]);
    
    // Accumulate variance: addss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
    
    // Increment counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back: jmp pass2_loop_start
    let pass2_jmp_pos = code.len();
    code.push(0xE9);
    let pass2_jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch pass2 jumps
    let pass2_loop_end = code.len();
    let pass2_jge_offset = (pass2_loop_end as i32 - (pass2_jge_pos + 6) as i32) as i32;
    code[pass2_jge_pos + 2..pass2_jge_pos + 6].copy_from_slice(&pass2_jge_offset.to_le_bytes());
    let pass2_jmp_offset = (pass2_loop_start as i32 - (pass2_jmp_pos + 5) as i32) as i32;
    code[pass2_jmp_offset_placeholder..pass2_jmp_offset_placeholder + 4].copy_from_slice(&pass2_jmp_offset.to_le_bytes());
    
    // Compute variance: divss xmm0, [count]
    // Convert count to float: cvtsi2ss xmm1, r11
    code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]); // cvtsi2ss xmm1, r11
    // Divide sum by count: divss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
    // xmm0 now contains variance - save to xmm5
    code.extend_from_slice(&[0x0F, 0x28, 0xE8]); // movaps xmm5, xmm0
    
    // Compute std = sqrt(variance + epsilon): addss xmm5, xmm6; sqrtss xmm5, xmm5
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xEE]); // addss xmm5, xmm6
    code.extend_from_slice(&[0xF3, 0x0F, 0x51, 0xED]); // sqrtss xmm5, xmm5
    
    // ============================================================================
    // PASS 3: Normalize and apply scale/bias
    // ============================================================================
    // Reset loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass3_loop_start = code.len();
    
    // Check if done: cmp rcx, r11 (input length, same as output for BatchNorm)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass3_jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate byte offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load input: movss xmm0, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Compute (input - mean): subss xmm0, xmm4
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
    
    // Divide by std: divss xmm0, xmm5
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC5]);
    
    // Multiply by gamma: mulss xmm0, xmm2
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC2]);
    
    // Add beta: addss xmm0, xmm3
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]);
    
    // Store: movss [output_buffer_reg + rax], xmm0
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    // Increment counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back: jmp pass3_loop_start
    let pass3_jmp_pos = code.len();
    code.push(0xE9);
    let pass3_jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch pass3 jumps
    let pass3_loop_end = code.len();
    let pass3_jge_offset = (pass3_loop_end as i32 - (pass3_jge_pos + 6) as i32) as i32;
    code[pass3_jge_pos + 2..pass3_jge_pos + 6].copy_from_slice(&pass3_jge_offset.to_le_bytes());
    let pass3_jmp_offset = (pass3_loop_start as i32 - (pass3_jmp_pos + 5) as i32) as i32;
    code[pass3_jmp_offset_placeholder..pass3_jmp_offset_placeholder + 4].copy_from_slice(&pass3_jmp_offset.to_le_bytes());
    
    Ok(code)
}
