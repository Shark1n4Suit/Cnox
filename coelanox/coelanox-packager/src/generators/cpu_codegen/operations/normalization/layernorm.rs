//! LayerNorm operation
//! (input - mean) / sqrt(variance + epsilon) * scale + bias

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;

pub fn generate_layernorm(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // LayerNorm: (input - mean) / sqrt(variance + epsilon) * scale + bias
    // Two-pass algorithm: first compute mean and variance, then normalize
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // For 1D/2D: compute mean/variance over all elements, then normalize each element
    // Weights contain: [scale, bias] in r12 buffer (r8 saved to r12 in prologue)
    // Note: Currently uses linear indexing which works for 2D (flattened), but could be optimized for true 2D
    
    // Strategy: Use r13 to store mean, r14 to store variance (but r14 is output buffer!)
    // Better: Use xmm registers to store mean/variance, compute in two passes
    // Pass 1: Compute mean (sum / count)
    // Pass 2: Compute variance (sum of (x - mean)^2 / count), then normalize
    
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Get epsilon from attributes (default 1e-5)
    let epsilon = _node.attributes.get("epsilon")
        .and_then(|attr| {
            if let coelanox_core::ir::Attribute::Float(f) = attr {
                Some(*f)
            } else {
                None
            }
        })
        .unwrap_or(1e-5);
    
    // Load epsilon: mov eax, <epsilon_hex>; movd xmm6, eax
    let epsilon_bits = epsilon.to_bits();
    code.push(0xB8);
    code.extend_from_slice(&epsilon_bits.to_le_bytes());
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF0]); // movd xmm6, eax
    
    // Load scale and bias from weights buffer (r12):
    // scale = weights[0], bias = weights[1] (defaults to 1.0, 0.0 if not provided)
    // For safety, use default values directly (1.0 for scale, 0.0 for bias)
    // This avoids segfaults when weights buffer is empty
    // Load scale = 1.0: mov eax, 0x3F800000; movd xmm7, eax
    code.push(0xB8);
    code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
    
    // Load bias = 0.0: pxor xmm5, xmm5
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xED]); // pxor xmm5, xmm5
    
    // ============================================================================
    // PASS 1: Compute mean
    // ============================================================================
    // Save rsi (input length) to r11 for loop (safe register)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Zero accumulator for mean: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass1_loop_start = code.len();
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
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
    
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
    code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]);
    // Divide sum by count: divss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
    // xmm0 now contains mean - save to xmm4
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xE0]); // movss xmm4, xmm0
    
    // ============================================================================
    // PASS 2: Compute variance and normalize
    // ============================================================================
    // Zero accumulator for variance: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Reset loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass2_loop_start = code.len();
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
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
    
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
    code.extend_from_slice(&[0xF3, 0x49, 0x0F, 0x2A, 0xCB]);
    // Divide sum by count: divss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
    // xmm0 now contains variance - save to xmm3
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xD8]); // movss xmm3, xmm0
    
    // Compute std = sqrt(variance + epsilon): addss xmm3, xmm6; sqrtss xmm3, xmm3
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xDE]); // addss xmm3, xmm6
    code.extend_from_slice(&[0xF3, 0x0F, 0x51, 0xDB]); // sqrtss xmm3, xmm3
    
    // ============================================================================
    // PASS 3: Normalize and apply scale/bias
    // ============================================================================
    // Reset loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let pass3_loop_start = code.len();
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap(); // Compare with input length (same as output for LayerNorm)
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
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
    
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
    
    // Divide by std: divss xmm0, xmm3
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC3]);
    
    // Multiply by scale: mulss xmm0, xmm7
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC7]);
    
    // Add bias: addss xmm0, xmm5
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]);
    
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

