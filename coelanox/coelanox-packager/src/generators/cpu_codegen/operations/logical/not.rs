//! Not operation
//! Element-wise logical NOT: output = (input == 0.0) ? 1.0 : 0.0

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_not(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Not: output = (input == 0.0) ? 1.0 : 0.0
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        return generate_2d_logical_not(generator, input_buffer_reg, output_buffer_reg, rows, cols);
    }
    
    // 1D case: simple linear loop
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load constant 0.0 into xmm1
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1
    
    // Load constant 1.0 into xmm2 and broadcast to all lanes
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax
    code.extend_from_slice(&[0x0F, 0xC6, 0xD2, 0x00]); // shufps xmm2, xmm2, 0 (broadcast to all lanes)
    
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R15).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
    
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Compare input == 0.0: cmpeqss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x00]); // cmpeqss xmm0, xmm1
    
    // Convert mask to float: andps xmm0, xmm2
    code.extend_from_slice(&[0x0F, 0x28, 0xD8]); // movaps xmm3, xmm0 (save mask)
    code.extend_from_slice(&[0x0F, 0x54, 0xDA]); // andps xmm3, xmm2 (mask with 1.0)
    code.extend_from_slice(&[0x0F, 0x28, 0xC3]); // movaps xmm0, xmm3 (restore result)
    
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jmp_pos = code.len();
    code.push(0xE9);
    let jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    let loop_end = code.len();
    let jge_offset = (loop_end as i32 - (jge_pos + 6) as i32) as i32;
    code[jge_pos + 2..jge_pos + 6].copy_from_slice(&jge_offset.to_le_bytes());
    let jmp_offset = (loop_start as i32 - (jmp_pos + 5) as i32) as i32;
    code[jmp_offset_placeholder..jmp_offset_placeholder + 4].copy_from_slice(&jmp_offset.to_le_bytes());
    
    Ok(code)
}

/// Generate 2D logical NOT operation
fn generate_2d_logical_not(
    generator: &CpuCodeGenerator,
    input_buffer_reg: Register,
    output_buffer_reg: Register,
    rows: usize,
    cols: usize
) -> CoelanoxResult<Vec<u8>> {
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Store cols in r9
    if cols <= 127 {
        code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
    } else {
        code.push(0x49); code.push(0xC7); code.push(0xC1);
        code.extend_from_slice(&(cols as u32).to_le_bytes());
    }
    
    // Load constants: 0.0 in xmm1, 1.0 in xmm2
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
    
    // Outer loop: rows (rbx)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let outer_loop_start = code.len();
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
    
    // Compare input == 0.0: cmpeqss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x00]);
    
    // Convert mask to float: andps xmm0, xmm2
    code.extend_from_slice(&[0x0F, 0x28, 0xD8]); // movaps xmm3, xmm0 (save mask)
    code.extend_from_slice(&[0x0F, 0x54, 0xDA]); // andps xmm3, xmm2 (mask with 1.0)
    code.extend_from_slice(&[0x0F, 0x28, 0xC3]); // movaps xmm0, xmm3 (restore result)
    
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
    
    let inner_jmp_pos = code.len();
    code.push(0xE9);
    let inner_jmp_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
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
    
    let outer_jmp_pos = code.len();
    code.push(0xE9);
    let outer_jmp_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    let outer_loop_end = code.len();
    let outer_jge_offset = (outer_loop_end as i32 - (outer_jge_pos + 6) as i32) as i32;
    code[outer_jge_pos + 2..outer_jge_pos + 6].copy_from_slice(&outer_jge_offset.to_le_bytes());
    let outer_jmp_offset = (outer_loop_start as i32 - (outer_jmp_pos + 5) as i32) as i32;
    code[outer_jmp_placeholder..outer_jmp_placeholder + 4].copy_from_slice(&outer_jmp_offset.to_le_bytes());
    
    Ok(code)
}

