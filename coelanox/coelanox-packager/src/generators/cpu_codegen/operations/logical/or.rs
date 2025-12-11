//! Or operation
//! Element-wise logical OR: output = (input_a != 0.0 || input_b != 0.0) ? 1.0 : 0.0

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_or(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Or: output = (input_a != 0.0 || input_b != 0.0) ? 1.0 : 0.0
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        return generate_2d_logical_or(generator, input_buffer_reg, output_buffer_reg, rows, cols);
    }
    
    // 1D case: simple linear loop
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load constant 0.0 into xmm2
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xD2]); // pxor xmm2, xmm2
    
    // Load constant 1.0 into xmm3
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD8]); // cvtsi2ss xmm3, eax
    
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
    
    // Compare input_a != 0.0
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC2, 0x04]); // cmpneqss xmm0, xmm2
    
    code.extend_from_slice(&[0x4D, 0x89, 0xF8]);
    code.extend_from_slice(&[0x49, 0xC1, 0xE0, 0x02]);
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::R8, input_buffer_reg).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.extend_from_slice(&[0x49, 0x01, 0xC0]);
    
    code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x08]);
    
    // Compare input_b != 0.0
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xCA, 0x04]); // cmpneqss xmm1, xmm2
    
    // OR the masks: orps xmm0, xmm1
    code.extend_from_slice(&[0x0F, 0x56, 0xC1]); // orps xmm0, xmm1
    
    // Convert mask to float: andps xmm0, xmm3
    code.extend_from_slice(&[0x0F, 0x54, 0xC3]); // andps xmm0, xmm3
    
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

/// Generate 2D logical OR operation
fn generate_2d_logical_or(
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
    
    // Store input_b offset
    let input_b_offset = rows * cols * 4;
    if input_b_offset <= 127 {
        code.push(0x49); code.push(0xC7); code.push(0xC4); code.push(input_b_offset as u8); code.push(0x00); code.push(0x00); code.push(0x00);
    } else {
        code.push(0x49); code.push(0xC7); code.push(0xC4);
        code.extend_from_slice(&(input_b_offset as u32).to_le_bytes());
    }
    
    // Load constants: 0.0 in xmm2, 1.0 in xmm3
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xD2]); // pxor xmm2, xmm2 (0.0)
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD8]); // cvtsi2ss xmm3, eax (1.0)
    
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
    
    // Load input_a: movss xmm0, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Compare input_a != 0.0: cmpneqss xmm0, xmm2
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC2, 0x04]);
    
    // Save rax, calculate input_b offset
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RDX, Register::RAX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RDX, Register::R12).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Load input_b: movss xmm1, [input_buffer_reg + rdx]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x12]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x13]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x15]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x16]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Compare input_b != 0.0: cmpneqss xmm1, xmm2
    code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xCA, 0x04]);
    
    // OR the masks: orps xmm0, xmm1
    code.extend_from_slice(&[0x0F, 0x56, 0xC1]);
    
    // Convert mask to float: andps xmm0, xmm3
    code.extend_from_slice(&[0x0F, 0x54, 0xC3]);
    
    // Restore rax (output offset)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RDX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Sub_rm64_r64, Register::RAX, Register::R12).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
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

