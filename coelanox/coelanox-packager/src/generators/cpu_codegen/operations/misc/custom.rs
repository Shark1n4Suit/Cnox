//! Custom operation
//! For now, custom operations are treated as pass-through (copy input to output)
//! This allows models with custom ops to generate code, though they won't execute correctly
//! TODO: Map common custom ops (Embedding, AttentionMatMul) to supported operations

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use iced_x86::{Code, Instruction, Register, Encoder};

pub fn generate_custom(generator: &CpuCodeGenerator, node: &GraphNode, input_shape: Option<&Shape>, output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Get custom operation name
    let custom_name = match &node.op_type {
        coelanox_core::ir::OpType::Custom(name) => name,
        _ => return Err(CoelanoxError::Internal("Expected Custom operation".to_string())),
    };
    
    // For now, implement as pass-through (copy input to output)
    // This allows code generation to succeed, though execution may not be correct
    // TODO: Map specific custom operations:
    //   - "Embedding" -> Gather operation
    //   - "AttentionMatMul" -> MatMul operation
    //   - etc.
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Get output length from shape or use input length
    let output_len = output_shape
        .map(|s| s.total_elements())
        .or_else(|| input_shape.map(|s| s.total_elements()))
        .unwrap_or(1);
    
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Simple copy loop: copy input to output
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    
    // Check if done: cmp rcx, r15 (output length)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R15).unwrap();
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
    
    // Load from input: movss xmm0, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Store to output: movss [output_buffer_reg + rax], xmm0
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
    
    // Jump back: jmp loop_start
    let jmp_pos = code.len();
    code.push(0xE9);
    let jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch jumps
    let loop_end = code.len();
    let jge_offset = (loop_end as i32 - (jge_pos + 6) as i32) as i32;
    code[jge_pos + 2..jge_pos + 6].copy_from_slice(&jge_offset.to_le_bytes());
    let jmp_offset = (loop_start as i32 - (jmp_pos + 5) as i32) as i32;
    code[jmp_offset_placeholder..jmp_offset_placeholder + 4].copy_from_slice(&jmp_offset.to_le_bytes());
    
    // Log warning about custom operation
    eprintln!("⚠️  Warning: Custom operation '{}' implemented as pass-through (copy). Execution may be incorrect.", custom_name);
    
    Ok(code)
}

