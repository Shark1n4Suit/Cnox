//! Shared helper functions for comparison operations

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::Shape;
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

/// Generate comparison binary operation (1D and 2D)
/// cmp_op: 0=equal, 1=notequal, 2=less, 3=lessequal, 4=greater, 5=greaterequal
pub(crate) fn generate_comparison_binary(generator: &CpuCodeGenerator, cmp_op: u8, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        return generate_2d_comparison_binary(generator, input_buffer_reg, output_buffer_reg, rows, cols, cmp_op);
    }
        
        // 1D case: simple linear loop
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Initialize loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Load constant 1.0 into xmm2 before the loop
        code.push(0xB8); // mov eax, 1
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax
        
        let loop_start = code.len();
        
        // Check if done: cmp rcx, r15
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
        
        // Load input_a: movss xmm0, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Calculate input_b address: r8 = r15 * 4 + input_buffer_reg + rax
        code.extend_from_slice(&[0x4D, 0x89, 0xF8]); // mov r8, r15
        code.extend_from_slice(&[0x49, 0xC1, 0xE0, 0x02]); // shl r8, 2
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::R8, input_buffer_reg).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.extend_from_slice(&[0x49, 0x01, 0xC0]); // add r8, rax
        
        // Load input_b: movss xmm1, [r8]
        code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x08]);
        
        // Perform comparison based on cmp_op
        match cmp_op {
            0 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x00]), // cmpeqss xmm0, xmm1
            1 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x04]), // cmpneqss xmm0, xmm1
            2 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x01]), // cmpltss xmm0, xmm1
            3 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x02]), // cmpless xmm0, xmm1
            4 => {
                // Greater: swap operands and use less
                code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC8, 0x01]); // cmpltss xmm1, xmm0 (b < a means a > b)
                code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1 (move result to xmm0)
            }
            5 => {
                // GreaterEqual: swap operands and use lessequal
                code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC8, 0x02]); // cmpless xmm1, xmm0 (b <= a means a >= b)
                code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
            }
            _ => return Err(CoelanoxError::Internal("Invalid comparison op".to_string())),
        }
        
        // Convert mask to float: andps xmm0, xmm2
        code.extend_from_slice(&[0x0F, 0x54, 0xC2]); // andps xmm0, xmm2
        
        // Store result: movss [output_buffer_reg + rax], xmm0
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
        
        Ok(code)
}

/// Generate 2D comparison binary operation
pub(crate) fn generate_2d_comparison_binary(
    generator: &CpuCodeGenerator,
    input_buffer_reg: Register,
    output_buffer_reg: Register,
    rows: usize,
    cols: usize,
    cmp_op: u8
) -> CoelanoxResult<Vec<u8>> {
        // 2D comparison with mask-to-float conversion
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
        
        // Load constant 1.0 into xmm2 and broadcast to all lanes
        code.push(0xB8);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax
        code.extend_from_slice(&[0x0F, 0xC6, 0xD2, 0x00]); // shufps xmm2, xmm2, 0 (broadcast to all lanes)
        
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
        
        // Perform comparison
        match cmp_op {
            0 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x00]), // cmpeqss
            1 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x04]), // cmpneqss
            2 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x01]), // cmpltss
            3 => code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC1, 0x02]), // cmpless
            4 => {
                code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC8, 0x01]); // cmpltss xmm1, xmm0
                code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
            }
            5 => {
                code.extend_from_slice(&[0xF3, 0x0F, 0xC2, 0xC8, 0x02]); // cmpless xmm1, xmm0
                code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
            }
            _ => return Err(CoelanoxError::Internal("Invalid comparison op".to_string())),
        }
        
        // Convert mask to float: andps xmm0, xmm2
        code.extend_from_slice(&[0x0F, 0x54, 0xC2]); // andps xmm0, xmm2
        
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

