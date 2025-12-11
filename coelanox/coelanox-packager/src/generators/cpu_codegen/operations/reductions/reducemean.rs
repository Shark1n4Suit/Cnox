//! ReduceMean operation
//! Mean of all elements to scalar (1D) or row-wise mean (2D)

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_mean(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // ReduceMean: mean = sum / count
    // 1D: mean of all elements to single scalar
    // 2D: mean of each row to one value per row (row-wise reduction)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_input_shape) {
        // 2D case: row-wise reduction - mean of each row
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Store cols in r9 for offset calculation
        if cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC1);
            code.extend_from_slice(&(cols as u32).to_le_bytes());
        }
        
        // Store cols as float in xmm7 for division (cols per row)
        code.push(0xB8);
        code.extend_from_slice(&(cols as u32).to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xF8]); // cvtsi2ss xmm7, eax
        
        // Outer loop: iterate over rows (rbx = row index)
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
        
        // Zero accumulator for this row: pxor xmm0, xmm0
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
        
        // Inner loop: iterate over cols in current row (r8 = col index)
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
        
        // Calculate input offset: rax = (rbx * cols + r8) * 4
        code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
        
        // Load input value: movss xmm1, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Add to accumulator: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Increment inner loop counter: inc r8
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
        
        // After inner loop: xmm0 contains sum for current row
        // Divide by cols to get mean: divss xmm0, xmm7
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC7]); // divss xmm0, xmm7
        
        // Calculate output offset: rax = rbx * 4 (one float per row)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Store result: movss [output_buffer_reg + rax], xmm0
        match output_buffer_reg {
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
        }
        
        // Increment outer loop counter: inc rbx
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
    
    // 1D case: mean of all elements to single scalar
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Zero accumulator: pxor xmm0, xmm0
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]);
    
    // Save input length (rsi) to r11 for loop
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    
    // Check if done: cmp rcx, r11 (input length)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
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
    
    // Load input: movss xmm1, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x0C, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Add to accumulator: addss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
    
    // Increment loop counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back to loop start
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
    
    // Divide by count: convert r11 to float and divide
    code.push(0xB8);
    code.extend_from_slice(&(0u32).to_le_bytes()); // Will be patched
    // Actually, we need to load r11 as float - use cvtsi2ss
    // Save r11 to eax first
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xC8]); // cvtsi2ss xmm1, eax
    
    // Divide: divss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC1]);
    
    // Store result: movss [output_buffer_reg], xmm0
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    Ok(code)
}
