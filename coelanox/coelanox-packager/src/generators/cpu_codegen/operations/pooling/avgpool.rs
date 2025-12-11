//! AvgPool operation
//! Average pooling with 2x2 window and stride 2 - supports 1D and 2D

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_avgpool(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // AvgPool: 2x2 average pooling with stride 2
    // 1D: output[i] = (input[i*2] + input[i*2+1]) / 2
    // 2D: output[i,j] = average of 2x2 window starting at input[i*2, j*2]
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: 2x2 average pooling with stride 2
        // Input dimensions are 2x output dimensions
        let input_rows = output_rows * 2;
        let input_cols = output_cols * 2;
        
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Store output_cols in r9 (for output offset calculation)
        if output_cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC1);
            code.extend_from_slice(&(output_cols as u32).to_le_bytes());
        }
        
        // Store input_cols in r12 (for row stride calculation)
        if input_cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC4); code.push(input_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC4);
            code.extend_from_slice(&(input_cols as u32).to_le_bytes());
        }
        
        // Load constant 0.25 (for division by 4): mov eax, 0x3E800000; movd xmm7, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3E800000u32.to_le_bytes()); // 0.25f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        
        // Outer loop: output rows (rbx)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let outer_loop_start = code.len();
        
        // Check if done: cmp rbx, output_rows
        if output_rows <= 127 {
            code.push(0x48); code.push(0x83); code.push(0xFB); code.push(output_rows as u8);
        } else {
            code.push(0x48); code.push(0x81); code.push(0xFB);
            code.extend_from_slice(&(output_rows as u32).to_le_bytes());
        }
        
        let outer_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Inner loop: output cols (r8)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_loop_start = code.len();
        
        // Check if done: cmp r8, r9 (output_cols)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let inner_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate input base offset for 2x2 window:
        // input_row = rbx * 2, input_col = r8 * 2
        // offset = (input_row * input_cols + input_col) * 4
        // = ((rbx * 2) * input_cols + (r8 * 2)) * 4
        // = (rbx * input_cols * 2 + r8 * 2) * 4
        // = (rbx * input_cols + r8) * 8
        
        // Calculate: rax = rbx * input_cols + r8
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by input_cols: rax = rax * r12
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Mul_rm64, Register::R12).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Add r8: rax = rax + r8
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Multiply by 8: rax = rax * 8 (shl by 3)
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x03);
        
        // Add input buffer base: rax = rax + input_buffer_reg
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, input_buffer_reg).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Load 4 values from 2x2 window and compute average
        // Window layout:
        // [0,0] at [rax]
        // [0,1] at [rax + 4]
        // [1,0] at [rax + input_cols*4]
        // [1,1] at [rax + input_cols*4 + 4]
        
        // Load [0,0]: movss xmm0, [rax]
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]);
        
        // Load [0,1]: movss xmm1, [rax + 4]
        code.push(0x48); code.push(0x83); code.push(0xC0); code.push(0x04); // add rax, 4
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x08]); // movss xmm1, [rax]
        
        // Add: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Load [1,0]: need to go to rax + input_cols*4 - 4 (back to base, then add row stride)
        code.push(0x48); code.push(0x83); code.push(0xE8); code.push(0x04); // sub rax, 4 (back to base)
        let input_cols_bytes = input_cols * 4;
        if input_cols_bytes <= 127 {
            code.push(0x48); code.push(0x83); code.push(0xC0); code.push(input_cols_bytes as u8); // add rax, input_cols*4
        } else {
            code.push(0x48); code.push(0x05);
            code.extend_from_slice(&(input_cols_bytes as u32).to_le_bytes()); // add rax, input_cols*4
        }
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x08]); // movss xmm1, [rax]
        
        // Add: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Load [1,1]: movss xmm1, [rax + 4]
        code.push(0x48); code.push(0x83); code.push(0xC0); code.push(0x04); // add rax, 4
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x08]); // movss xmm1, [rax]
        
        // Add: addss xmm0, xmm1 (xmm0 now contains sum of all 4 values)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
        
        // Divide by 4: mulss xmm0, xmm7 (multiply by 0.25)
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC7]);
        
        // Calculate output offset: rax = (rbx * output_cols + r8) * 4
        code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
        
        // Store result: movss [output_buffer_reg + rax], xmm0
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
        let outer_jge_offset = (outer_loop_end as i32 - (outer_jmp_pos + 6) as i32) as i32;
        code[outer_jge_pos + 2..outer_jge_pos + 6].copy_from_slice(&outer_jge_offset.to_le_bytes());
        let outer_jmp_offset = (outer_loop_start as i32 - (outer_jmp_pos + 5) as i32) as i32;
        code[outer_jmp_placeholder..outer_jmp_placeholder + 4].copy_from_slice(&outer_jmp_offset.to_le_bytes());
        
        return Ok(code);
    }
    
    // 1D case: average of pairs
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load constant 0.5 (for division by 2): mov eax, 0x3F000000; movd xmm2, eax
    code.push(0xB8);
    code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
    
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
    
    // Calculate input offset: rax = rcx * 8 (rcx * 2 elements * 4 bytes)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x03); // shl rax, 3
    
    // Add input buffer base
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, input_buffer_reg).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Load first element: movss xmm0, [rax]
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]);
    
    // Load second element: movss xmm1, [rax + 4]
    code.push(0x48); code.push(0x83); code.push(0xC0); code.push(0x04); // add rax, 4
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x08]); // movss xmm1, [rax]
    
    // Add: addss xmm0, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]);
    
    // Divide by 2: mulss xmm0, xmm2
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC2]);
    
    // Calculate output offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
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
    
    Ok(code)
}
