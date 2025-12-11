//! Broadcast operation
//! Replicate tensor to larger shape

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_broadcast(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
        // Broadcast: replicate tensor to larger shape
        // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
        // Simple implementation: replicate first element to all output positions
        
        let input_buffer_reg = generator.get_current_input_buffer();
        let output_buffer_reg = generator.get_current_output_buffer();
        
        // Check if we have 2D shapes (or higher, flattened to 2D)
        if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
            // 2D case: use 2D element-wise unary with constant replication
            let mut code = Vec::new();
            
            // Load first element from input into xmm0
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x02]), // movss xmm0, [r10]
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x03]), // movss xmm0, [r11]
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x05]), // movss xmm0, [r13]
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x06]), // movss xmm0, [r14]
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Use 2D element-wise unary helper to replicate xmm0 to all output positions
            // We'll generate a 2D loop that stores xmm0 to each position
            let mut encoder = Encoder::new(64);
            
            // Store cols in r9
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
            
            // Store xmm0 to output: movss [output_buffer_reg + rax], xmm0
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
            
            return Ok(code);
        }
        
        // 1D case: simple linear loop
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Load first element from input into xmm0
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x02]), // movss xmm0, [r10]
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x03]), // movss xmm0, [r11]
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x05]), // movss xmm0, [r13]
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x06]), // movss xmm0, [r14]
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Loop through output, writing the same value
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
        
        // Store xmm0 to output[rcx]
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
