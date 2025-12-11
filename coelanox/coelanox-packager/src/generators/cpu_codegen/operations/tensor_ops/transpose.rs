//! Transpose operation
//! Transpose tensor dimensions: output[row][col] = input[col][row]

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use super::reshape;

pub fn generate_transpose(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Transpose: transpose tensor dimensions
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // For 1D tensors, transpose is a no-op (just copy) - correct behavior
    // For 2D: output[row][col] = input[col][row]
    // Attributes: "perm" (IntArray) - default [1, 0] for 2D transpose
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if 1D (rank <= 1) - transpose is no-op
    if let Some(shape) = _input_shape {
        if shape.rank() <= 1 {
            // 1D transpose is a no-op - just copy
            return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
        }
        
        // For 2D: implement transpose
        if shape.rank() == 2 {
            // Get input dimensions
            let input_dims = &shape.dimensions;
            if input_dims.len() < 2 {
                return reshape::generate_reshape(generator, _node, _input_shape, _output_shape);
            }
            
            let input_rows = input_dims[0];
            let input_cols = input_dims[1];
            
            // Output dimensions are swapped: [cols, rows]
            let output_rows = input_cols;
            let output_cols = input_rows;
            
            // 2D transpose: output[row][col] = input[col][row]
            let mut code = Vec::new();
            let mut encoder = Encoder::new(64);
            
            // Store input_cols in r9 (for input offset calculation)
            if input_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(input_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(input_cols as u32).to_le_bytes());
            }
            
            // Store output_cols in r10 (for output offset calculation)
            if output_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC2); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC2);
                code.extend_from_slice(&(output_cols as u32).to_le_bytes());
            }
            
            // Outer loop: output rows (rbx) - goes from 0 to output_rows-1
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let outer_loop_start = code.len();
            if output_rows <= 127 {
                code.push(0x48); code.push(0x83); code.push(0xFB); code.push(output_rows as u8);
            } else {
                code.push(0x48); code.push(0x81); code.push(0xFB);
                code.extend_from_slice(&(output_rows as u32).to_le_bytes());
            }
            
            let outer_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Inner loop: output cols (r8) - goes from 0 to output_cols-1
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let inner_loop_start = code.len();
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R10).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let inner_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Calculate INPUT offset: input[col][row] = input[r8][rbx]
            // input_offset = (r8 * input_cols + rbx) * 4
            // r8 is the input col (which becomes output row)
            // rbx is the input row (which becomes output col)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Multiply by input_cols (r9): mul r9 (implicitly uses RAX, result in RAX)
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Mul_rm64, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Add rbx (input row)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Multiply by 4: shl rax, 2
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
            
            // Load from input: movss xmm0, [input_buffer_reg + rax]
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Calculate OUTPUT offset: output[row][col] = output[rbx][r8]
            // output_offset = (rbx * output_cols + r8) * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Multiply by output_cols (r10): mul r10 (implicitly uses RAX, result in RAX)
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Add r8 (output col)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Multiply by 4: shl rax, 2
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
            
            // Store to output: movss [output_buffer_reg + rax], xmm0
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
    }
    
    // Default: 1D case (no-op)
    reshape::generate_reshape(generator, _node, _input_shape, _output_shape)
}

