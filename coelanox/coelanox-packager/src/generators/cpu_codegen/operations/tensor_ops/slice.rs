//! Slice operation
//! Extract portion of tensor with start/end/stride

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;

pub fn generate_slice(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Slice: extract portion of tensor
    // Slice extracts elements from start to end (exclusive) with optional stride
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // For 1D: output[i] = input[start + i * stride] for i in 0..output_len
    // For 2D: output[row][col] = input[start_row + row * stride_row][start_col + col * stride_col]
    // Attributes: "starts" (Vec<Int>), "ends" (Vec<Int>), "axes" (Vec<Int>), "steps" (Vec<Int>)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
        if let Some((input_rows, input_cols)) = generator.get_2d_dimensions(_input_shape) {
            // 2D case: proper slicing with row/col start/end/stride
            let mut code = Vec::new();
            let mut encoder = Encoder::new(64);
            
            // Get starts array (default [0, 0])
            let starts = _node.attributes.get("starts")
                .and_then(|attr| {
                    if let coelanox_core::ir::Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&i| i as usize).collect::<Vec<_>>())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![0, 0]);
            let start_row = starts.get(0).copied().unwrap_or(0);
            let start_col = starts.get(1).copied().unwrap_or(0);
            
            // Get steps array (default [1, 1])
            let steps = _node.attributes.get("steps")
                .and_then(|attr| {
                    if let coelanox_core::ir::Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&i| i as usize).collect::<Vec<_>>())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![1, 1]);
            let stride_row = steps.get(0).copied().unwrap_or(1);
            let stride_col = steps.get(1).copied().unwrap_or(1);
            
            // Store output_cols in r9
            if output_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(output_cols as u32).to_le_bytes());
            }
            
            // Store input_cols in r10
            if input_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC2); code.push(input_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC2);
                code.extend_from_slice(&(input_cols as u32).to_le_bytes());
            }
            
            // Store start_row in r13
            if start_row <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC5); code.push(start_row as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC5);
                code.extend_from_slice(&(start_row as u32).to_le_bytes());
            }
            
            // Store start_col in r12
            if start_col <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC4); code.push(start_col as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC4);
                code.extend_from_slice(&(start_col as u32).to_le_bytes());
            }
            
            // Outer loop: output rows (rbx)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let row_loop_start = code.len();
            // Compare rbx with output_rows
            if output_rows <= 127 {
                code.push(0x48); code.push(0x83); code.push(0xFB); code.push(output_rows as u8); // cmp rbx, output_rows
            } else {
                code.push(0x48); code.push(0x81); code.push(0xFB);
                code.extend_from_slice(&(output_rows as u32).to_le_bytes());
            }
            
            let row_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Inner loop: output cols (r8)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let col_loop_start = code.len();
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let col_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Calculate input row: rax = start_row + rbx * stride_row
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            if stride_row == 1 {
                // No multiplication needed
            } else if stride_row == 2 {
                code.push(0x48); code.push(0xD1); code.push(0xE0); // shl rax, 1
            } else if stride_row == 4 {
                code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
            } else {
                // General case: multiply by stride_row
                // Store stride_row in a temporary register (use r15, but save it first)
                // Actually, use rcx temporarily (we'll overwrite it anyway)
                code.push(0x48); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(stride_row as u32).to_le_bytes());
                encoder = Encoder::new(64);
                let inst = Instruction::with1(Code::Mul_rm64, Register::RCX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
            }
            
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R13).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Now rax = input_row
            
            // Calculate input col: rcx = start_col + r8 * stride_col
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RCX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            if stride_col == 1 {
                // No multiplication needed
            } else if stride_col == 2 {
                code.push(0x48); code.push(0xD1); code.push(0xE1); // shl rcx, 1
            } else if stride_col == 4 {
                code.push(0x48); code.push(0xC1); code.push(0xE1); code.push(0x02); // shl rcx, 2
            } else {
                // General case: multiply by stride_col
                // Store stride_col in r15 temporarily
                code.push(0x49); code.push(0xC7); code.push(0xC7);
                code.extend_from_slice(&(stride_col as u32).to_le_bytes());
                encoder = Encoder::new(64);
                let inst = Instruction::with1(Code::Mul_rm64, Register::R15).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
            }
            
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RCX, Register::R12).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            // Now rcx = input_col
            
            // Calculate input offset: rax = (input_row * input_cols + input_col) * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
            // Save to r12
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R12, Register::RAX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // Load input: movss xmm0, [input_buffer_reg + r12]
            match input_buffer_reg {
                Register::R10 => {
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
                },
                Register::R11 => {
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R11).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
                },
                Register::R13 => {
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
                },
                Register::R14 => {
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R14).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
                },
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Calculate output offset: rax = (rbx * output_cols + r8) * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Mul_rm64, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
            
            // Store: movss [output_buffer_reg + rax], xmm0
            match output_buffer_reg {
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
            }
            
            // Increment col: inc r8
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let col_jmp_pos = code.len();
            code.push(0xE9);
            let col_jmp_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            let col_loop_end = code.len();
            let col_jge_offset = (col_loop_end as i32 - (col_jge_pos + 6) as i32) as i32;
            code[col_jge_pos + 2..col_jge_pos + 6].copy_from_slice(&col_jge_offset.to_le_bytes());
            let col_jmp_offset = (col_loop_start as i32 - (col_jmp_pos + 5) as i32) as i32;
            code[col_jmp_placeholder..col_jmp_placeholder + 4].copy_from_slice(&col_jmp_offset.to_le_bytes());
            
            // Increment row: inc rbx
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Inc_rm64, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let row_jmp_pos = code.len();
            code.push(0xE9);
            let row_jmp_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            let row_loop_end = code.len();
            let row_jge_offset = (row_loop_end as i32 - (row_jge_pos + 6) as i32) as i32;
            code[row_jge_pos + 2..row_jge_pos + 6].copy_from_slice(&row_jge_offset.to_le_bytes());
            let row_jmp_offset = (row_loop_start as i32 - (row_jmp_pos + 5) as i32) as i32;
            code[row_jmp_placeholder..row_jmp_placeholder + 4].copy_from_slice(&row_jmp_offset.to_le_bytes());
            
            return Ok(code);
        }
    }
    
    // 1D case: linear slicing
    
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Get start index from attributes (default 0)
    let start = _node.attributes.get("starts")
        .and_then(|attr| {
            if let coelanox_core::ir::Attribute::IntArray(arr) = attr {
                arr.first().map(|&i| i as usize)
            } else if let coelanox_core::ir::Attribute::Int(i) = attr {
                Some(*i as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);
    
    // Get end index from attributes (default input_len)
    let _end = _node.attributes.get("ends")
        .and_then(|attr| {
            if let coelanox_core::ir::Attribute::IntArray(arr) = attr {
                arr.first().map(|&i| i as usize)
            } else if let coelanox_core::ir::Attribute::Int(i) = attr {
                Some(*i as usize)
            } else {
                None
            }
        })
        .or_else(|| {
            // If no end specified, use input length from shape
            _input_shape.map(|s| s.total_elements())
        })
        .unwrap_or(0);
    
    // Get stride from attributes (default 1)
    let stride = _node.attributes.get("steps")
        .and_then(|attr| {
            if let coelanox_core::ir::Attribute::IntArray(arr) = attr {
                arr.first().map(|&i| i as usize)
            } else if let coelanox_core::ir::Attribute::Int(i) = attr {
                Some(*i as usize)
            } else {
                None
            }
        })
        .unwrap_or(1);
    
    // Initialize loop counter: rcx = 0 (output index)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R15).unwrap(); // Compare with output length
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate input index: rax = start + rcx * stride
    // First: rax = rcx * stride
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    if stride == 1 {
        // No multiplication needed
    } else if stride == 2 {
        code.push(0x48); code.push(0xD1); code.push(0xE0); // shl rax, 1
    } else if stride == 4 {
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    } else {
        // General case: multiply by stride
        // mov rbx, stride; mul rbx (mul is implicit with rax)
        code.push(0x48); code.push(0xC7); code.push(0xC3);
        code.extend_from_slice(&(stride as u32).to_le_bytes());
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Mul_rm64, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
    }
    
    // Add start: rax = rax + start
    if start > 0 {
        if start <= 127 {
            code.push(0x48); code.push(0x83); code.push(0xC0); code.push(start as u8);
        } else {
            code.push(0x48); code.push(0x05);
            code.extend_from_slice(&(start as u32).to_le_bytes());
        }
    }
    
    // Calculate byte offset: rax = rax * 4
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load from input: movss xmm0, [input_buffer_reg + rax]
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Calculate output byte offset: rbx = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RBX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE3); code.push(0x02); // shl rbx, 2
    
    // Store to output: movss [output_buffer_reg + rbx], xmm0
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x1B]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x1D]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x1E]),
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
    
    Ok(code)
}

