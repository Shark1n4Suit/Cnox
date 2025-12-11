//! Split operation
//! Split tensor into multiple outputs

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::tensor_ops;

pub fn generate_split(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
        // Split: split tensor into multiple outputs
        // For 1D: Split input into multiple output tensors
        // For 2D: proper split along axis 0 (rows) or axis 1 (cols)
        //   axis 0: split rows - output gets first portion of rows
        //   axis 1: split cols - output gets first portion of cols
        // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
        // TODO: Add support for multiple output buffers to handle all Split outputs
        //       Currently only writes to first output buffer (infrastructure limitation)
        //       Need to extend get_current_output_buffer() to support output_index parameter
        //       and iterate over all output_tensors to write each split portion
        
        // Check if we have multiple outputs
        if _node.output_tensors.len() <= 1 {
            // Single output: just copy (correct for single-output case)
            return tensor_ops::generate_reshape(generator, _node, _input_shape, _output_shape);
        }
        
        // Get axis from attributes (default 0)
        let axis = _node.attributes.get("axis")
            .and_then(|attr| {
                if let coelanox_core::ir::Attribute::Int(i) = attr {
                    Some(*i as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        
        // Get split attribute (optional, defaults to equal splits)
        // For now, we'll use the output shape to determine the split size
        
        // Check if we have 2D shapes (or higher, flattened to 2D)
        if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
            if let Some((input_rows, input_cols)) = generator.get_2d_dimensions(_input_shape) {
                // 2D case: proper split along axis
                let input_buffer_reg = generator.get_input_buffer_for_tensor(_node, 0)?;
                let output_buffer_reg = generator.get_current_output_buffer();
                let mut code = Vec::new();
                let mut encoder = Encoder::new(64);
                
                if axis == 0 {
                    // Split along axis 0 (rows): copy first output_rows rows
                    // Store output_cols in r9
                    if output_cols <= 127 {
                        code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
                    } else {
                        code.push(0x49); code.push(0xC7); code.push(0xC1);
                        code.extend_from_slice(&(output_cols as u32).to_le_bytes());
                    }
                    
                    // Store input_cols in r10 (should be same as output_cols for axis 0 split)
                    if input_cols <= 127 {
                        code.push(0x49); code.push(0xC7); code.push(0xC2); code.push(input_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
                    } else {
                        code.push(0x49); code.push(0xC7); code.push(0xC2);
                        code.extend_from_slice(&(input_cols as u32).to_le_bytes());
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
                    
                    // Inner loop: cols (r8)
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
                    
                    // Calculate input offset: rax = (rbx * input_cols + r8) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
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
                } else if axis == 1 {
                    // Split along axis 1 (cols): copy first output_cols columns
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
                    
                    // Outer loop: rows (rbx)
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
                    
                    // Calculate input offset: rax = (rbx * input_cols + r8) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
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
        }
        
        // 1D case: linear split
        let input_buffer_reg = generator.get_input_buffer_for_tensor(_node, 0)?;
        let output_buffer_reg = generator.get_current_output_buffer();
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Initialize loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::RSI).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate input byte offset: rax = rcx * 4
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
        
        // Calculate output byte offset: rax = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
        
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
