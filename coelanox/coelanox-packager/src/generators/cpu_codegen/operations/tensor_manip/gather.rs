//! Gather operation
//! Gather elements by indices

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::tensor_ops;

pub fn generate_gather(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
        // Gather: gather elements by indices
        // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
        // For 1D: output[i] = input[indices[i]]
        // For 2D: proper gathering along axis 0 (rows) or axis 1 (cols)
        //   axis 0: output[row][col] = data[indices[row]][col]
        //   axis 1: output[row][col] = data[row][indices[col]]
        // Input 0: data tensor (the tensor to gather from)
        // Input 1: indices tensor (the indices to gather)
        
        // Check if we have at least 2 inputs (data and indices)
        if _node.input_tensors.len() < 2 {
            // No indices provided: just copy (correct for no-op gather)
            return tensor_ops::generate_reshape(generator, _node, _input_shape, _output_shape);
        }
        
        // Get buffer registers for data (input 0) and indices (input 1)
        let data_buffer_reg = generator.get_input_buffer_for_tensor(_node, 0)?;
        let indices_buffer_reg = generator.get_input_buffer_for_tensor(_node, 1)?;
        let output_buffer_reg = generator.get_current_output_buffer();
        
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
        
        // Check if we have 2D shapes (or higher, flattened to 2D)
        if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
            if let Some((input_rows, input_cols)) = generator.get_2d_dimensions(_input_shape) {
                // 2D case: proper gathering along axis
                let mut code = Vec::new();
                let mut encoder = Encoder::new(64);
                
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
                
                if axis == 0 {
                    // Gather along axis 0 (rows): output[row][col] = data[indices[row]][col]
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
                    
                    // Load index for this row: rax = indices[rbx]
                    // Calculate indices offset: rax = rbx * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
                    
                    // Load index (as i32) and sign-extend: movsxd rax, [indices_buffer_reg + rax]
                    match indices_buffer_reg {
                        Register::R10 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x02]), // movsxd rax, [r10 + rax]
                        Register::R11 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x03]), // movsxd rax, [r11 + rax]
                        Register::R13 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x05]), // movsxd rax, [r13 + rax]
                        Register::R14 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x06]), // movsxd rax, [r14 + rax]
                        _ => return Err(CoelanoxError::Internal("Unsupported indices buffer register".to_string())),
                    }
                    // Now rax = data_row (index from indices array)
                    // Save to r13
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
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
                    
                    // Calculate data offset: rax = (data_row * input_cols + r8) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R13).unwrap();
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
                    
                    // Load data: movss xmm0, [data_buffer_reg + r12]
                    match data_buffer_reg {
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
                        _ => return Err(CoelanoxError::Internal("Unsupported data buffer register".to_string())),
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
                    // Gather along axis 1 (cols): output[row][col] = data[row][indices[col]]
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
                    
                    // Load index for this col: rax = indices[r8]
                    // Calculate indices offset: rax = r8 * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
                    
                    // Load index (as i32) and sign-extend: movsxd rax, [indices_buffer_reg + rax]
                    match indices_buffer_reg {
                        Register::R10 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x02]), // movsxd rax, [r10 + rax]
                        Register::R11 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x03]), // movsxd rax, [r11 + rax]
                        Register::R13 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x05]), // movsxd rax, [r13 + rax]
                        Register::R14 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x06]), // movsxd rax, [r14 + rax]
                        _ => return Err(CoelanoxError::Internal("Unsupported indices buffer register".to_string())),
                    }
                    // Now rax = data_col (index from indices array)
                    // Save to r13
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Calculate data offset: rax = (rbx * input_cols + data_col) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
                    // Save to r12
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R12, Register::RAX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Load data: movss xmm0, [data_buffer_reg + r12]
                    match data_buffer_reg {
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
                        _ => return Err(CoelanoxError::Internal("Unsupported data buffer register".to_string())),
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
        
        // 1D case: linear gathering
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
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
        
        // Calculate indices byte offset: rax = rcx * 4 (indices are i32)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load index (as i32) and sign-extend to 64-bit: movsxd rax, [indices_buffer_reg + rax]
        match indices_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x02]), // movsxd rax, [r10 + rax]
            Register::R11 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x03]), // movsxd rax, [r11 + rax]
            Register::R13 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x05]), // movsxd rax, [r13 + rax]
            Register::R14 => code.extend_from_slice(&[0x48, 0x63, 0x04, 0x06]), // movsxd rax, [r14 + rax]
            _ => return Err(CoelanoxError::Internal("Unsupported indices buffer register".to_string())),
        }
        
        // Calculate data byte offset: rax = rax * 4 (index * 4 bytes per float)
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load data: movss xmm0, [data_buffer_reg + rax]
        match data_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported data buffer register".to_string())),
        }
        
        // Calculate output byte offset: rbx = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RBX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE3); code.push(0x02); // shl rbx, 2
        
        // Store: movss [output_buffer_reg + rbx], xmm0
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
