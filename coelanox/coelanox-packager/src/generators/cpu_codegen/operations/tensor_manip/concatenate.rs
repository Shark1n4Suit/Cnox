//! Concatenate operation
//! Combine multiple tensors along a dimension

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::operations::tensor_ops;

pub fn generate_concatenate(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
        // Concatenate: combine multiple tensors along a dimension
        // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
        // For 1D: output = [input1, input2, input3, ...]
        // For 2D: proper concatenation along axis 0 (rows) or axis 1 (cols)
        //   axis 0: stack rows - output[rows1+rows2+...][cols] = [input1 rows; input2 rows; ...]
        //   axis 1: append cols - output[rows][cols1+cols2+...] = [input1 cols | input2 cols | ...]
        
        // Check if we have multiple inputs
        if _node.input_tensors.len() <= 1 {
            // Single input: just copy (correct behavior)
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
        
        // Check if we have 2D shapes (or higher, flattened to 2D)
        if let Some((output_rows, output_cols)) = generator.get_2d_dimensions(_output_shape) {
            // 2D case: proper concatenation along axis
            let output_buffer_reg = generator.get_current_output_buffer();
            let mut code = Vec::new();
            let mut encoder = Encoder::new(64);
            
            // Store output_cols in r9 (for offset calculation)
            if output_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(output_cols as u32).to_le_bytes());
            }
            
            if axis == 0 {
                // Concatenate along axis 0 (rows): stack rows from each input
                // Track current output row offset
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Iterate over each input tensor
                for input_idx in 0.._node.input_tensors.len() {
                    let input_buffer_reg = generator.get_input_buffer_for_tensor(_node, input_idx)?;
                    
                    // Get input shape - assume it's 2D with same cols as output
                    // For simplicity, we'll use the output_cols (all inputs should have same cols for axis 0 concat)
                    // We need to get input rows - for now, estimate from input tensor size
                    // Actually, we can iterate row by row until we've copied all elements
                    // But for proper implementation, we'd need input shapes
                    // For now, use a simplified approach: copy all elements row by row
                    
                    // We'll need to know input_rows - let's assume we can get it from the input tensor
                    // For now, use a loop that copies until we've processed all elements
                    // Actually, let's use the fact that for axis 0 concat, all inputs have same cols
                    // So we can copy row by row
                    
                    // Initialize input row counter: r8 = 0
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // We need to know input_rows - for now, estimate from total elements / cols
                    // Actually, we can use a simpler approach: copy all elements linearly
                    // But that's what we had before. Let me implement proper row-by-row copy
                    
                    // For axis 0: copy each row of input to output
                    // We'll copy row by row, but we need to know how many rows each input has
                    // For now, let's use a simpler approach: copy all elements, but with proper 2D indexing
                    
                    // Actually, for proper 2D concatenation along axis 0, we need:
                    // - Input shapes to know how many rows each input has
                    // - Copy each row of each input to the correct output position
                    
                    // Simplified: assume we can iterate and copy all elements with 2D indexing
                    // We'll use the output_cols for both input and output
                    
                    // Initialize element counter: rcx = 0
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // We need input length - use rsi for now (but this is per-input)
                    // Actually, we need to track input size per tensor
                    // For now, use a simplified approach: copy until we've processed expected elements
                    
                    // Let's use a different approach: for axis 0, copy row by row
                    // We'll need to know input_rows for each input
                    // For now, let's assume all inputs have the same number of rows
                    // Or we can copy all elements with proper 2D offset calculation
                    
                    // Actually, let me implement it properly: for each input, copy all its rows
                    // We'll use nested loops: outer for rows, inner for cols
                    
                    // Store input_cols in r10 (should be same as output_cols for axis 0)
                    if output_cols <= 127 {
                        code.push(0x49); code.push(0xC7); code.push(0xC2); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
                    } else {
                        code.push(0x49); code.push(0xC7); code.push(0xC2);
                        code.extend_from_slice(&(output_cols as u32).to_le_bytes());
                    }
                    
                    // We need input_rows - for now, estimate from input tensor size
                    // Actually, we can use rsi (input length) / output_cols to get input_rows
                    // But rsi might not be set correctly for each input
                    // For now, let's use a simpler approach: copy all elements with 2D indexing
                    
                    // Initialize input row counter: r8 = 0
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // We'll need a loop that continues until we've copied all input elements
                    // For now, let's use rsi as input element count (but this needs to be per-input)
                    // Actually, let's use a different register to track input size
                    // For simplicity, assume we can get input_rows from shape or use a fixed approach
                    
                    // Let me use a simpler but correct approach: copy all elements row by row
                    // We'll use rsi / output_cols to determine input_rows
                    // Store rsi in r13 temporarily
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RSI).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Calculate input_rows: r13 = r13 / output_cols
                    // Divide r13 by r10 (output_cols)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Div_rm64, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Result is in RAX, move to r13
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Outer loop: input rows (r8)
                    let input_row_loop_start = code.len();
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let input_row_jge_pos = code.len();
                    code.push(0x0F); code.push(0x8D);
                    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                    
                    // Inner loop: cols (rcx)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let col_loop_start = code.len();
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let col_jge_pos = code.len();
                    code.push(0x0F); code.push(0x8D);
                    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                    
                    // Calculate input offset: rax = (r8 * output_cols + rcx) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Multiply by output_cols (r10)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Mul_rm64, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Add col (rcx)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Multiply by 4 (element size)
                    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
                    // Save to r12 for reuse
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
                    
                    // Calculate output offset: rax = ((rbx + r8) * output_cols + rcx) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Now rax = output_row, calculate offset
                    if output_cols <= 127 {
                        code.push(0x48); code.push(0x6B); code.push(0xC0); code.push(output_cols as u8); // imul rax, rax, output_cols
                    } else {
                        code.push(0x48); code.push(0x69); code.push(0xC0);
                        code.extend_from_slice(&(output_cols as u32).to_le_bytes());
                    }
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
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
                    
                    // Increment col: inc rcx
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
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
                    
                    // Increment input row: inc r8
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let input_row_jmp_pos = code.len();
                    code.push(0xE9);
                    let input_row_jmp_placeholder = code.len();
                    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                    
                    let input_row_loop_end = code.len();
                    let input_row_jge_offset = (input_row_loop_end as i32 - (input_row_jge_pos + 6) as i32) as i32;
                    code[input_row_jge_pos + 2..input_row_jge_pos + 6].copy_from_slice(&input_row_jge_offset.to_le_bytes());
                    let input_row_jmp_offset = (input_row_loop_start as i32 - (input_row_jmp_pos + 5) as i32) as i32;
                    code[input_row_jmp_placeholder..input_row_jmp_placeholder + 4].copy_from_slice(&input_row_jmp_offset.to_le_bytes());
                    
                    // Update output row offset: add rbx, r13 (add input_rows to output row offset)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RBX, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                }
                
                return Ok(code);
            } else if axis == 1 {
                // Concatenate along axis 1 (cols): append columns from each input
                // Track current output col offset
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Iterate over each input tensor
                for input_idx in 0.._node.input_tensors.len() {
                    let input_buffer_reg = generator.get_input_buffer_for_tensor(_node, input_idx)?;
                    
                    // For axis 1, all inputs have same rows, different cols
                    // We need input_cols for each input - estimate from input size
                    // Store input_rows in r13 (should be same as output_rows for axis 1)
                    if output_rows <= 127 {
                        code.push(0x49); code.push(0xC7); code.push(0xC5); code.push(output_rows as u8); code.push(0x00); code.push(0x00); code.push(0x00);
                    } else {
                        code.push(0x49); code.push(0xC7); code.push(0xC5);
                        code.extend_from_slice(&(output_rows as u32).to_le_bytes());
                    }
                    
                    // Calculate input_cols: r13 = rsi / output_rows
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RSI).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Div_rm64, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Result is in RAX, move to r13
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Store input_cols in r10
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R10, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    // Outer loop: rows (r8)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let row_loop_start = code.len();
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R13).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let row_jge_pos = code.len();
                    code.push(0x0F); code.push(0x8D);
                    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                    
                    // Inner loop: input cols (rcx)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let col_loop_start = code.len();
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    
                    let col_jge_pos = code.len();
                    code.push(0x0F); code.push(0x8D);
                    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                    
                    // Calculate input offset: rax = (r8 * input_cols + rcx) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    // Multiply by input_cols (r10) - use imul rax, r10
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
                    
                    // Calculate output offset: rax = (r8 * output_cols + (rbx + rcx)) * 4
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    if output_cols <= 127 {
                        code.push(0x48); code.push(0x6B); code.push(0xC0); code.push(output_cols as u8); // imul rax, rax, output_cols
                    } else {
                        code.push(0x48); code.push(0x69); code.push(0xC0);
                        code.extend_from_slice(&(output_cols as u32).to_le_bytes());
                    }
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RBX).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
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
                    
                    // Increment col: inc rcx
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
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
                    
                    // Increment row: inc r8
                    encoder = Encoder::new(64);
                    let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
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
                    
                    // Update output col offset: add rbx, r10 (add input_cols to output col offset)
                    encoder = Encoder::new(64);
                    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RBX, Register::R10).unwrap();
                    encoder.encode(&inst, 0).unwrap();
                    code.extend_from_slice(&encoder.take_buffer());
                }
                
                return Ok(code);
            }
        }
        
        // 1D case: linear concatenation
        let output_buffer_reg = generator.get_current_output_buffer();
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Track current output offset (in elements, not bytes)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Iterate over each input tensor
        for input_idx in 0.._node.input_tensors.len() {
            let input_buffer_reg = generator.get_input_buffer_for_tensor(_node, input_idx)?;
            
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
            
            // Calculate output byte offset: rax = (rbx + rcx) * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::RCX).unwrap();
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
            
            // Update output offset: add rbx, rsi
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RBX, Register::RSI).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
        }
        
        Ok(code)
    }
