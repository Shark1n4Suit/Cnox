//! Scatter operation
//! Scatter elements by indices: output[indices[i]] = values[i]

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_scatter(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Scatter: output[indices[i]] = values[i]
    // For 1D: output[indices[i]] = values[i]
    // For 2D: proper scattering along axis 0 (rows) or axis 1 (cols)
    // Test setup: input = destination, weights = [values..., indices...]
    // Weights buffer (r12) contains: [values[0], values[1], ..., indices[0], indices[1], ...]
    
    let input_buffer_reg = generator.get_current_input_buffer();
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
            // 2D case: proper scattering along axis
            let mut code = Vec::new();
            let mut encoder = Encoder::new(64);
            
            // Store output_cols in r9
            if output_cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(output_cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(output_cols as u32).to_le_bytes());
            }
            
            // First, copy input (destination) to output
            // Then scatter values from weights buffer
            
            // Calculate total elements
            let total_elements = input_rows * input_cols;
            
            // Copy input to output first (simple memcpy loop)
            // Initialize loop counter: rcx = 0
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let copy_loop_start = code.len();
            
            // Check if done: cmp rcx, total_elements
            if total_elements <= 127 {
                code.push(0x48); code.push(0x83); code.push(0xF9); code.push(total_elements as u8);
            } else {
                code.push(0x48); code.push(0x81); code.push(0xF9);
                code.extend_from_slice(&(total_elements as u32).to_le_bytes());
            }
            
            let copy_jge_pos = code.len();
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
            
            // Jump back: jmp copy_loop_start
            let copy_jmp_pos = code.len();
            code.push(0xE9);
            let copy_jmp_offset_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Patch copy loop jumps
            let copy_loop_end = code.len();
            let copy_jge_offset = (copy_loop_end as i32 - (copy_jge_pos + 6) as i32) as i32;
            code[copy_jge_pos + 2..copy_jge_pos + 6].copy_from_slice(&copy_jge_offset.to_le_bytes());
            let copy_jmp_offset = (copy_loop_start as i32 - (copy_jmp_pos + 5) as i32) as i32;
            code[copy_jmp_offset_placeholder..copy_jmp_offset_placeholder + 4].copy_from_slice(&copy_jmp_offset.to_le_bytes());
            
            // Check if weights buffer is available (r9 >= 2 * total_elements for values + indices)
            // If not, just return (copy already done above)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            code.push(0x48); code.push(0x01); code.push(0xC0); // add rax, rax (rax = 2 * total_elements)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R9, Register::RAX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let jb_no_weights_2d_pos = code.len();
            code.push(0x0F); code.push(0x82);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Now scatter values from weights buffer
            // Weights layout: [values[0..total_elements], indices[0..total_elements]]
            // Store values_offset = 0, indices_offset = total_elements * 4 in r10, r11
            let indices_offset = total_elements * 4;
            
            // Store indices_offset in r11
            if indices_offset <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC3); code.push(indices_offset as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC3);
                code.extend_from_slice(&(indices_offset as u32).to_le_bytes());
            }
            
            if axis == 0 {
                // Scatter along axis 0 (rows): output[indices[row]][col] = values[row][col]
                // Outer loop: input rows (rbx)
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                let row_loop_start = code.len();
                
                // Check if done: cmp rbx, input_rows
                if input_rows <= 127 {
                    code.push(0x48); code.push(0x83); code.push(0xFB); code.push(input_rows as u8);
                } else {
                    code.push(0x48); code.push(0x81); code.push(0xFB);
                    code.extend_from_slice(&(input_rows as u32).to_le_bytes());
                }
                
                let row_jge_pos = code.len();
                code.push(0x0F); code.push(0x8D);
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Load index for this row: rax = weights[r11 + rbx * 4] (indices offset + row * 4)
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R11).unwrap(); // Add indices_offset
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap(); // Add weights base (r12)
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Load index (as float) and convert to int: movss xmm0, [rax]; cvttss2si rax, xmm0
                code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
                code.extend_from_slice(&[0xF3, 0x48, 0x0F, 0x2C, 0xC0]); // cvttss2si rax, xmm0 (convert float to int)
                // Save to r13 (output_row) - use r13 instead of r11 to avoid conflict
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Bounds check: ensure output_row is within [0, output_rows)
                // Compare output_row (r13) with output_rows
                if output_rows <= 127 {
                    code.push(0x49); code.push(0x83); code.push(0xFD); code.push(output_rows as u8); // cmp r13, output_rows
                } else {
                    code.push(0x49); code.push(0x81); code.push(0xFD);
                    code.extend_from_slice(&(output_rows as u32).to_le_bytes());
                }
                let jae_skip_row_pos = code.len();
                code.push(0x0F); code.push(0x83);
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Also check if output_row < 0: test r13, r13; js skip
                code.push(0x49); code.push(0x85); code.push(0xED); // test r13, r13
                let js_skip_row_pos = code.len();
                code.push(0x0F); code.push(0x88);
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Inner loop: input cols (r8)
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                let col_loop_start = code.len();
                
                // Check if done: cmp r8, r9 (output_cols)
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                let col_jge_pos = code.len();
                code.push(0x0F); code.push(0x8D);
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Calculate values offset: rax = (rbx * input_cols + r8) * 4
                code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
                // Add weights base (r12) - values are at start of weights buffer
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Load value: movss xmm0, [rax]
                code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]);
                
                // Calculate output offset: rax = (output_row * output_cols + r8) * 4
                // Use r13 for output_row (saved above)
                encoder = Encoder::new(64);
                let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R13).unwrap(); // Use r13 (saved output_row)
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Multiply by output_cols: rax = rax * r9
                encoder = Encoder::new(64);
                let inst = Instruction::with1(Code::Mul_rm64, Register::R9).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Add col: rax = rax + r8
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
                
                // Skip point (if row index out of bounds)
                let skip_row_store_pos = code.len();
                let jae_skip_row_offset = (skip_row_store_pos as i32 - (jae_skip_row_pos + 6) as i32) as i32;
                code[jae_skip_row_pos + 2..jae_skip_row_pos + 6].copy_from_slice(&jae_skip_row_offset.to_le_bytes());
                let js_skip_row_offset = (skip_row_store_pos as i32 - (js_skip_row_pos + 6) as i32) as i32;
                code[js_skip_row_pos + 2..js_skip_row_pos + 6].copy_from_slice(&js_skip_row_offset.to_le_bytes());
                
                // Increment col: inc r8
                encoder = Encoder::new(64);
                let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
                encoder.encode(&inst, 0).unwrap();
                code.extend_from_slice(&encoder.take_buffer());
                
                // Jump back to col loop start
                let col_jmp_pos = code.len();
                code.push(0xE9);
                let col_jmp_placeholder = code.len();
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Patch col loop jumps
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
                
                // Jump back to row loop start
                let row_jmp_pos = code.len();
                code.push(0xE9);
                let row_jmp_placeholder = code.len();
                code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
                
                // Patch row loop jumps
                let row_loop_end = code.len();
                let row_jge_offset = (row_loop_end as i32 - (row_jge_pos + 6) as i32) as i32;
                code[row_jge_pos + 2..row_jge_pos + 6].copy_from_slice(&row_jge_offset.to_le_bytes());
                let row_jmp_offset = (row_loop_start as i32 - (row_jmp_pos + 5) as i32) as i32;
                code[row_jmp_placeholder..row_jmp_placeholder + 4].copy_from_slice(&row_jmp_offset.to_le_bytes());
                
                // No weights case: just return (copy already done above)
                let no_weights_2d_start = code.len();
                let jb_no_weights_2d_offset = (no_weights_2d_start as i32 - (jb_no_weights_2d_pos + 6) as i32) as i32;
                code[jb_no_weights_2d_pos + 2..jb_no_weights_2d_pos + 6].copy_from_slice(&jb_no_weights_2d_offset.to_le_bytes());
                
                return Ok(code);
            } else if axis == 1 {
                // Scatter along axis 1 (cols): output[row][indices[col]] = values[row][col]
                // Similar structure but scatter along columns
                // For simplicity, implement similar to axis 0 but with column indices
                return Err(CoelanoxError::Internal("Scatter axis 1 not yet implemented".to_string()));
            }
        }
    }
    
    // 1D case: output[indices[i]] = values[i]
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Save input length (rsi) to r11 for loop
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Check if weights buffer is available (r9 >= 2 * input_len for values + indices)
    // If not, just copy input to output
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0x01); code.push(0xC0); // add rax, rax (rax = 2 * input_len)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R9, Register::RAX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jb_no_weights_pos = code.len();
    code.push(0x0F); code.push(0x82);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // First, copy input (destination) to output
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let copy_loop_start = code.len();
    
    // Check if done: cmp rcx, r11
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let copy_jge_pos = code.len();
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
    
    // Jump back: jmp copy_loop_start
    let copy_jmp_pos = code.len();
    code.push(0xE9);
    let copy_jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch copy loop jumps
    let copy_loop_end = code.len();
    let copy_jge_offset = (copy_loop_end as i32 - (copy_jge_pos + 6) as i32) as i32;
    code[copy_jge_pos + 2..copy_jge_pos + 6].copy_from_slice(&copy_jge_offset.to_le_bytes());
    let copy_jmp_offset = (copy_loop_start as i32 - (copy_jmp_pos + 5) as i32) as i32;
    code[copy_jmp_offset_placeholder..copy_jmp_offset_placeholder + 4].copy_from_slice(&copy_jmp_offset.to_le_bytes());
    
    // Now scatter values from weights buffer
    // Weights layout: [values[0..len], indices[0..len]]
    // Store indices_offset = len * 4 in r10 (use r10 instead of r11 to avoid conflicts)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2 (indices_offset)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R10, Register::RAX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Initialize scatter loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Save input length to rbx for scatter loop
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RBX, Register::RSI).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let scatter_loop_start = code.len();
    
    // Check if done: cmp rcx, rbx
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::RBX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let scatter_jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Load index: rax = weights[r10 + rcx * 4] (indices_offset + index * 4)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R10).unwrap(); // Add indices_offset
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap(); // Add weights base (r12)
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Load index (as float) and convert to int: movss xmm0, [rax]; cvttss2si rax, xmm0
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]); // movss xmm0, [rax]
    code.extend_from_slice(&[0xF3, 0x48, 0x0F, 0x2C, 0xC0]); // cvttss2si rax, xmm0 (convert float to int)
    // Save to r13 (output_index)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R13, Register::RAX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Bounds check: ensure output_index is within [0, output_len)
    // Compare output_index (r13) with output_len (r15)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R13, Register::R15).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    let jae_skip_pos = code.len();
    code.push(0x0F); code.push(0x83);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Also check if output_index < 0: test r13, r13; js skip
    code.push(0x49); code.push(0x85); code.push(0xED); // test r13, r13
    let js_skip_pos = code.len();
    code.push(0x0F); code.push(0x88);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate values offset: rax = rcx * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap(); // Add weights base
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Load value: movss xmm0, [rax]
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0x00]);
    
    // Calculate output offset: rax = output_index * 4
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R13).unwrap(); // Use r13 for output_index
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
    
    // Skip point (if index out of bounds)
    let skip_store_pos = code.len();
    let jae_skip_offset = (skip_store_pos as i32 - (jae_skip_pos + 6) as i32) as i32;
    code[jae_skip_pos + 2..jae_skip_pos + 6].copy_from_slice(&jae_skip_offset.to_le_bytes());
    let js_skip_offset = (skip_store_pos as i32 - (js_skip_pos + 6) as i32) as i32;
    code[js_skip_pos + 2..js_skip_pos + 6].copy_from_slice(&js_skip_offset.to_le_bytes());
    
    // Increment counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    // Jump back: jmp scatter_loop_start
    let scatter_jmp_pos = code.len();
    code.push(0xE9);
    let scatter_jmp_offset_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Patch scatter loop jumps
    let scatter_loop_end = code.len();
    let scatter_jge_offset = (scatter_loop_end as i32 - (scatter_jge_pos + 6) as i32) as i32;
    code[scatter_jge_pos + 2..scatter_jge_pos + 6].copy_from_slice(&scatter_jge_offset.to_le_bytes());
    let scatter_jmp_offset = (scatter_loop_start as i32 - (scatter_jmp_pos + 5) as i32) as i32;
    code[scatter_jmp_offset_placeholder..scatter_jmp_offset_placeholder + 4].copy_from_slice(&scatter_jmp_offset.to_le_bytes());
    
    // No weights case: just copy input to output (already done above, jump to end)
    let no_weights_start = code.len();
    let jb_no_weights_offset = (no_weights_start as i32 - (jb_no_weights_pos + 6) as i32) as i32;
    code[jb_no_weights_pos + 2..jb_no_weights_pos + 6].copy_from_slice(&jb_no_weights_offset.to_le_bytes());
    
    // Copy input to output (same as above)
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let no_weights_loop_start = code.len();
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let no_weights_jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02);
    
    match input_buffer_reg {
        Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
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
    
    let no_weights_jmp_pos = code.len();
    code.push(0xE9);
    let no_weights_jmp_placeholder = code.len();
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    let no_weights_loop_end = code.len();
    let no_weights_jge_offset = (no_weights_loop_end as i32 - (no_weights_jge_pos + 6) as i32) as i32;
    code[no_weights_jge_pos + 2..no_weights_jge_pos + 6].copy_from_slice(&no_weights_jge_offset.to_le_bytes());
    let no_weights_jmp_offset = (no_weights_loop_start as i32 - (no_weights_jmp_pos + 5) as i32) as i32;
    code[no_weights_jmp_placeholder..no_weights_jmp_placeholder + 4].copy_from_slice(&no_weights_jmp_offset.to_le_bytes());
    
    Ok(code)
}
