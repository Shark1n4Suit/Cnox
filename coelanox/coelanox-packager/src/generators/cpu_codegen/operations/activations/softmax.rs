//! Softmax operation
//! Element-wise softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_softmax(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    // Algorithm:
    // 1. Find max (for numerical stability)
    // 2. Compute exp(x - max) for each element
    // 3. Sum all exp values
    // 4. Normalize: output[i] = exp(x[i] - max) / sum
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // For 2D: applies softmax row-wise (across columns)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
            // 2D case: proper row-wise softmax
            // For each row i: softmax(row_i) = exp(row_i - max(row_i)) / sum(exp(row_i - max(row_i)))
            
            let mut code = Vec::new();
            let mut encoder = Encoder::new(64);
            
            // Store cols in r9 (for offset calculation)
            if cols <= 127 {
                code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
            } else {
                code.push(0x49); code.push(0xC7); code.push(0xC1);
                code.extend_from_slice(&(cols as u32).to_le_bytes());
            }
            
            // Load constants: 0.5 in xmm2, 1.0 in xmm3 (for exp approximation)
            code.push(0xB8);
            code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax (0.5)
            
            code.push(0xB8);
            code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax (1.0)
            
            // Outer loop: rows (rbx)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RBX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let outer_loop_start = code.len();
            // Compare rbx with rows: cmp rbx, rows
            if rows <= 127 {
                code.push(0x48); code.push(0x83); code.push(0xFB); code.push(rows as u8);
            } else {
                code.push(0x48); code.push(0x81); code.push(0xFB);
                code.extend_from_slice(&(rows as u32).to_le_bytes());
            }
            
            // Jump if rbx >= rows (exit loop): jge outer_loop_end
            let outer_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // ============================================================================
            // Calculate row base offset once per row: r12 = (rbx * cols) * 4
            // ============================================================================
            // rax = rbx (row index)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RBX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // rax = rax * cols (row * cols) - mul instruction uses rax implicitly
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Mul_rm64, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // rax = rax * 4 (byte offset for row base)
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
            
            // Save row base offset to r12 for use in all three passes
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R12, Register::RAX).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // ============================================================================
            // PASS 1: Find max in current row
            // ============================================================================
            // Calculate offset for first element: rax = r12 + 0
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R12).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // Load first element of row as initial max: movss xmm0, [input_buffer_reg + rax], then movss xmm4, xmm0
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]), // movss xmm0, [r10 + rax]
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]), // movss xmm0, [r11 + rax]
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]), // movss xmm0, [r13 + rax]
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]), // movss xmm0, [r14 + rax]
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xE0]); // movss xmm4, xmm0
            
            // Inner loop for max: cols (r8), start at 1 (skip first element, already loaded)
            code.push(0x49); code.push(0xC7); code.push(0xC0); code.push(0x01); code.push(0x00); code.push(0x00); code.push(0x00); // mov r8, 1
            
            let pass1_loop_start = code.len();
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass1_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Calculate offset for element in row: rax = r12 + r8 * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2 (rax = r8 * 4)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // Load element: movss xmm0, [input_buffer_reg + rax]
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Update max: maxss xmm4, xmm0
            code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xE0]);
            
            // Increment: inc r8
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass1_jmp_pos = code.len();
            code.push(0xE9);
            let pass1_jmp_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            let pass1_loop_end = code.len();
            let pass1_jge_offset = (pass1_loop_end as i32 - (pass1_jge_pos + 6) as i32) as i32;
            code[pass1_jge_pos + 2..pass1_jge_pos + 6].copy_from_slice(&pass1_jge_offset.to_le_bytes());
            let pass1_jmp_offset = (pass1_loop_start as i32 - (pass1_jmp_pos + 5) as i32) as i32;
            code[pass1_jmp_placeholder..pass1_jmp_placeholder + 4].copy_from_slice(&pass1_jmp_offset.to_le_bytes());
            
            // xmm4 now contains max of current row
            
            // ============================================================================
            // PASS 2: Compute exp(x - max) and sum for current row
            // ============================================================================
            // Zero accumulator for sum: pxor xmm5, xmm5
            code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xED]);
            
            // Inner loop for sum: cols (r8), start at 0
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass2_loop_start = code.len();
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass2_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Calculate 2D offset: rax = (rbx * cols + r8) * 4
            code.extend_from_slice(&elementwise::generate_2d_offset_calculation(generator, Register::R9, 4));
            
            // Load element: movss xmm0, [input_buffer_reg + rax]
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Compute (x - max): subss xmm0, xmm4
            code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
            
            // Compute exp(x - max) using better approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
            // Save x in xmm1: movss xmm1, xmm0
            code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]);
            
            // Compute x²: mulss xmm1, xmm1
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]);
            
            // Compute x³: save x², then x³ = x² * x
            code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
            
            // Compute x⁴: xmm6 = x² * x²
            code.extend_from_slice(&[0x0F, 0x28, 0xF1]); // movaps xmm6, xmm1 (xmm6 = x²)
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (xmm6 = x⁴)
            
            // Load 1/6 and 1/24 constants
            code.push(0xB8);
            code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax (xmm7 = 1/6)
            
            code.push(0xB8);
            code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax (xmm4 = 1/24, overwrite 0.5)
            
            // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24
            // Start with 1 + x: addss xmm0, xmm3 (xmm3 has 1.0)
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]);
            
            // Add x²/2: mulss xmm1, xmm2 (xmm2 has 0.5)
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]);
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
            
            // Add x³/6: mulss xmm5, xmm7; addss xmm0, xmm5
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
            
            // Add x⁴/24: mulss xmm6, xmm4; addss xmm0, xmm6
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
            
            // Clamp to [0, large_value]: maxss xmm0, [0.0]
            code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
            code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
            
            // Accumulate sum: addss xmm5, xmm0
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xE8]);
            
            // Increment: inc r8
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass2_jmp_pos = code.len();
            code.push(0xE9);
            let pass2_jmp_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            let pass2_loop_end = code.len();
            let pass2_jge_offset = (pass2_loop_end as i32 - (pass2_jge_pos + 6) as i32) as i32;
            code[pass2_jge_pos + 2..pass2_jge_pos + 6].copy_from_slice(&pass2_jge_offset.to_le_bytes());
            let pass2_jmp_offset = (pass2_loop_start as i32 - (pass2_jmp_pos + 5) as i32) as i32;
            code[pass2_jmp_placeholder..pass2_jmp_placeholder + 4].copy_from_slice(&pass2_jmp_offset.to_le_bytes());
            
            // xmm5 now contains sum of exp(x - max) for current row - save to xmm6
            code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xF5]); // movss xmm6, xmm5
            
            // ============================================================================
            // PASS 3: Normalize current row (divide each exp(x - max) by sum)
            // ============================================================================
            // Inner loop for normalize: cols (r8), start at 0
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Xor_rm64_r64, Register::R8, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass3_loop_start = code.len();
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::R8, Register::R9).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass3_jge_pos = code.len();
            code.push(0x0F); code.push(0x8D);
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            // Calculate offset for element in row: rax = r12 + r8 * 4
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2 (rax = r8 * 4)
            encoder = Encoder::new(64);
            let inst = Instruction::with2(Code::Add_rm64_r64, Register::RAX, Register::R12).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            // Load element: movss xmm0, [input_buffer_reg + rax]
            match input_buffer_reg {
                Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
            }
            
            // Compute (x - max): subss xmm0, xmm4
            code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
            
            // Recompute exp(x - max) using same better approximation
            // Save x in xmm1: movss xmm1, xmm0
            code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]);
            
            // Compute x², x³, x⁴ (same as pass 2)
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]); // mulss xmm1, xmm1 (x²)
            code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (x³)
            code.extend_from_slice(&[0x0F, 0x28, 0xF1]); // movaps xmm6, xmm1
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (x⁴)
            
            // Reload 1/6 and 1/24 (xmm7 and xmm4 were overwritten, reload)
            code.push(0xB8);
            code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
            code.push(0xB8);
            code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
            code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
            
            // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]); // addss xmm0, xmm3 (1 + x)
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]); // mulss xmm1, xmm2 (x²/2)
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7 (x³/6)
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
            code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (x⁴/24)
            code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
            
            // Clamp: maxss xmm0, [0.0]
            code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
            code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
            
            // Divide by sum: divss xmm0, xmm6 (xmm6 has sum)
            code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC6]);
            
            // Store: movss [output_buffer_reg + rax], xmm0
            match output_buffer_reg {
                Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
                Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
                Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
                _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
            }
            
            // Increment: inc r8
            encoder = Encoder::new(64);
            let inst = Instruction::with1(Code::Inc_rm64, Register::R8).unwrap();
            encoder.encode(&inst, 0).unwrap();
            code.extend_from_slice(&encoder.take_buffer());
            
            let pass3_jmp_pos = code.len();
            code.push(0xE9);
            let pass3_jmp_placeholder = code.len();
            code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
            
            let pass3_loop_end = code.len();
            let pass3_jge_offset = (pass3_loop_end as i32 - (pass3_jge_pos + 6) as i32) as i32;
            code[pass3_jge_pos + 2..pass3_jge_pos + 6].copy_from_slice(&pass3_jge_offset.to_le_bytes());
            let pass3_jmp_offset = (pass3_loop_start as i32 - (pass3_jmp_pos + 5) as i32) as i32;
            code[pass3_jmp_placeholder..pass3_jmp_placeholder + 4].copy_from_slice(&pass3_jmp_offset.to_le_bytes());
            
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
        
        // For 1D scalar implementation, we'll do:
        // Pass 1: Find max
        // Pass 2: Compute exp(x - max) and sum
        // Pass 3: Normalize (divide by sum)
        
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        let input_buffer_reg = generator.get_current_input_buffer();
        let output_buffer_reg = generator.get_current_output_buffer();
        
        // ============================================================================
        // PASS 1: Find maximum (for numerical stability)
        // ============================================================================
        // Save rsi (input length) to r11 for loop (r11 is safe - typically intermediate buffer, not used as input/output in single-op tests)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::R11, Register::RSI).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Load first element as initial max
        // movss xmm4, [input_buffer_reg] (first element)
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x22]), // movss xmm4, [r10]
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x23]), // movss xmm4, [r11]
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x25]), // movss xmm4, [r13]
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x26]), // movss xmm4, [r14]
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Initialize loop counter: rcx = 1 (skip first element, already loaded)
        code.push(0x48); code.push(0xC7); code.push(0xC1); code.push(0x01); code.push(0x00); code.push(0x00); code.push(0x00); // mov rcx, 1
        
        let pass1_loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass1_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate byte offset: rax = rcx * 4
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
        
        // Update max: maxss xmm4, xmm0
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xE0]);
        
        // Increment counter: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back: jmp pass1_loop_start
        let pass1_jmp_pos = code.len();
        code.push(0xE9);
        let pass1_jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch pass1 jumps
        let pass1_loop_end = code.len();
        let pass1_jge_offset = (pass1_loop_end as i32 - (pass1_jge_pos + 6) as i32) as i32;
        code[pass1_jge_pos + 2..pass1_jge_pos + 6].copy_from_slice(&pass1_jge_offset.to_le_bytes());
        let pass1_jmp_offset = (pass1_loop_start as i32 - (pass1_jmp_pos + 5) as i32) as i32;
        code[pass1_jmp_offset_placeholder..pass1_jmp_offset_placeholder + 4].copy_from_slice(&pass1_jmp_offset.to_le_bytes());
        
        // xmm4 now contains max - we'll keep it there
        
        // ============================================================================
        // Load constants once (used in both pass 2 and pass 3)
        // ============================================================================
        // Load 0.5 into xmm2: mov eax, 0x3F000000; movd xmm2, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
        
        // Load 1.0 into xmm3: mov eax, 0x3F800000; movd xmm3, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
        
        // ============================================================================
        // PASS 2: Compute exp(x - max) and sum
        // ============================================================================
        // Zero accumulator for sum: pxor xmm5, xmm5
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xED]);
        
        // Reset loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass2_loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass2_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate byte offset: rax = rcx * 4
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
        
        // Compute (x - max): subss xmm0, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
        
        // Compute exp(x - max) using better approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        // Save x in xmm1: movss xmm1, xmm0
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]);
        
        // Compute x², x³, x⁴
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]); // mulss xmm1, xmm1 (x²)
        code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (x³)
        code.extend_from_slice(&[0x0F, 0x28, 0xF1]); // movaps xmm6, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (x⁴)
        
        // Load 1/6 and 1/24
        code.push(0xB8);
        code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
        
        // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]); // addss xmm0, xmm3 (1 + x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]); // mulss xmm1, xmm2 (x²/2)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7 (x³/6)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (x⁴/24)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
        
        // Clamp to [0, large_value] to avoid overflow
        // Zero xmm1 for clamp: pxor xmm1, xmm1 (0.0)
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
        
        // Store exp(x - max) temporarily (we'll use it in pass 3)
        // For now, accumulate sum: addss xmm5, xmm0
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xE8]);
        
        // Store exp value to intermediate location (we need to store it for pass 3)
        // Since we don't have intermediate storage, we'll recompute in pass 3
        // For now, just accumulate sum
        
        // Increment counter: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back: jmp pass2_loop_start
        let pass2_jmp_pos = code.len();
        code.push(0xE9);
        let pass2_jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch pass2 jumps
        let pass2_loop_end = code.len();
        let pass2_jge_offset = (pass2_loop_end as i32 - (pass2_jge_pos + 6) as i32) as i32;
        code[pass2_jge_pos + 2..pass2_jge_pos + 6].copy_from_slice(&pass2_jge_offset.to_le_bytes());
        let pass2_jmp_offset = (pass2_loop_start as i32 - (pass2_jmp_pos + 5) as i32) as i32;
        code[pass2_jmp_offset_placeholder..pass2_jmp_offset_placeholder + 4].copy_from_slice(&pass2_jmp_offset.to_le_bytes());
        
        // xmm5 now contains sum of exp(x - max) - save to xmm6
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xF5]); // movss xmm6, xmm5
        
        // ============================================================================
        // PASS 3: Normalize (divide each exp(x - max) by sum)
        // ============================================================================
        // Reset loop counter: rcx = 0
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass3_loop_start = code.len();
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R11).unwrap(); // Compare with input length (same as output length for softmax)
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        let pass3_jge_pos = code.len();
        code.push(0x0F); code.push(0x8D);
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Calculate byte offset: rax = rcx * 4
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
        
        // Compute (x - max): subss xmm0, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC4]);
        
        // Recompute exp(x - max) using same better approximation
        // Save x in xmm1: movss xmm1, xmm0
        code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]);
        
        // Compute x², x³, x⁴
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC9]); // mulss xmm1, xmm1 (x²)
        code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (x³)
        code.extend_from_slice(&[0x0F, 0x28, 0xF1]); // movaps xmm6, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (x⁴)
        
        // Reload 1/6 and 1/24
        code.push(0xB8);
        code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
        
        // exp(x) = 1 + x + x²/2 + x³/6 + x⁴/24
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC3]); // addss xmm0, xmm3 (1 + x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCA]); // mulss xmm1, xmm2 (x²/2)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7 (x³/6)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (x⁴/24)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
        
        // Clamp to [0, large_value]: zero xmm1 and maxss
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
        
        // Divide by sum: divss xmm0, xmm6 (xmm6 has sum)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC6]);
        
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
        
        // Jump back: jmp pass3_loop_start
        let pass3_jmp_pos = code.len();
        code.push(0xE9);
        let pass3_jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
        
        // Patch pass3 jumps
        let pass3_loop_end = code.len();
        let pass3_jge_offset = (pass3_loop_end as i32 - (pass3_jge_pos + 6) as i32) as i32;
        code[pass3_jge_pos + 2..pass3_jge_pos + 6].copy_from_slice(&pass3_jge_offset.to_le_bytes());
        let pass3_jmp_offset = (pass3_loop_start as i32 - (pass3_jmp_pos + 5) as i32) as i32;
        code[pass3_jmp_offset_placeholder..pass3_jmp_offset_placeholder + 4].copy_from_slice(&pass3_jmp_offset.to_le_bytes());
        
        Ok(code)
}
