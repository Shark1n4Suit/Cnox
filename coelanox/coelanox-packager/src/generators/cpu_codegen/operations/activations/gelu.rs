//! GELU operation
//! Element-wise GELU: GELU(x) ≈ x * sigmoid(1.702 * x)

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_gelu(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // GELU: GELU(x) ≈ x * sigmoid(1.702 * x)
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        return generate_2d_gelu(generator, input_buffer_reg, output_buffer_reg, rows, cols);
    }
        
        // 1D case: simple linear loop
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Load constants BEFORE the loop (like sigmoid does)
        // Load constant 1.702 in xmm7: mov eax, 0x3FD9DB23; movd xmm7, eax
        code.push(0xB8);
        code.extend_from_slice(&0x3FD9DB23u32.to_le_bytes()); // 1.702f (correct hex value)
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        
        // Load constants: 1.0 in xmm2, 0.5 in xmm3 (same as sigmoid)
        code.push(0xB8); // mov eax, 1
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
        
        // For 0.5, load as float constant: mov eax, 0x3F000000 (0.5f); movd xmm3, eax
        code.push(0xB8); // mov eax, 0x3F000000
        code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax (0.5)
        
        // Initialize loop counter: rcx = 0
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
        
        // Calculate byte offset: rax = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load input: movss xmm0, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // GELU: x * sigmoid(1.702 * x)
        // Use exact same pattern as sigmoid but multiply by 1.702 first, then by original x
        // Save original x in xmm1
        code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0
        
        // Multiply by 1.702: mulss xmm0, xmm7
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC7]); // mulss xmm0, xmm7
        // Now xmm0 = 1.702*x, xmm1 = original x
        
        // Now compute sigmoid(1.702 * x) using exact same pattern as generate_sigmoid
        // Compute |1.702*x|: copy to xmm4, negate, then max (same as sigmoid)
        code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0
        code.extend_from_slice(&[0x0F, 0x57, 0xE4]); // xorps xmm4, xmm4 (zero)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xE4]); // subss xmm4, xmm0 (0 - x = -x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC4]); // maxss xmm0, xmm4 (max(x, -x) = |x|)
        // Now xmm0 = |1.702*x|, xmm1 = original x
        
        // Compute 1 + |1.702*x|
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (xmm2 has 1.0)
        // Now xmm0 = 1 + |1.702*x|
        
        // Restore 1.702*x to xmm4 for division
        code.extend_from_slice(&[0x0F, 0x28, 0xE1]); // movaps xmm4, xmm1 (copy original x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE7]); // mulss xmm4, xmm7 (1.702*x)
        
        // Compute (1.702*x) / (1 + |1.702*x|): divss xmm4, xmm0
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xE0]); // divss xmm4, xmm0
        // Now xmm4 = (1.702*x) / (1 + |1.702*x|)
        
        // Compute 1 + (1.702*x)/(1+|1.702*x|)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xE2]); // addss xmm4, xmm2 (xmm2 has 1.0)
        // Now xmm4 = 1 + (1.702*x)/(1+|1.702*x|)
        
        // Multiply by 0.5: mulss xmm4, xmm3 (xmm3 has 0.5)
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE3]); // mulss xmm4, xmm3
        // Now xmm4 = sigmoid(1.702 * x)
        
        // Multiply by original input: mulss xmm1, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCC]); // mulss xmm1, xmm4
        code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
        
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

fn generate_2d_gelu(
    generator: &CpuCodeGenerator,
    input_buffer_reg: Register,
    output_buffer_reg: Register,
    rows: usize,
    cols: usize
) -> CoelanoxResult<Vec<u8>> {
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Store cols in r9
        if cols <= 127 {
            code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
        } else {
            code.push(0x49); code.push(0xC7); code.push(0xC1);
            code.extend_from_slice(&(cols as u32).to_le_bytes());
        }
        
        // Load constant 1.702 in xmm4 (for GELU: x * sigmoid(1.702 * x))
        code.push(0xB8);
        code.extend_from_slice(&0x3FD9B6DBu32.to_le_bytes()); // 1.702f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax (1.702)
        
        // Load constants: 1.0 in xmm2, 0.5 in xmm3 (for sigmoid)
        code.push(0xB8);
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
        code.push(0xB8);
        code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax (0.5)
        
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
        
        // Load input: movss xmm0, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // GELU: x * sigmoid(1.702 * x)
        // Save original x in xmm1
        code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
        // Compute 1.702 * x: mulss xmm0, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC4]); // mulss xmm0, xmm4
        // Compute sigmoid(1.702 * x) - reuse sigmoid approximation
        // Save 1.702*x in xmm5
        code.extend_from_slice(&[0x0F, 0x28, 0xE8]); // movaps xmm5, xmm0
        // Compute |1.702*x|: copy to xmm6, negate, then max
        code.extend_from_slice(&[0x0F, 0x28, 0xF0]); // movaps xmm6, xmm0 (copy xmm0 to xmm6)
        // Negate xmm6: pxor xmm7, xmm7; subss xmm7, xmm6
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xFF]); // pxor xmm7, xmm7 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xFE]); // subss xmm7, xmm6 (0 - xmm6 = -xmm6)
        // Max: maxss xmm6, xmm7 (max(x, -x) = |x|)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xF7]); // maxss xmm6, xmm7
        // Compute 1 + |1.702*x|: addss xmm6, xmm2 (xmm2 is 1.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xF2]); // addss xmm6, xmm2
        // Compute (1.702*x) / (1 + |1.702*x|): divss xmm5, xmm6
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xEE]); // divss xmm5, xmm6
        // Compute 1 + (1.702*x)/(1+|1.702*x|): addss xmm5, xmm2
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xEA]); // addss xmm5, xmm2
        // Multiply by 0.5: mulss xmm5, xmm3
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEB]); // mulss xmm5, xmm3
        // Clamp sigmoid to [0, 1]
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC0]); // pxor xmm0, xmm0 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xE8]); // maxss xmm5, xmm0 (clamp to 0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xEA]); // minss xmm5, xmm2 (clamp to 1)
        // Multiply by original x: mulss xmm1, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCD]); // mulss xmm1, xmm5
        code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
        
        // Store: movss [output_buffer_reg + rax], xmm0
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
        
        Ok(code)
}
