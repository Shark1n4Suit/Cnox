//! Tanh operation
//! Element-wise tanh using polynomial approximation: tanh(x) ≈ x - x^3/3 + 2*x^5/15

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_tanh(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Tanh: polynomial approximation tanh(x) ≈ x - x^3/3 + 2*x^5/15
    // For |x| > 2.5, clamp to ±1.0
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: use custom function with constants loaded before loops
        return generate_2d_tanh(generator, input_buffer_reg, output_buffer_reg, rows, cols);
    }
    
    // 1D case: use simple loop with polynomial
    let mut code = Vec::new();
    let mut encoder = iced_x86::Encoder::new(64);
    
    // Load constants: 1.0 in xmm2, -1.0 in xmm3, 1/3 in xmm7, 2/15 in xmm4
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
    
    code.push(0xB8);
    code.extend_from_slice(&0xBF800000u32.to_le_bytes()); // -1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3EAAAAABu32.to_le_bytes()); // 1/3 ≈ 0.333333f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3E088889u32.to_le_bytes()); // 2/15 ≈ 0.133333f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
    
    // Initialize loop counter: rcx = 0
    encoder = iced_x86::Encoder::new(64);
    let inst = iced_x86::Instruction::with2(iced_x86::Code::Xor_rm64_r64, iced_x86::Register::RCX, iced_x86::Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    
    // Check if done: cmp rcx, r15 (output length)
    encoder = iced_x86::Encoder::new(64);
    let inst = iced_x86::Instruction::with2(iced_x86::Code::Cmp_rm64_r64, iced_x86::Register::RCX, iced_x86::Register::R15).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let jge_pos = code.len();
    code.push(0x0F); code.push(0x8D);
    code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00);
    
    // Calculate byte offset: rax = rcx * 4
    encoder = iced_x86::Encoder::new(64);
    let inst = iced_x86::Instruction::with2(iced_x86::Code::Mov_r64_rm64, iced_x86::Register::RAX, iced_x86::Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
    
    // Load input: movss xmm0, [input_buffer_reg + rax]
    match input_buffer_reg {
        iced_x86::Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        iced_x86::Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        iced_x86::Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        iced_x86::Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Better tanh polynomial: tanh(x) ≈ x - x³/3 + 2*x⁵/15 - 17*x⁷/315
    // Save original x in xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]); // movss xmm1, xmm0 (save x)
    
    // Compute x²: xmm1 = x * x
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = x²)
    
    // Compute x³: xmm5 = x² * x
    code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
    
    // Compute x⁵: xmm6 = x³ * x²
    code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5 (xmm6 = x³)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (xmm6 = x⁵)
    
    // Compute tanh(x) = x - x³/3 + 2*x⁵/15
    // xmm0 = x, xmm5 = x³, xmm6 = x⁵, xmm7 = 1/3, xmm4 = 2/15
    // Compute -x³/3: mulss xmm5, xmm7; subss xmm0, xmm5
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7 (xmm5 = x³/3)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC5]); // subss xmm0, xmm5 (xmm0 = x - x³/3)
    
    // Compute 2*x⁵/15: mulss xmm6, xmm4; addss xmm0, xmm6
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = 2*x⁵/15)
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6 (xmm0 = tanh(x))
    
    // Clamp to [-1, 1]: minss with 1.0, maxss with -1.0
    code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC2]); // minss xmm0, xmm2 (min(xmm0, 1.0))
    code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC3]); // maxss xmm0, xmm3 (max(xmm0, -1.0))
    
    // Store result: movss [output_buffer_reg + rax], xmm0
    match output_buffer_reg {
        iced_x86::Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
        iced_x86::Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
        iced_x86::Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    // Increment loop counter: inc rcx
    encoder = iced_x86::Encoder::new(64);
    let inst = iced_x86::Instruction::with1(iced_x86::Code::Inc_rm64, iced_x86::Register::RCX).unwrap();
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

fn generate_2d_tanh(
    generator: &CpuCodeGenerator,
    input_buffer_reg: iced_x86::Register,
    output_buffer_reg: iced_x86::Register,
    rows: usize,
    cols: usize
) -> CoelanoxResult<Vec<u8>> {
    use iced_x86::{Code, Instruction, Register, Encoder};
    
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Store cols in r9
    if cols <= 127 {
        code.push(0x49); code.push(0xC7); code.push(0xC1); code.push(cols as u8); code.push(0x00); code.push(0x00); code.push(0x00);
    } else {
        code.push(0x49); code.push(0xC7); code.push(0xC1);
        code.extend_from_slice(&(cols as u32).to_le_bytes());
    }
    
    // Load constants BEFORE loops: 1.0 in xmm2, -1.0 in xmm3, 1/3 in xmm7, 2/15 in xmm4
    // Use same approach as Sigmoid for 1.0
    code.push(0xB8);
    code.extend_from_slice(&1u32.to_le_bytes());
    code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
    
    code.push(0xB8);
    code.extend_from_slice(&0xBF800000u32.to_le_bytes()); // -1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3EAAAAABu32.to_le_bytes()); // 1/3 ≈ 0.333333f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3E088889u32.to_le_bytes()); // 2/15 ≈ 0.133333f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
    
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
        iced_x86::Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
        iced_x86::Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
        iced_x86::Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
        iced_x86::Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
    }
    
    // Better tanh polynomial: tanh(x) ≈ x - x³/3 + 2*x⁵/15
    // xmm0 contains input x, constants are already loaded in xmm2, xmm3, xmm4, xmm7
    // Save original x in xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x10, 0xC8]); // movss xmm1, xmm0 (save x)
    
    // Compute x²: xmm1 = x * x
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = x²)
    
    // Compute x³: xmm5 = x² * x
    code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
    
    // Compute x⁵: xmm6 = x³ * x²
    code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5 (xmm6 = x³)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (xmm6 = x⁵)
    
    // Compute tanh(x) = x - x³/3 + 2*x⁵/15
    // xmm0 = x, xmm5 = x³, xmm6 = x⁵, xmm7 = 1/3, xmm4 = 2/15
    // Compute -x³/3: mulss xmm5, xmm7; subss xmm0, xmm5
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEF]); // mulss xmm5, xmm7 (xmm5 = x³/3)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC5]); // subss xmm0, xmm5 (xmm0 = x - x³/3)
    
    // Compute 2*x⁵/15: mulss xmm6, xmm4; addss xmm0, xmm6
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = 2*x⁵/15)
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6 (xmm0 = tanh(x))
    
    // Clamp to [-1, 1]: minss with 1.0, maxss with -1.0
    code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC2]); // minss xmm0, xmm2 (min(xmm0, 1.0))
    code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC3]); // maxss xmm0, xmm3 (max(xmm0, -1.0))
    
    // Store: movss [output_buffer_reg + rax], xmm0
    match output_buffer_reg {
        iced_x86::Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
        iced_x86::Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
        iced_x86::Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
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
