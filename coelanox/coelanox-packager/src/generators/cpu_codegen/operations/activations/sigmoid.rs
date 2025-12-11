//! Sigmoid operation
//! Element-wise sigmoid: sigmoid(x) ≈ 0.5 * (1 + x / (1 + |x|))

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_sigmoid(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Sigmoid: fast approximation sigmoid(x) ≈ 0.5 * (1 + x / (1 + |x|))
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: use custom helper with constants setup
        return generate_2d_sigmoid(generator, input_buffer_reg, output_buffer_reg, rows, cols);
    }
        
        // 1D case: simple linear loop
        let mut code = Vec::new();
        let mut encoder = Encoder::new(64);
        
        // Load constants BEFORE loop: 1.0 in xmm2, 0.5 in xmm3, 1/3 in xmm7, 2/15 in xmm4, -1.0 in xmm5
        code.push(0xB8); // mov eax, 1
        code.extend_from_slice(&1u32.to_le_bytes());
        code.extend_from_slice(&[0xF3, 0x0F, 0x2A, 0xD0]); // cvtsi2ss xmm2, eax (1.0)
        
        // For 0.5, load as float constant: mov eax, 0x3F000000 (0.5f); movd xmm3, eax
        code.push(0xB8); // mov eax, 0x3F000000
        code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax (0.5)
        
        // Load 1/3 for tanh polynomial
        code.push(0xB8);
        code.extend_from_slice(&0x3EAAAAABu32.to_le_bytes()); // 1/3 ≈ 0.333333f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        
        // Load 2/15 for tanh polynomial
        code.push(0xB8);
        code.extend_from_slice(&0x3E088889u32.to_le_bytes()); // 2/15 ≈ 0.133333f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
        
        // Load -1.0 for tanh clamping (use xmm5 instead of xmm6 to avoid conflict)
        code.push(0xB8);
        code.extend_from_slice(&0xBF800000u32.to_le_bytes()); // -1.0f
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE8]); // movd xmm5, eax
        
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
        
        match input_buffer_reg {
            Register::R10 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]),
            Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]),
            Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]),
            Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]),
            _ => return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string())),
        }
        
        // Better sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        // Save original x in xmm1
        code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
        
        // Compute -x: xmm0 = 0 - x
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (zero, reuse xmm1)
        code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (restore x)
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xED]); // pxor xmm5, xmm5 (zero)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xE8]); // subss xmm5, xmm0 (xmm5 = -x)
        code.extend_from_slice(&[0x0F, 0x28, 0xC5]); // movaps xmm0, xmm5 (xmm0 = -x)
        
        // Compute exp(-x) using better approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        // Save -x in xmm4 for later use
        code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0 (xmm4 = -x)
        
        // Compute x²: xmm5 = xmm4 * xmm4
        code.extend_from_slice(&[0x0F, 0x28, 0xEC]); // movaps xmm5, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEC]); // mulss xmm5, xmm4 (xmm5 = x²)
        
        // Compute x³: xmm6 = xmm5 * xmm4
        code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = x³)
        
        // Compute x⁴: save in xmm1 temporarily
        code.extend_from_slice(&[0x0F, 0x28, 0xCD]); // movaps xmm1, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCD]); // mulss xmm1, xmm5 (xmm1 = x⁴)
        
        // Load 1/6 and 1/24 constants (xmm7 already has 1/3, we'll overwrite it)
        code.push(0xB8);
        code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax
        
        // Load 1/24 into a free register - use xmm4's value temporarily, then reload
        // Actually, save x⁴ calculation, load 1/24, then use it
        code.push(0xB8);
        code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
        // We need to save x⁴ first - use xmm4 temporarily (it has -x, we'll recompute if needed)
        // Actually better: keep x⁴ in xmm1, load 1/24 into xmm4 (overwrite -x, we don't need it anymore)
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax (xmm4 = 1/24, overwrites -x)
        
        // Compute exp(-x) = 1 + (-x) + (-x)²/2 + (-x)³/6 + (-x)⁴/24
        // xmm0 = -x (will accumulate result)
        // xmm5 = (-x)², xmm6 = (-x)³, xmm1 = (-x)⁴
        // xmm3 = 0.5, xmm7 = 1/6, xmm4 = 1/24
        
        // Start with 1 + (-x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (xmm0 = -x + 1)
        
        // Add x²/2
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEB]); // mulss xmm5, xmm3 (xmm5 = x²/2)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
        
        // Add x³/6
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF7]); // mulss xmm6, xmm7 (xmm6 = x³/6)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
        
        // Add x⁴/24
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCC]); // mulss xmm1, xmm4 (xmm1 = x⁴/24)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1 (xmm0 = exp(-x))
        
        // Clamp exp to positive
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
        
        // Compute 1 + exp(-x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (xmm0 = 1 + exp(-x))
        
        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        code.extend_from_slice(&[0x0F, 0x28, 0xCA]); // movaps xmm1, xmm2 (xmm1 = 1.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC8]); // divss xmm1, xmm0 (xmm1 = 1 / (1 + exp(-x)))
        code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1 (xmm0 = sigmoid(x))
        
        // Clamp to [0, 1]: max with 0, min with 1
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (0.0, reuse xmm1)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1 (max(xmm0, 0))
        code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC2]); // minss xmm0, xmm2 (min(xmm0, 1.0))
        
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

fn generate_2d_sigmoid(
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
        
        // Load constants: 1.0 in xmm2, 0.5 in xmm3
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
        
        // Better sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        // Save original x in xmm1
        code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
        
        // Compute -x: negate xmm0
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1 (zero, reuse)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC1]); // subss xmm1, xmm0 (xmm1 = -x)
        code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1 (xmm0 = -x)
        
        // Compute exp(-x) using better approximation
        // Save -x in xmm4
        code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0
        
        // Compute x², x³, x⁴
        code.extend_from_slice(&[0x0F, 0x28, 0xEC]); // movaps xmm5, xmm4
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEC]); // mulss xmm5, xmm4 (xmm5 = x²)
        code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = x³)
        code.extend_from_slice(&[0x0F, 0x28, 0xFE]); // movaps xmm7, xmm6
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xFC]); // mulss xmm7, xmm4 (xmm7 = x⁴)
        
        // Load 1/6 and 1/24
        code.push(0xB8);
        code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax (overwrite x⁴)
        code.push(0xB8);
        code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/24
        code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
        
        // Recompute x⁴ in xmm1
        code.extend_from_slice(&[0x0F, 0x28, 0xCD]); // movaps xmm1, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCD]); // mulss xmm1, xmm5 (xmm1 = x⁴)
        
        // exp(-x) = 1 + (-x) + (-x)²/2 + (-x)³/6 + (-x)⁴/24
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (1 + (-x))
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEB]); // mulss xmm5, xmm3 (x²/2)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF7]); // mulss xmm6, xmm7 (x³/6)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6
        code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCC]); // mulss xmm1, xmm4 (x⁴/24)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1 (exp(-x))
        
        // Clamp exp
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xC9]); // pxor xmm1, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC1]); // maxss xmm0, xmm1
        
        // 1 + exp(-x)
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2
        
        // sigmoid = 1 / (1 + exp(-x))
        code.extend_from_slice(&[0x0F, 0x28, 0xCA]); // movaps xmm1, xmm2
        code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xC8]); // divss xmm1, xmm0
        code.extend_from_slice(&[0x0F, 0x28, 0xC1]); // movaps xmm0, xmm1
        
        // Clamp to [0, 1]
        code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xED]); // pxor xmm5, xmm5 (0.0)
        code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC5]); // maxss xmm0, xmm5 (max(xmm0, 0))
        code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xC2]); // minss xmm0, xmm2 (min(xmm0, 1.0))
        
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

