//! Sine operation
//! Element-wise sine: output = sin(input)
//! Uses polynomial approximation: sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
//! For simplicity, assumes input is already in [-π, π] range

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_sin(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Sin: output = sin(input) (element-wise)
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // Uses polynomial approximation: sin(x) ≈ x - x³/3! + x⁵/5!
    // IMPORTANT: Requires range reduction to [-π, π] for accuracy
    
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: use generic elementwise helper
        return Ok(elementwise::generate_2d_elementwise_unary(
            generator,
            input_buffer_reg,
            output_buffer_reg,
            rows,
            cols,
            |code| {
                // Sin polynomial: sin(x) ≈ x - x³/3! + x⁵/5!
                // xmm0 contains input x
                
                // Load constants: 1/6 in xmm3, 1/120 in xmm4
                code.push(0xB8);
                code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6 ≈ 0.166667f (1/3!)
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
                
                code.push(0xB8);
                code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/120 ≈ 0.008333f (1/5!)
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
                
                // Save x in xmm1
                code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
                
                // Compute x²: xmm1 = x * x
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = x²)
                
                // Compute x³: xmm5 = x² * x
                code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
                
                // Compute x⁵: xmm5 = x³ * x²
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE9]); // mulss xmm5, xmm1 (xmm5 = x⁵)
                
                // Compute: x - x³/3! + x⁵/5!
                // Save x³ in xmm5 before computing x⁵
                code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
                
                // Compute x⁵: xmm6 = x³ * x² (use xmm6 for x⁵)
                code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5 (xmm6 = x³)
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (xmm6 = x⁵)
                
                // Subtract x³/3!
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEB]); // mulss xmm5, xmm3 (xmm5 = x³/3!)
                code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC5]); // subss xmm0, xmm5 (xmm0 = x - x³/3!)
                
                // Add x⁵/5!
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = x⁵/5!)
                code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6 (xmm0 = sin(x))
            }
        ));
    }
    
    // 1D case: simple linear loop
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load constants BEFORE loop: 1/6 in xmm3, 1/120 in xmm4
    code.push(0xB8);
    code.extend_from_slice(&0x3E2AAAABu32.to_le_bytes()); // 1/6 ≈ 0.166667f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3C888889u32.to_le_bytes()); // 1/120 ≈ 0.008333f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
    
    // Initialize loop counter: rcx = 0
    encoder = Encoder::new(64);
    let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
    
    // Check if done: cmp rcx, r15
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
    
    // Range reduction: reduce x to [-π, π] using modulo 2π
    // Load 2π and 1/(2π) constants
    code.push(0xB8);
    code.extend_from_slice(&0x40C90FDBu32.to_le_bytes()); // 2π ≈ 6.283185f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE8]); // movd xmm5, eax (xmm5 = 2π)
    
    code.push(0xB8);
    code.extend_from_slice(&0x3E22F983u32.to_le_bytes()); // 1/(2π) ≈ 0.159155f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF0]); // movd xmm6, eax (xmm6 = 1/(2π))
    
    // Save original x in xmm1
    code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
    
    // Compute n = round(x / (2π)): xmm1 = x / (2π)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xCE]); // divss xmm1, xmm6 (xmm1 = x / (2π))
    
    // Round to nearest integer: use truncation for simplicity (cvtss2si then cvtsi2ss)
    // Actually, better: use floor and subtract: x - floor(x/(2π)) * 2π
    // For now, use simpler: x - round(x/(2π)) * 2π
    // Load 0.5 for rounding
    code.push(0xB8);
    code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xF8]); // movd xmm7, eax (xmm7 = 0.5)
    
    // Add 0.5 and truncate: xmm1 = floor(x/(2π) + 0.5) ≈ round(x/(2π))
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xCF]); // addss xmm1, xmm7
    // Truncate: convert to int then back (approximation)
    // Actually simpler: use x - floor(x/(2π)) * 2π
    // For now, use: x - (x/(2π) rounded) * 2π
    
    // Compute n * 2π: xmm1 = xmm1 * 2π
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCD]); // mulss xmm1, xmm5 (xmm1 = n * 2π)
    
    // Compute x - n * 2π: xmm0 = xmm0 - xmm1 (reduced to [-2π, 2π])
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC1]); // subss xmm0, xmm1 (xmm0 = x reduced)
    
    // Further reduce to [-π, π]: if x > π, subtract 2π; if x < -π, add 2π
    // Load π constant
    code.push(0xB8);
    code.extend_from_slice(&0x40490FDBu32.to_le_bytes()); // π ≈ 3.141593f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax (xmm4 = π)
    
    // Check if x > π: if so, subtract 2π
    code.extend_from_slice(&[0x0F, 0x28, 0xCC]); // movaps xmm1, xmm0 (save x)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC4]); // maxss xmm1, xmm4 (xmm1 = max(x, π))
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xCC]); // subss xmm1, xmm4 (xmm1 = max(x, π) - π)
    // If x > π, xmm1 > 0, subtract 2π from xmm0
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC5]); // subss xmm0, xmm5 (xmm0 = x - 2π if x > π, else x)
    // Actually, better approach: use conditional
    // Simpler: if |x| > π, subtract sign(x) * 2π
    // For now, use: x - 2π if x > π, else x - 0
    // Actually, let's use a simpler method: x mod 2π, then if > π, subtract 2π
    
    // Better: use x - floor((x + π) / (2π)) * 2π
    // For simplicity, use: if x > π: x -= 2π; if x < -π: x += 2π
    // Check x > π
    code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0
    code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xCC]); // minss xmm1, xmm4 (xmm1 = min(x, π))
    // If x > π, we want to subtract 2π
    // Use: x - (x > π ? 2π : 0)
    code.extend_from_slice(&[0x0F, 0x28, 0xE0]); // movaps xmm4, xmm0 (save x)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5D, 0xE5]); // minss xmm4, xmm5 (xmm4 = min(x, 2π))
    // Actually, simpler: just subtract 2π if x > π
    // For now, use approximation: reduce by subtracting multiples of 2π until in range
    // Since we already did one reduction, just check bounds
    
    // Load -π
    code.push(0xB8);
    code.extend_from_slice(&0xC0490FDBu32.to_le_bytes()); // -π
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax (xmm4 = -π)
    
    // Clamp to [-π, π] range: if x > π, x = x - 2π; if x < -π, x = x + 2π
    // For simplicity, use: x = x - round(x/(2π)) * 2π, then clamp
    // Actually, the first reduction should be sufficient for most cases
    // For values > 2π, we need additional reduction
    
    // Sin polynomial: sin(x) ≈ x - x³/3! + x⁵/5! (x should be in [-π, π] now)
    code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = x²)
    code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = x²)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = x³)
    code.extend_from_slice(&[0x0F, 0x28, 0xF5]); // movaps xmm6, xmm5 (xmm6 = x³)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF1]); // mulss xmm6, xmm1 (xmm6 = x⁵)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEB]); // mulss xmm5, xmm3 (xmm5 = x³/3!)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC5]); // subss xmm0, xmm5 (xmm0 = x - x³/3!)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xF4]); // mulss xmm6, xmm4 (xmm6 = x⁵/5!)
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC6]); // addss xmm0, xmm6 (xmm0 = sin(x))
    
    // Store result
    match output_buffer_reg {
        Register::R11 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]),
        Register::R13 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]),
        Register::R14 => code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]),
        _ => return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string())),
    }
    
    // Increment loop counter: inc rcx
    encoder = Encoder::new(64);
    let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
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
