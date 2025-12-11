//! Natural logarithm operation
//! Element-wise natural logarithm: output = log(input)
//! Uses polynomial approximation: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_log(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Log: output = log(input) (element-wise, natural logarithm)
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    // Uses polynomial approximation: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3
    
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
                // Log polynomial: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3
                // xmm0 contains input x
                
                // Load constants: 1.0 in xmm2, 0.5 in xmm3, 1/3 in xmm4
                code.push(0xB8);
                code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
                
                code.push(0xB8);
                code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
                
                code.push(0xB8);
                code.extend_from_slice(&0x3EAAAAABu32.to_le_bytes()); // 1/3 ≈ 0.333333f
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xE0]); // movd xmm4, eax
                
                // Compute (x-1): subss xmm0, xmm2
                code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC2]); // subss xmm0, xmm2 (xmm0 = x - 1)
                
                // Save (x-1) in xmm1
                code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x-1)
                
                // Compute (x-1)²: xmm1 = (x-1) * (x-1)
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = (x-1)²)
                
                // Compute (x-1)³: xmm5 = (x-1)² * (x-1)
                code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = (x-1)²)
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = (x-1)³)
                
                // Compute: (x-1) - (x-1)²/2 + (x-1)³/3
                // xmm0 = (x-1), subtract (x-1)²/2
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCB]); // mulss xmm1, xmm3 (xmm1 = (x-1)²/2)
                code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC1]); // subss xmm0, xmm1 (xmm0 = (x-1) - (x-1)²/2)
                
                // Add (x-1)³/3
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEC]); // mulss xmm5, xmm4 (xmm5 = (x-1)³/3)
                code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5 (xmm0 = log(x))
            }
        ));
    }
    
    // 1D case: simple linear loop
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Load constants BEFORE loop: 1.0 in xmm2, 0.5 in xmm3, 1/3 in xmm4
    code.push(0xB8);
    code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3EAAAAABu32.to_le_bytes()); // 1/3 ≈ 0.333333f
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
    
    // Log polynomial: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC2]); // subss xmm0, xmm2 (xmm0 = x - 1)
    code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0 (save x-1)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xC8]); // mulss xmm1, xmm0 (xmm1 = (x-1)²)
    code.extend_from_slice(&[0x0F, 0x28, 0xE9]); // movaps xmm5, xmm1 (xmm5 = (x-1)²)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE8]); // mulss xmm5, xmm0 (xmm5 = (x-1)³)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xCB]); // mulss xmm1, xmm3 (xmm1 = (x-1)²/2)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xC1]); // subss xmm0, xmm1 (xmm0 = (x-1) - (x-1)²/2)
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xEC]); // mulss xmm5, xmm4 (xmm5 = (x-1)³/3)
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC5]); // addss xmm0, xmm5 (xmm0 = log(x))
    
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
