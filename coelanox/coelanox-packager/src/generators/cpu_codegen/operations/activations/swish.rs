//! Swish operation
//! Element-wise Swish: swish(x) = x * sigmoid(x)

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_swish(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // Swish: x * sigmoid(x)
    // Using fast sigmoid approximation: sigmoid(x) ≈ 0.5 * (1 + x / (1 + |x|))
    // So swish(x) = x * 0.5 * (1 + x / (1 + |x|))
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
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
                // Swish calculation: x * sigmoid(x)
                // xmm0 contains input x
                
                // Load constants: 1.0 in xmm2, 0.5 in xmm3
                code.push(0xB8);
                code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
                
                code.push(0xB8);
                code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
                code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
                
                // Save original x in xmm1 (for final multiplication)
                code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0
                
                // Compute |x|: xmm4 = -x, then xmm0 = max(x, -x) = |x|
                code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xE4]); // pxor xmm4, xmm4 (zero)
                code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xE0]); // subss xmm4, xmm0 (xmm4 = -x)
                code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC4]); // maxss xmm0, xmm4 (xmm0 = |x|)
                
                // Compute 1 + |x|
                code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (xmm0 = 1 + |x|)
                
                // Compute x / (1 + |x|): restore original x to xmm4, then divide
                code.extend_from_slice(&[0x0F, 0x28, 0xE1]); // movaps xmm4, xmm1 (restore x)
                code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xE0]); // divss xmm4, xmm0 (xmm4 = x / (1 + |x|))
                
                // Compute 1 + x/(1+|x|)
                code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xE2]); // addss xmm4, xmm2 (xmm4 = 1 + x/(1+|x|))
                
                // Multiply by 0.5: mulss xmm4, xmm3
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE3]); // mulss xmm4, xmm3 (xmm4 = sigmoid(x))
                
                // Multiply by original x: mulss xmm4, xmm1
                code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE1]); // mulss xmm4, xmm1 (xmm4 = x * sigmoid(x))
                
                // Move result to xmm0 for output
                code.extend_from_slice(&[0x0F, 0x28, 0xC4]); // movaps xmm0, xmm4
            }
        ));
    }
    
    // 1D case: use generic 1D loop helper would be ideal, but for now use simple loop
    // For 1D, we'll use a simple approach similar to 2D but with single loop
    let mut code = Vec::new();
    let mut encoder = iced_x86::Encoder::new(64);
    
    // Load constants: 1.0 in xmm2, 0.5 in xmm3
    code.push(0xB8);
    code.extend_from_slice(&0x3F800000u32.to_le_bytes()); // 1.0f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
    
    code.push(0xB8);
    code.extend_from_slice(&0x3F000000u32.to_le_bytes()); // 0.5f
    code.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD8]); // movd xmm3, eax
    
    // Initialize loop counter: rcx = 0
    encoder = iced_x86::Encoder::new(64);
    let inst = iced_x86::Instruction::with2(iced_x86::Code::Xor_rm64_r64, iced_x86::Register::RCX, iced_x86::Register::RCX).unwrap();
    encoder.encode(&inst, 0).unwrap();
    code.extend_from_slice(&encoder.take_buffer());
    
    let loop_start = code.len();
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
    
    // Swish calculation: x * sigmoid(x)
    // Save original x in xmm1
    code.extend_from_slice(&[0x0F, 0x28, 0xC8]); // movaps xmm1, xmm0
    
    // Compute |x|: xmm4 = -x, then xmm0 = max(x, -x) = |x|
    code.extend_from_slice(&[0x66, 0x0F, 0xEF, 0xE4]); // pxor xmm4, xmm4 (zero)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5C, 0xE0]); // subss xmm4, xmm0 (xmm4 = -x)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5F, 0xC4]); // maxss xmm0, xmm4 (xmm0 = |x|)
    
    // Compute 1 + |x|
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC2]); // addss xmm0, xmm2 (xmm0 = 1 + |x|)
    
    // Compute x / (1 + |x|): restore original x to xmm4, then divide
    code.extend_from_slice(&[0x0F, 0x28, 0xE1]); // movaps xmm4, xmm1 (restore x)
    code.extend_from_slice(&[0xF3, 0x0F, 0x5E, 0xE0]); // divss xmm4, xmm0 (xmm4 = x / (1 + |x|))
    
    // Compute 1 + x/(1+|x|)
    code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xE2]); // addss xmm4, xmm2 (xmm4 = 1 + x/(1+|x|))
    
    // Multiply by 0.5: mulss xmm4, xmm3
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE3]); // mulss xmm4, xmm3 (xmm4 = sigmoid(x))
    
    // Multiply by original x: mulss xmm4, xmm1
    code.extend_from_slice(&[0xF3, 0x0F, 0x59, 0xE1]); // mulss xmm4, xmm1 (xmm4 = x * sigmoid(x))
    
    // Move result to xmm0 for output
    code.extend_from_slice(&[0x0F, 0x28, 0xC4]); // movaps xmm0, xmm4
    
    // Store: movss [output_buffer_reg + rax], xmm0
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
