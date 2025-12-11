//! Add operation
//! Element-wise addition: output = input_a + input_b

use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::ir::{GraphNode, Shape};
use iced_x86::{Code, Instruction, Register, Encoder};
use crate::generators::cpu_codegen::CpuCodeGenerator;
use crate::generators::cpu_codegen::helpers::elementwise;

pub fn generate_add(generator: &CpuCodeGenerator, _node: &GraphNode, _input_shape: Option<&Shape>, _output_shape: Option<&Shape>) -> CoelanoxResult<Vec<u8>> {
    // ADD: element-wise addition
    // For binary ADD: input_a is at [input_buffer_reg], input_b is at [input_buffer_reg + r15*4]
    // output = input_a + input_b (element-wise)
    // Shape-aware: Supports 1D and 2D+ (flattened to 2D)
    
    let mut code = Vec::new();
    let mut encoder = Encoder::new(64);
    
    // Get buffer registers from generator context
    let input_buffer_reg = generator.get_current_input_buffer();
    let output_buffer_reg = generator.get_current_output_buffer();
    
    // Check if we have 2D shapes (or higher, flattened to 2D)
    if let Some((rows, cols)) = generator.get_2d_dimensions(_output_shape) {
        // 2D case: nested loops (rows, cols)
        return generate_add_2d(generator, input_buffer_reg, output_buffer_reg, rows, cols, _input_shape);
    }
        
        // 1D case: simple linear loop
        // Simple scalar loop: for i in 0..r15 (output_len)
        //   output[i] = input_a[i] + input_b[i]
        // where input_a = [input_buffer_reg + i*4]
        // and input_b = [input_buffer_reg + r15*4 + i*4]
        
        // Initialize loop counter: rcx = 0 (will count elements)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Xor_rm64_r64, Register::RCX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Loop start label
        let loop_start = code.len();
        
        // Check if done: cmp rcx, r15 (rcx >= output_len)
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Cmp_rm64_r64, Register::RCX, Register::R15).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump to end if done: jge loop_end
        let jge_pos = code.len();
        code.push(0x0F); code.push(0x8D); // jge (long form)
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00); // placeholder
        
        // Calculate byte offset: rax = rcx * 4
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Mov_r64_rm64, Register::RAX, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        code.push(0x48); code.push(0xC1); code.push(0xE0); code.push(0x02); // shl rax, 2
        
        // Load input_a: movss xmm0, [input_buffer_reg + rax]
        match input_buffer_reg {
            Register::R10 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x02]); // movss xmm0, [r10 + rax]
            }
            Register::R11 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x03]); // movss xmm0, [r11 + rax]
            }
            Register::R13 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x05]); // movss xmm0, [r13 + rax]
            }
            Register::R14 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x04, 0x06]); // movss xmm0, [r14 + rax]
            }
            _ => {
                return Err(CoelanoxError::Internal("Unsupported input buffer register".to_string()));
            }
        }
        
        // Calculate input_b address: r8 = r15 * 4 + input_buffer_reg + rax
        // mov r8, r15
        code.extend_from_slice(&[0x4D, 0x89, 0xF8]); // mov r8, r15
        // shl r8, 2
        code.extend_from_slice(&[0x49, 0xC1, 0xE0, 0x02]); // shl r8, 2
        // add r8, input_buffer_reg
        encoder = Encoder::new(64);
        let inst = Instruction::with2(Code::Add_rm64_r64, Register::R8, input_buffer_reg).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        // add r8, rax
        code.extend_from_slice(&[0x49, 0x01, 0xC0]); // add r8, rax
        
        // Load input_b: movss xmm1, [r8]
        code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x10, 0x08]); // movss xmm1, [r8]
        
        // ADD: addss xmm0, xmm1
        code.extend_from_slice(&[0xF3, 0x0F, 0x58, 0xC1]); // addss xmm0, xmm1
        
        // Store result: movss [output_buffer_reg + rax], xmm0
        match output_buffer_reg {
            Register::R11 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x03]); // movss [r11 + rax], xmm0
            }
            Register::R13 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x05]); // movss [r13 + rax], xmm0
            }
            Register::R14 => {
                code.extend_from_slice(&[0xF3, 0x41, 0x0F, 0x11, 0x04, 0x06]); // movss [r14 + rax], xmm0
            }
            _ => {
                return Err(CoelanoxError::Internal("Unsupported output buffer register".to_string()));
            }
        }
        
        // Increment counter: inc rcx
        encoder = Encoder::new(64);
        let inst = Instruction::with1(Code::Inc_rm64, Register::RCX).unwrap();
        encoder.encode(&inst, 0).unwrap();
        code.extend_from_slice(&encoder.take_buffer());
        
        // Jump back to loop start
        let jmp_pos = code.len();
        code.push(0xE9); // jmp
        let jmp_offset_placeholder = code.len();
        code.push(0x00); code.push(0x00); code.push(0x00); code.push(0x00); // placeholder
        
        let loop_end = code.len();
        
        // Patch jump offsets
        let jge_offset = (loop_end as i32 - (jge_pos + 6) as i32) as i32;
        code[jge_pos + 2..jge_pos + 6].copy_from_slice(&jge_offset.to_le_bytes());
        
        let jmp_offset = (loop_start as i32 - (jmp_pos + 5) as i32) as i32;
        code[jmp_offset_placeholder..jmp_offset_placeholder + 4].copy_from_slice(&jmp_offset.to_le_bytes());
        
        Ok(code)
}

/// Generate 2D ADD operation with nested loops
/// Uses the generic 2D element-wise helper
fn generate_add_2d(
    generator: &CpuCodeGenerator,
    input_buffer_reg: Register,
    output_buffer_reg: Register,
    rows: usize,
    cols: usize,
    _input_shape: Option<&Shape>
) -> CoelanoxResult<Vec<u8>> {
    let input_b_offset = rows * cols * 4; // input_b starts after input_a
    let code = elementwise::generate_2d_elementwise_binary(
        generator,
        input_buffer_reg,
        output_buffer_reg,
        rows,
        cols,
        input_b_offset,
        &[0xF3, 0x0F, 0x58, 0xC1] // addss xmm0, xmm1
    );
    Ok(code)
}

