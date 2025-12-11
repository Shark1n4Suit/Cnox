// Quick tool to extract machine code from container and show assembly
use std::fs;
use std::path::Path;
use coelanox_core::format::CoelanoxContainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let container_file = if args.len() > 1 { &args[1] } else { "models/output/bert_model.cnox" };
    
    println!("Loading container: {}", container_file);
    let container = CoelanoxContainer::read_binary_file(Path::new(container_file))?;
    
    println!("Machine code size: {} bytes", container.optimized_code.len());
    
    // Save to file
    let output_file = "/tmp/coelanox_machine_code.bin";
    fs::write(output_file, &container.optimized_code)?;
    println!("Saved machine code to: {}", output_file);
    
    // Show first 200 bytes in hex
    println!("\n=== First 200 bytes (hex) ===");
    for (i, chunk) in container.optimized_code.iter().take(200).enumerate() {
        if i % 16 == 0 {
            print!("\n{:04x}: ", i);
        }
        print!("{:02x} ", chunk);
    }
    println!("\n");
    
    // Try to disassemble using capstone or show instruction patterns
    println!("=== Disassembly (manual analysis) ===");
    println!("Prologue should start with:");
    println!("  55                    push rbp");
    println!("  48 89 e5              mov rbp, rsp");
    println!("  49 89 fa              mov r10, rdi");
    println!("  53                    push rbx");
    println!("  41 54                 push r12");
    
    let prologue = &container.optimized_code[0..20];
    println!("\nActual prologue bytes:");
    for (i, b) in prologue.iter().enumerate() {
        print!("{:02x} ", b);
        if (i + 1) % 8 == 0 {
            println!();
        }
    }
    println!();
    
    // Check for expected patterns
    if prologue[0] == 0x55 {
        println!("✓ Starts with push rbp");
    }
    if prologue.len() >= 3 && prologue[1] == 0x48 && prologue[2] == 0x89 && prologue[3] == 0xe5 {
        println!("✓ mov rbp, rsp present");
    }
    
    Ok(())
}

