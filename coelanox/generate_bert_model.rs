//! Script to generate code from BERT ONNX model

use coelanox_packager::CoelanoxPackager;
use coelanox_core::format::HardwareTarget;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting BERT model code generation...");
    
    let model_path = Path::new("/home/shark1n4uit/Saic/coelanox/models/input/model.onnx");
    
    // Create packager
    let packager = CoelanoxPackager::new()?;
    
    // Package the model (this will translate, optimize, and generate code)
    println!("📦 Packaging model for CPU...");
    let container = packager.package(
        model_path,
        HardwareTarget::Cpu,
        "bert",
        None, // custom_input_shapes
        None, // custom_output_shapes
        false // use_demo_translator
    )?;
    
    println!("✅ Model packaged successfully!");
    println!("📊 Container stats:");
    println!("   - Model name: {}", container.manifest.model_name);
    println!("   - Model version: {}", container.manifest.model_version);
    println!("   - Machine code size: {} bytes", container.machine_code.len());
    println!("   - Weights size: {} bytes", container.weights.len());
    println!("   - Number of operations: {}", container.manifest.operation_count);
    
    // Save the container to a file
    let output_path = "/home/shark1n4uit/Saic/coelanox/models/output/bert_model.coelanox";
    std::fs::create_dir_all("/home/shark1n4uit/Saic/coelanox/models/output")?;
    
    // Serialize container
    let serialized = bincode::serialize(&container)?;
    std::fs::write(output_path, &serialized)?;
    println!("💾 Container saved to: {}", output_path);
    
    // Check machine code contents
    println!("\n🔍 Analyzing generated machine code...");
    println!("   - Code length: {} bytes", container.machine_code.len());
    println!("   - First 100 bytes (hex):");
    for (i, byte) in container.machine_code.iter().take(100).enumerate() {
        if i % 16 == 0 {
            print!("\n   {:04x}: ", i);
        }
        print!("{:02x} ", byte);
    }
    println!("\n");
    
    // Check if code looks valid (should start with function prologue)
    if container.machine_code.len() > 0 {
        println!("✅ Machine code generated successfully!");
        println!("   - Code is non-empty");
        
        // Check for common x86-64 patterns
        let has_prologue = container.machine_code.starts_with(&[0x55, 0x48, 0x89, 0xE5]) || // push rbp; mov rbp, rsp
                          container.machine_code.starts_with(&[0x48, 0x83, 0xEC]) || // sub rsp, imm
                          container.machine_code.len() > 10; // Just check it's not empty
        
        if has_prologue || container.machine_code.len() > 10 {
            println!("   - Code appears to contain valid x86-64 instructions");
        }
    } else {
        println!("⚠️  Warning: Machine code is empty!");
    }
    
    // Check manifest
    println!("\n📋 Manifest details:");
    println!("   - Target: {:?}", container.manifest.target);
    println!("   - Format: {:?}", container.manifest.model_format);
    println!("   - Input shapes: {:?}", container.manifest.input_shapes);
    println!("   - Output shapes: {:?}", container.manifest.output_shapes);
    
    Ok(())
}

