//! Test the ONNX translator with the actual model.onnx file
//! 
//! This test loads the real ONNX model and translates it through the COELANOX pipeline.

use onnx_translator::OnnxTranslator;
use coelanox_core::plugins::TranslatorPlugin;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing COELANOX ONNX Translator with Real Model");
    println!("==================================================");
    
    // Create the translator
    let translator = OnnxTranslator::new();
    
    println!("✅ Translator created: {} v{} by {}", 
             translator.name(), 
             translator.version(), 
             translator.author());
    
    // Load the actual ONNX model
    let model_path = "/home/shark1n4uit/Saic/coelanox/models/input/model.onnx";
    println!("📁 Loading model from: {}", model_path);
    
    let model_data = match fs::read(model_path) {
        Ok(data) => {
            println!("✅ Model loaded successfully: {} bytes", data.len());
            data
        }
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            return Err(e.into());
        }
    };
    
    // Test translation to UniversalIR
    println!("\n🔄 Translating ONNX model to UniversalIR...");
    let start_time = std::time::Instant::now();
    
    match translator.translate(&model_data) {
        Ok(ir) => {
            let duration = start_time.elapsed();
            println!("✅ Translation successful! (took {:?})", duration);
            println!("   Model name: {}", ir.metadata.name);
            println!("   Model version: {}", ir.metadata.version);
            println!("   Number of nodes: {}", ir.nodes.len());
            println!("   Number of weight tensors: {}", ir.weight_tensors.len());
            println!("   Number of intermediate tensors: {}", ir.intermediate_tensors.len());
            
            // Show input/output shapes
            println!("   Input shapes: {:?}", ir.metadata.input_shapes);
            println!("   Output shapes: {:?}", ir.metadata.output_shapes);
            
            // Show some operation types
            let mut op_types = std::collections::HashSet::new();
            for node in &ir.nodes {
                match &node.op_type {
                    coelanox_core::ir::OpType::Custom(name) => {
                        op_types.insert(format!("Custom({})", name));
                    }
                    op => {
                        op_types.insert(format!("{:?}", op));
                    }
                }
            }
            println!("   Operation types found: {:?}", op_types);
            
            // Show some tensor details
            if !ir.weight_tensors.is_empty() {
                println!("   First weight tensor: {} {:?}", 
                         ir.weight_tensors[0].name, 
                         ir.weight_tensors[0].shape);
            }
            
            // Validate the IR
            println!("\n🔍 Validating UniversalIR...");
            match ir.validate() {
                Ok(_) => println!("✅ IR validation passed!"),
                Err(e) => {
                    println!("❌ IR validation failed: {}", e);
                    return Err(e.into());
                }
            }
            
            // Test weight extraction
            println!("\n🔄 Extracting weights...");
            let weight_start = std::time::Instant::now();
            
            match translator.extract_weights(&model_data) {
                Ok(weights) => {
                    let weight_duration = weight_start.elapsed();
                    println!("✅ Weight extraction successful! (took {:?})", weight_duration);
                    println!("   Extracted {} weight values", weights.len());
                    if !weights.is_empty() {
                        println!("   First few weights: {:?}", &weights[..weights.len().min(5)]);
                        println!("   Weight range: {} to {}", 
                                 weights.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                                 weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                    }
                }
                Err(e) => {
                    println!("❌ Weight extraction failed: {}", e);
                    return Err(e.into());
                }
            }
            
            println!("\n🎉 SUCCESS! The ONNX model has been successfully translated to UniversalIR!");
            println!("   The model is ready for optimization and packaging into a COELANOX container.");
            
        }
        Err(e) => {
            println!("❌ Translation failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}
