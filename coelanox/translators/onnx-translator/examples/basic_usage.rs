//! Example usage of the COELANOX ONNX Translator
//! 
//! This example shows how to use the ONNX translator to convert
//! ONNX models into COELANOX's UniversalIR format.

use onnx_translator::OnnxTranslator;
use coelanox_core::plugins::TranslatorPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the translator
    let translator = OnnxTranslator::new();
    
    println!("Translator: {} v{} by {}", 
             translator.name(), 
             translator.version(), 
             translator.author());
    
    // Check supported formats
    let formats = translator.supported_formats();
    println!("Supported formats: {:?}", formats);
    
    // Example: Load an ONNX model (you would read from file in practice)
    let onnx_model_data = b""; // Empty for demo - replace with actual ONNX model bytes
    
    // Translate to UniversalIR
    match translator.translate(onnx_model_data) {
        Ok(ir) => {
            println!("Successfully translated ONNX model to UniversalIR");
            println!("Model name: {}", ir.metadata.name);
            println!("Model version: {}", ir.metadata.version);
            println!("Number of nodes: {}", ir.nodes.len());
            println!("Number of weight tensors: {}", ir.weight_tensors.len());
            println!("Number of intermediate tensors: {}", ir.intermediate_tensors.len());
        }
        Err(e) => {
            println!("Translation failed: {}", e);
        }
    }
    
    // Extract weights
    match translator.extract_weights(onnx_model_data) {
        Ok(weights) => {
            println!("Extracted {} weight values", weights.len());
        }
        Err(e) => {
            println!("Weight extraction failed: {}", e);
        }
    }
    
    Ok(())
}
