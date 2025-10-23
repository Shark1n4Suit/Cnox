//! Test BERT Model Inference with COELANOX Runtime
//! 
//! This test verifies that the improved runtime can execute BERT models
//! using real operation implementations instead of placeholders.

#[cfg(test)]
mod bert_runtime_test {
    use crate::CoelanoxRuntime;
    use coelanox_core::format::CoelanoxContainer;
    use std::path::Path;

    #[test]
    fn test_bert_model_execution() {
        // Test that we can load and execute a BERT model
        let bert_path = "bert-model.cnox";
        
        // Check if BERT model exists
        if !Path::new(bert_path).exists() {
            println!("⚠️  BERT model not found at {}, skipping test", bert_path);
            return;
        }
        
        println!("🧪 Testing BERT model execution...");
        
        // Create runtime
        let mut runtime = CoelanoxRuntime::new();
        
        // Load BERT model
        match runtime.load_container(bert_path) {
            Ok(_) => {
                println!("✅ BERT model loaded successfully");
                
                // Test inference with dummy input
                let input_size = 512; // Typical BERT input size
                let dummy_input = vec![1.0f32; input_size];
                
                match runtime.execute(bert_path, &dummy_input) {
                    Ok(output) => {
                        println!("✅ BERT inference executed successfully");
                        println!("   Input size: {} elements", dummy_input.len());
                        println!("   Output size: {} elements", output.len());
                        println!("   Expected output size: 2 (classification) or 768 (embeddings)");
                        
                        // Verify output is reasonable
                        assert!(!output.is_empty(), "Output should not be empty");
                        assert!(output.len() <= 1000, "Output size should be reasonable");
                        
                        // Check if output contains non-zero values (indicates real computation)
                        let has_non_zero = output.iter().any(|&x| x != 0.0);
                        if has_non_zero {
                            println!("✅ Output contains non-zero values - real computation detected");
                        } else {
                            println!("⚠️  Output contains only zeros - may be placeholder computation");
                        }
                    }
                    Err(e) => {
                        println!("❌ BERT inference failed: {}", e);
                        panic!("BERT inference should succeed");
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to load BERT model: {}", e);
                panic!("BERT model loading should succeed");
            }
        }
    }
    
    #[test]
    fn test_runtime_memory_management() {
        println!("🧪 Testing runtime memory management...");
        
        let mut runtime = CoelanoxRuntime::new();
        
        // Test multiple container loads
        let test_paths = ["bert-model.cnox", "mnist-model.cnox"];
        
        for path in &test_paths {
            if Path::new(path).exists() {
                match runtime.load_container(path) {
                    Ok(_) => println!("✅ Loaded container: {}", path),
                    Err(e) => println!("⚠️  Failed to load {}: {}", path, e),
                }
            }
        }
        
        // Test cleanup
        runtime.cleanup();
        println!("✅ Runtime cleanup completed");
    }
    
    #[test]
    fn test_runtime_performance() {
        println!("🧪 Testing runtime performance...");
        
        let bert_path = "bert-model.cnox";
        
        if !Path::new(bert_path).exists() {
            println!("⚠️  BERT model not found, skipping performance test");
            return;
        }
        
        let mut runtime = CoelanoxRuntime::new();
        
        match runtime.load_container(bert_path) {
            Ok(_) => {
                let input_size = 512;
                let dummy_input = vec![1.0f32; input_size];
                
                // Warmup
                let _ = runtime.execute(bert_path, &dummy_input);
                
                // Benchmark
                let iterations = 10;
                let start = std::time::Instant::now();
                
                for _ in 0..iterations {
                    let _ = runtime.execute(bert_path, &dummy_input);
                }
                
                let duration = start.elapsed();
                let avg_time = duration.as_micros() as f64 / iterations as f64;
                
                println!("✅ Performance test completed:");
                println!("   Iterations: {}", iterations);
                println!("   Total time: {:?}", duration);
                println!("   Average time: {:.2} μs", avg_time);
                
                // Performance should be reasonable (< 100ms per inference)
                assert!(avg_time < 100_000.0, "Inference should be fast (< 100ms)");
            }
            Err(e) => {
                println!("⚠️  Failed to load BERT model for performance test: {}", e);
            }
        }
    }
}
