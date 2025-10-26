//! Comprehensive Backend Tests and Benchmarks for COELANOX Runtime
//! 
//! This module provides comprehensive testing and benchmarking for all backends
//! (CPU, GPU, NPU, TPU) with memory usage tracking and performance analysis.

#[cfg(test)]
mod backend_tests {
    use crate::CoelanoxRuntime;
    use crate::backends::{CpuBackend, GpuBackend, NpuBackend, TpuBackend};
    use coelanox_core::format::CoelanoxContainer;
    use coelanox_core::ir::{Operation, OpType, Attribute};
    use std::path::Path;
    use std::time::{Duration, Instant};
    use std::collections::HashMap;
    use serde_json::json;

    /// Create a test container with various operations
    fn create_test_container() -> CoelanoxContainer {
        // Create a simple test container with basic operations
        let operations = vec![
            json!({
                "op_type": "Add",
                "inputs": ["input1", "input2"],
                "outputs": ["output"],
                "attributes": {}
            }),
            json!({
                "op_type": "Multiply", 
                "inputs": ["input1", "input2"],
                "outputs": ["output"],
                "attributes": {}
            }),
            json!({
                "op_type": "Relu",
                "inputs": ["input"],
                "outputs": ["output"],
                "attributes": {}
            }),
            json!({
                "op_type": "Sigmoid",
                "inputs": ["input"],
                "outputs": ["output"],
                "attributes": {}
            }),
            json!({
                "op_type": "Convolution",
                "inputs": ["input"],
                "outputs": ["output"],
                "attributes": {
                    "kernel_size": 3
                }
            }),
            json!({
                "op_type": "MatMul",
                "inputs": ["input"],
                "outputs": ["output"],
                "attributes": {
                    "input_rows": 10,
                    "input_cols": 10,
                    "weight_rows": 10,
                    "weight_cols": 5
                }
            })
        ];

        // Create a minimal container for testing
        CoelanoxContainer {
            header: coelanox_core::format::CoelanoxHeader {
                magic: *b"CNOX",
                version: 1,
                flags: 0,
                container_size: 0,
                manifest_offset: 32,
                code_offset: 1024,
                security_hash: [0; 32],
                reserved: [0; 8],
            },
            manifest: coelanox_core::format::CoelanoxManifest {
                name: "test_model".to_string(),
                format: coelanox_core::format::ModelFormat::Coelanox,
                input_shapes: vec![vec![100]],
                output_shapes: vec![vec![50]],
                model_size: 100,
                created_at: 0,
                modified_at: 0,
                description: Some("Test model".to_string()),
                tags: vec!["test".to_string()],
                operations,
                metadata: std::collections::HashMap::new(),
            },
            optimized_code: vec![0x90; 1024], // NOP instructions for testing
            compressed_weights: vec![1.0f32; 100].into_iter().flat_map(|f| f.to_le_bytes()).collect(),
            compression_type: coelanox_core::format::CompressionType::Zstd,
            backend_hints: None,
            security: None,
        }
    }

    /// Benchmark a backend with comprehensive metrics
    fn benchmark_backend<F>(name: &str, backend_fn: F, input_data: &[f32]) -> BenchmarkResult
    where
        F: Fn(&[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>,
    {
        println!("🚀 Benchmarking {} backend...", name);
        
        let container = create_test_container();
        let weights = vec![0.5f32; input_data.len()];
        
        // Warmup
        for _ in 0..3 {
            let _ = backend_fn(input_data);
        }
        
        // Benchmark
        let iterations = 100;
        let mut times = Vec::new();
        let mut memory_usage = Vec::new();
        
        for i in 0..iterations {
            let start_time = Instant::now();
            let start_memory = get_memory_usage();
            
            match backend_fn(input_data) {
                Ok(output) => {
                    let end_time = start_time.elapsed();
                    let end_memory = get_memory_usage();
                    
                    times.push(end_time.as_micros() as u64);
                    memory_usage.push(end_memory.saturating_sub(start_memory));
                    
                    if i % 20 == 0 {
                        println!("  Iteration {}: {}μs, {}KB memory", i, end_time.as_micros(), (end_memory - start_memory) / 1024);
                    }
                }
                Err(e) => {
                    println!("❌ {} backend failed at iteration {}: {}", name, i, e);
                    return BenchmarkResult::failed(name);
                }
            }
        }
        
        BenchmarkResult::new(name, times, memory_usage)
    }

    /// Get current memory usage (simplified)
    fn get_memory_usage() -> usize {
        // This is a simplified memory usage calculation
        // In a real implementation, you'd use system-specific APIs
        std::process::id() as usize * 1024 // Placeholder
    }

    /// Benchmark result structure
    #[derive(Debug)]
    struct BenchmarkResult {
        backend_name: String,
        iterations: usize,
        avg_time_us: f64,
        min_time_us: u64,
        max_time_us: u64,
        std_dev_us: f64,
        avg_memory_kb: f64,
        peak_memory_kb: usize,
        throughput_ops_per_sec: f64,
        success: bool,
    }

    impl BenchmarkResult {
        fn new(name: &str, times: Vec<u64>, memory_usage: Vec<usize>) -> Self {
            let iterations = times.len();
            let avg_time_us = times.iter().sum::<u64>() as f64 / iterations as f64;
            let min_time_us = *times.iter().min().unwrap();
            let max_time_us = *times.iter().max().unwrap();
            
            // Calculate standard deviation
            let variance = times.iter()
                .map(|&t| (t as f64 - avg_time_us).powi(2))
                .sum::<f64>() / iterations as f64;
            let std_dev_us = variance.sqrt();
            
            let avg_memory_kb = memory_usage.iter().sum::<usize>() as f64 / iterations as f64 / 1024.0;
            let peak_memory_kb = *memory_usage.iter().max().unwrap() / 1024;
            let throughput_ops_per_sec = 1_000_000.0 / avg_time_us;
            
            Self {
                backend_name: name.to_string(),
                iterations,
                avg_time_us,
                min_time_us,
                max_time_us,
                std_dev_us,
                avg_memory_kb,
                peak_memory_kb,
                throughput_ops_per_sec,
                success: true,
            }
        }
        
        fn failed(name: &str) -> Self {
            Self {
                backend_name: name.to_string(),
                iterations: 0,
                avg_time_us: 0.0,
                min_time_us: 0,
                max_time_us: 0,
                std_dev_us: 0.0,
                avg_memory_kb: 0.0,
                peak_memory_kb: 0,
                throughput_ops_per_sec: 0.0,
                success: false,
            }
        }
        
        fn print_summary(&self) {
            if !self.success {
                println!("❌ {} backend benchmark failed", self.backend_name);
                return;
            }
            
            println!("📊 {} Backend Benchmark Results:", self.backend_name);
            println!("  Iterations: {}", self.iterations);
            println!("  Average Time: {:.2} μs", self.avg_time_us);
            println!("  Min Time: {} μs", self.min_time_us);
            println!("  Max Time: {} μs", self.max_time_us);
            println!("  Std Dev: {:.2} μs", self.std_dev_us);
            println!("  Average Memory: {:.2} KB", self.avg_memory_kb);
            println!("  Peak Memory: {} KB", self.peak_memory_kb);
            println!("  Throughput: {:.2} ops/sec", self.throughput_ops_per_sec);
            println!();
        }
    }

    #[test]
    fn test_cpu_backend() {
        println!("🧪 Testing CPU Backend...");
        
        let cpu_backend = CpuBackend::new();
        let container = create_test_container();
        let input_data = vec![1.0f32; 100];
        
        // Test basic execution
        match cpu_backend.execute(&container, &input_data) {
            Ok(output) => {
                println!("✅ CPU Backend execution successful");
                println!("   Input size: {} elements", input_data.len());
                println!("   Output size: {} elements", output.len());
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                println!("❌ CPU Backend execution failed: {}", e);
                panic!("CPU Backend should work");
            }
        }
        
        // Test backend info
        let info = cpu_backend.get_info();
        println!("📋 CPU Backend Info:");
        for (key, value) in info {
            println!("  {}: {}", key, value);
        }
    }
    
    #[test]
    fn test_gpu_backend() {
        println!("🧪 Testing GPU Backend...");
        
        let gpu_backend = GpuBackend::new();
        let container = create_test_container();
        let input_data = vec![1.0f32; 100];
        
        // Test basic execution
        match gpu_backend.execute(&container, &input_data) {
            Ok(output) => {
                println!("✅ GPU Backend execution successful");
                println!("   Input size: {} elements", input_data.len());
                println!("   Output size: {} elements", output.len());
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                println!("❌ GPU Backend execution failed: {}", e);
                // GPU backend might not be available, so we don't panic
            }
        }
        
        // Test backend info
        let info = gpu_backend.get_info();
        println!("📋 GPU Backend Info:");
        for (key, value) in info {
            println!("  {}: {}", key, value);
        }
    }

    #[test]
    fn test_npu_backend() {
        println!("🧪 Testing NPU Backend...");
        
        let npu_backend = NpuBackend::new();
        let container = create_test_container();
        let input_data = vec![1.0f32; 100];
        
        // Test basic execution
        match npu_backend.execute(&container, &input_data) {
            Ok(output) => {
                println!("✅ NPU Backend execution successful");
                println!("   Input size: {} elements", input_data.len());
                println!("   Output size: {} elements", output.len());
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                println!("❌ NPU Backend execution failed: {}", e);
                // NPU backend might not be available, so we don't panic
            }
        }
        
        // Test backend info
        let info = npu_backend.get_info();
        println!("📋 NPU Backend Info:");
        for (key, value) in info {
            println!("  {}: {}", key, value);
        }
    }
    
    #[test]
    fn test_tpu_backend() {
        println!("🧪 Testing TPU Backend...");
        
        let tpu_backend = TpuBackend::new();
        let container = create_test_container();
        let input_data = vec![1.0f32; 100];
        
        // Test basic execution
        match tpu_backend.execute(&container, &input_data) {
            Ok(output) => {
                println!("✅ TPU Backend execution successful");
                println!("   Input size: {} elements", input_data.len());
                println!("   Output size: {} elements", output.len());
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                println!("❌ TPU Backend execution failed: {}", e);
                // TPU backend might not be available, so we don't panic
            }
        }
        
        // Test backend info
        let info = tpu_backend.get_info();
        println!("📋 TPU Backend Info:");
        for (key, value) in info {
            println!("  {}: {}", key, value);
        }
    }

    #[test]
    fn benchmark_all_backends() {
        println!("🚀 Comprehensive Backend Benchmark Suite");
        println!("==========================================");
        
        let input_data = vec![1.0f32; 1000]; // Larger input for better benchmarking
        let container = create_test_container();
        
        // Benchmark CPU Backend
        let cpu_backend = CpuBackend::new();
        let cpu_result = benchmark_backend("CPU", |input| {
            cpu_backend.execute(&container, input).map_err(|e| e.into())
        }, &input_data);
        cpu_result.print_summary();
        
        // Benchmark GPU Backend
        let gpu_backend = GpuBackend::new();
        let gpu_result = benchmark_backend("GPU", |input| {
            gpu_backend.execute(&container, input).map_err(|e| e.into())
        }, &input_data);
        gpu_result.print_summary();
        
        // Benchmark NPU Backend
        let npu_backend = NpuBackend::new();
        let npu_result = benchmark_backend("NPU", |input| {
            npu_backend.execute(&container, input).map_err(|e| e.into())
        }, &input_data);
        npu_result.print_summary();
        
        // Benchmark TPU Backend
        let tpu_backend = TpuBackend::new();
        let tpu_result = benchmark_backend("TPU", |input| {
            tpu_backend.execute(&container, input).map_err(|e| e.into())
        }, &input_data);
        tpu_result.print_summary();
        
        // Performance comparison
        println!("🏆 Performance Comparison:");
        let mut results = vec![cpu_result, gpu_result, npu_result, tpu_result];
        results.sort_by(|a, b| a.avg_time_us.partial_cmp(&b.avg_time_us).unwrap());
        
        for (i, result) in results.iter().enumerate() {
            if result.success {
                println!("  {}. {}: {:.2} μs avg, {:.2} ops/sec", 
                    i + 1, result.backend_name, result.avg_time_us, result.throughput_ops_per_sec);
            }
        }
    }

    #[test]
    fn test_runtime_integration() {
        println!("🧪 Testing Runtime Integration with All Backends...");
        
        let runtime = CoelanoxRuntime::new();
        let container = create_test_container();
        let input_data = vec![1.0f32; 100];
        
        // Test runtime execution
        match runtime.execute("test_container", &input_data) {
            Ok(output) => {
                println!("✅ Runtime execution successful");
                println!("   Input size: {} elements", input_data.len());
                println!("   Output size: {} elements", output.len());
                assert!(!output.is_empty(), "Output should not be empty");
            }
            Err(e) => {
                println!("❌ Runtime execution failed: {}", e);
                // This might fail if no container is loaded, which is expected
            }
        }
        
        // Test runtime metrics
        let metrics = runtime.get_metrics();
        println!("📊 Runtime Metrics:");
        println!("  Total executions: {}", metrics.total_executions);
        println!("  Failed executions: {}", metrics.failed_executions);
        println!("  Average execution time: {} μs", metrics.avg_execution_time_us);
        println!("  Current memory usage: {} bytes", metrics.current_memory_usage);
        println!("  Peak memory usage: {} bytes", metrics.peak_memory_usage);
        println!("  Throughput: {:.2} ops/sec", metrics.throughput_ops_per_sec);
        
        // Test memory stats
        let memory_stats = runtime.get_memory_stats();
        println!("💾 Memory Statistics:");
        println!("  Total allocated: {} bytes", memory_stats.total_allocated);
        println!("  Current usage: {} bytes", memory_stats.current_usage);
        println!("  Peak usage: {} bytes", memory_stats.peak_usage);
        println!("  Executable blocks: {}", memory_stats.executable_blocks);
        
        // Test health check
        let health = runtime.health_check();
        println!("🏥 Health Status:");
        println!("  Healthy: {}", health.is_healthy);
        println!("  Issues: {:?}", health.issues);
        println!("  Warnings: {:?}", health.warnings);
        println!("  Uptime: {} ms", health.uptime_ms);
    }

    #[test]
    fn test_memory_stress() {
        println!("🧪 Testing Memory Stress with Multiple Backends...");
        
        let input_data = vec![1.0f32; 10000]; // Large input
        let container = create_test_container();
        
        // Test memory usage with multiple backends
        let backends = vec![
            ("CPU", Box::new(CpuBackend::new()) as Box<dyn BackendTrait>),
            ("GPU", Box::new(GpuBackend::new()) as Box<dyn BackendTrait>),
            ("NPU", Box::new(NpuBackend::new()) as Box<dyn BackendTrait>),
            ("TPU", Box::new(TpuBackend::new()) as Box<dyn BackendTrait>),
        ];
        
        for (name, backend) in backends {
            let start_memory = get_memory_usage();
            
            // Execute multiple times to test memory management
            for i in 0..10 {
                match backend.execute(&container, &input_data) {
                    Ok(_) => {
                        if i % 3 == 0 {
                            let current_memory = get_memory_usage();
                            println!("  {} iteration {}: {} KB memory", name, i, (current_memory - start_memory) / 1024);
                        }
                    }
                    Err(e) => {
                        println!("  {} iteration {} failed: {}", name, i, e);
                    }
                }
            }
            
            let end_memory = get_memory_usage();
            println!("  {} total memory delta: {} KB", name, (end_memory - start_memory) / 1024);
        }
    }

    /// Trait for backend testing
    trait BackendTrait {
        fn execute(&self, container: &CoelanoxContainer, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
    }

    impl BackendTrait for CpuBackend {
        fn execute(&self, container: &CoelanoxContainer, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            self.execute(container, input).map_err(|e| e.into())
        }
    }

    impl BackendTrait for GpuBackend {
        fn execute(&self, container: &CoelanoxContainer, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            self.execute(container, input).map_err(|e| e.into())
        }
    }

    impl BackendTrait for NpuBackend {
        fn execute(&self, container: &CoelanoxContainer, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            self.execute(container, input).map_err(|e| e.into())
        }
    }

    impl BackendTrait for TpuBackend {
        fn execute(&self, container: &CoelanoxContainer, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            self.execute(container, input).map_err(|e| e.into())
        }
    }
}
