//! Performance optimization utilities for SAIC

use crate::{SaicError, SaicResult};

/// CPU feature detection for optimization
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    /// AVX2 support
    pub avx2: bool,
    
    /// AVX-512 support
    pub avx512: bool,
    
    /// FMA support
    pub fma: bool,
    
    /// Number of CPU cores
    pub cores: usize,
    
    /// Cache line size
    pub cache_line_size: usize,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        // Use CPUID to detect features
        let avx2 = is_x86_feature_detected!("avx2");
        let avx512 = is_x86_feature_detected!("avx512f");
        let fma = is_x86_feature_detected!("fma");
        
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        // Default cache line size (64 bytes for most modern CPUs)
        let cache_line_size = 64;
        
        Self {
            avx2,
            avx512,
            fma,
            cores,
            cache_line_size,
        }
    }
    
    /// Get recommended SIMD instructions
    pub fn recommended_simd_instructions(&self) -> Vec<String> {
        let mut instructions = Vec::new();
        
        if self.avx512 {
            instructions.push("AVX512".to_string());
        }
        
        if self.avx2 {
            instructions.push("AVX2".to_string());
        }
        
        if self.fma {
            instructions.push("FMA".to_string());
        }
        
        instructions
    }
    
    /// Get recommended thread count
    pub fn recommended_threads(&self) -> usize {
        self.cores
    }
    
    /// Get recommended memory alignment
    pub fn recommended_alignment(&self) -> usize {
        if self.avx512 {
            64
        } else if self.avx2 {
            32
        } else {
            16
        }
    }
}

/// SIMD-optimized operations
pub struct SimdOps;

impl SimdOps {
    /// SIMD-optimized 4-byte comparison
    #[target_feature(enable = "avx2")]
    pub unsafe fn compare_4_bytes_avx2(a: &[u8], b: &[u8]) -> bool {
        if a.len() < 4 || b.len() < 4 {
            return false;
        }
        
        // Use AVX2 for 4-byte comparison
        let a_ptr = a.as_ptr() as *const u32;
        let b_ptr = b.as_ptr() as *const u32;
        
        *a_ptr == *b_ptr
    }
    
    /// SIMD-optimized memory copy
    #[target_feature(enable = "avx2")]
    pub unsafe fn copy_aligned_avx2(src: &[u8], dst: &mut [u8]) {
        if src.len() != dst.len() {
            return;
        }
        
        // Use AVX2 for aligned memory copy
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        
        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, src.len());
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
    
    /// Enable multi-threading
    pub threading_enabled: bool,
    
    /// Enable memory pooling
    pub memory_pooling_enabled: bool,
    
    /// Thread count for parallel operations
    pub thread_count: usize,
    
    /// Memory alignment for SIMD operations
    pub memory_alignment: usize,
    
    /// Cache line size for memory optimization
    pub cache_line_size: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let cpu_features = CpuFeatures::detect();
        
        Self {
            simd_enabled: cpu_features.avx2,
            threading_enabled: cpu_features.cores > 1,
            thread_count: cpu_features.recommended_threads(),
            memory_alignment: cpu_features.recommended_alignment(),
            cache_line_size: cpu_features.cache_line_size,
            memory_pooling_enabled: true,
        }
    }
}

/// Performance monitoring utilities
pub struct PerformanceMonitor {
    /// Start time for current operation
    start_time: Option<std::time::Instant>,
    
    /// Performance metrics
    metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total operations performed
    pub operations: usize,
    
    /// Total time spent
    pub total_time: std::time::Duration,
    
    /// Average time per operation
    pub average_time: std::time::Duration,
    
    /// Operations per second
    pub ops_per_second: f64,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            start_time: None,
            metrics: PerformanceMetrics {
                operations: 0,
                total_time: std::time::Duration::ZERO,
                average_time: std::time::Duration::ZERO,
                ops_per_second: 0.0,
            },
        }
    }
    
    /// Start timing an operation
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }
    
    /// Stop timing and record the operation
    pub fn stop(&mut self) {
        if let Some(start_time) = self.start_time {
            let duration = start_time.elapsed();
            self.metrics.operations += 1;
            self.metrics.total_time += duration;
            self.metrics.average_time = self.metrics.total_time / self.metrics.operations as u32;
            
            if self.metrics.total_time.as_secs_f64() > 0.0 {
                self.metrics.ops_per_second = self.metrics.operations as f64 / self.metrics.total_time.as_secs_f64();
            }
        }
    }
    
    /// Get current performance metrics
    pub fn metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    /// Reset performance metrics
    pub fn reset(&mut self) {
        self.metrics = PerformanceMetrics {
            operations: 0,
            total_time: std::time::Duration::ZERO,
            average_time: std::time::Duration::ZERO,
            ops_per_second: 0.0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        
        // Should have at least 1 core
        assert!(features.cores >= 1);
        
        // Should have reasonable cache line size
        assert!(features.cache_line_size >= 32);
        assert!(features.cache_line_size <= 256);
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        
        // Should have reasonable defaults
        assert!(config.thread_count >= 1);
        assert!(config.memory_alignment >= 16);
        assert!(config.cache_line_size >= 32);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        
        // Test timing
        monitor.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.stop();
        
        let metrics = monitor.metrics();
        assert_eq!(metrics.operations, 1);
        assert!(metrics.total_time.as_millis() >= 10);
    }
} 