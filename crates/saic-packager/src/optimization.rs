//! SAIC Hardware Optimization
//! 
//! Handles hardware detection and performance optimization

use saic_core::optimization::CpuFeatures;
use tracing::info;

/// Hardware optimization configuration
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub cpu_features: CpuFeatures,
    pub optimization_level: u8,
    pub memory_alignment: usize,
    pub cache_line_size: usize,
    pub recommended_threads: usize,
}

impl HardwareConfig {
    /// Create hardware config with current system detection
    pub fn detect() -> Self {
        let cpu_features = CpuFeatures::detect();
        
        info!("Hardware detection completed:");
        info!("  CPU cores: {}", cpu_features.cores);
        info!("  AVX2: {}", cpu_features.avx2);
        info!("  AVX512: {}", cpu_features.avx512);
        info!("  FMA: {}", cpu_features.fma);
        
        Self {
            cpu_features: cpu_features.clone(),
            optimization_level: 2, // Default balanced optimization
            memory_alignment: cpu_features.recommended_alignment(),
            cache_line_size: cpu_features.cache_line_size,
            recommended_threads: cpu_features.recommended_threads(),
        }
    }
    
    /// Create hardware config with custom settings
    pub fn custom(
        optimization_level: u8,
        memory_alignment: usize,
        cache_line_size: usize,
        recommended_threads: usize,
    ) -> Self {
        let cpu_features = CpuFeatures::detect();
        
        Self {
            cpu_features,
            optimization_level,
            memory_alignment,
            cache_line_size,
            recommended_threads,
        }
    }
    
    /// Get recommended SIMD instructions
    pub fn recommended_simd_instructions(&self) -> Vec<String> {
        self.cpu_features.recommended_simd_instructions()
    }
    
    /// Get optimization level description
    pub fn optimization_level_description(&self) -> &str {
        match self.optimization_level {
            0 => "None - No optimization",
            1 => "Minimal - Basic optimization",
            2 => "Balanced - Good performance/size ratio",
            3 => "Maximum - Best performance",
            _ => "Unknown - Invalid optimization level",
        }
    }
    
    /// Check if hardware supports specific optimization
    pub fn supports_optimization(&self, optimization: &str) -> bool {
        match optimization {
            "avx2" => self.cpu_features.avx2,
            "avx512" => self.cpu_features.avx512,
            "fma" => self.cpu_features.fma,
            "multithreading" => self.recommended_threads > 1,
            _ => false,
        }
    }
    
    /// Get optimal batch size for this hardware
    pub fn optimal_batch_size(&self) -> usize {
        match self.optimization_level {
            0 => 1,
            1 => 4,
            2 => 8,
            3 => 16,
            _ => 8,
        }
    }
    
    /// Get memory pool size for this hardware
    pub fn memory_pool_size(&self) -> usize {
        let base_size = 1024 * 1024; // 1MB base
        let multiplier = match self.optimization_level {
            0 => 1,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => 4,
        };
        
        base_size * multiplier
    }
}

/// Performance optimization utilities
pub struct OptimizationUtils;

impl OptimizationUtils {
    /// Get optimal thread count for current hardware
    pub fn optimal_thread_count() -> usize {
        let config = HardwareConfig::detect();
        config.recommended_threads
    }
    
    /// Get optimal memory alignment for current hardware
    pub fn optimal_memory_alignment() -> usize {
        let config = HardwareConfig::detect();
        config.memory_alignment
    }
    
    /// Check if current hardware supports AVX2
    pub fn supports_avx2() -> bool {
        let config = HardwareConfig::detect();
        config.cpu_features.avx2
    }
    
    /// Check if current hardware supports AVX512
    pub fn supports_avx512() -> bool {
        let config = HardwareConfig::detect();
        config.cpu_features.avx512
    }
    
    /// Get hardware summary for logging
    pub fn hardware_summary() -> String {
        let config = HardwareConfig::detect();
        format!(
            "Hardware: {} cores, {} alignment, {} cache line, {} optimization",
            config.recommended_threads,
            config.memory_alignment,
            config.cache_line_size,
            config.optimization_level_description()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_config_detection() {
        let config = HardwareConfig::detect();
        
        assert!(config.recommended_threads > 0);
        assert!(config.memory_alignment > 0);
        assert!(config.cache_line_size > 0);
        assert!(config.optimization_level <= 3);
    }

    #[test]
    fn test_hardware_config_custom() {
        let config = HardwareConfig::custom(3, 64, 128, 8);
        
        assert_eq!(config.optimization_level, 3);
        assert_eq!(config.memory_alignment, 64);
        assert_eq!(config.cache_line_size, 128);
        assert_eq!(config.recommended_threads, 8);
    }

    #[test]
    fn test_optimization_level_description() {
        let config = HardwareConfig::custom(2, 32, 64, 4);
        let description = config.optimization_level_description();
        
        assert!(description.contains("Balanced"));
    }

    #[test]
    fn test_optimal_batch_size() {
        let config = HardwareConfig::custom(2, 32, 64, 4);
        let batch_size = config.optimal_batch_size();
        
        assert_eq!(batch_size, 8);
    }

    #[test]
    fn test_memory_pool_size() {
        let config = HardwareConfig::custom(2, 32, 64, 4);
        let pool_size = config.memory_pool_size();
        
        assert_eq!(pool_size, 4 * 1024 * 1024); // 4MB
    }

    #[test]
    fn test_optimization_utils() {
        let thread_count = OptimizationUtils::optimal_thread_count();
        let alignment = OptimizationUtils::optimal_memory_alignment();
        
        assert!(thread_count > 0);
        assert!(alignment > 0);
    }
} 