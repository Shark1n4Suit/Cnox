//! Hardware detection for SAIC inference engine

/// Hardware information for backend selection
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub has_cpu: bool,
    pub has_gpu: bool,
    pub is_intel: bool,
    pub is_nvidia: bool,
    pub cpu_cores: usize,
    pub available_memory_mb: usize,
    pub cpu_features: Vec<String>,
    pub gpu_memory_mb: Option<usize>,
}

impl HardwareInfo {
    /// Detect hardware capabilities
    pub fn detect() -> Self {
        let cpu_info = Self::detect_cpu();
        let gpu_info = Self::detect_gpu();
        let memory_info = Self::detect_memory();
        
        Self {
            has_cpu: true, // Assume CPU is always available
            has_gpu: gpu_info.is_some(),
            is_intel: cpu_info.vendor.contains("Intel"),
            is_nvidia: gpu_info.as_ref().map(|g| g.vendor.contains("NVIDIA")).unwrap_or(false),
            cpu_cores: cpu_info.cores,
            available_memory_mb: memory_info,
            cpu_features: cpu_info.features,
            gpu_memory_mb: gpu_info.map(|g| g.memory_mb),
        }
    }
    
    /// Detect CPU information
    fn detect_cpu() -> CpuInfo {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        let features = vec![
            "AVX2".to_string(),
            "FMA".to_string(),
        ];
        
        // Detect vendor (simplified)
        let vendor = if cfg!(target_arch = "x86_64") {
            "Intel".to_string()
        } else {
            "Unknown".to_string()
        };
        
        CpuInfo {
            vendor,
            cores,
            features,
        }
    }
    
    /// Detect GPU information
    fn detect_gpu() -> Option<GpuInfo> {
        // Simplified GPU detection
        // In a real implementation, this would use nvidia-smi or similar
        None // For now, assume no GPU
    }
    
    /// Detect available memory
    fn detect_memory() -> usize {
        // Simplified memory detection
        // In a real implementation, this would read from /proc/meminfo
        8192 // Assume 8GB for now
    }
}

/// CPU information
#[derive(Debug, Clone)]
struct CpuInfo {
    vendor: String,
    cores: usize,
    features: Vec<String>,
}

/// GPU information
#[derive(Debug, Clone)]
struct GpuInfo {
    vendor: String,
    memory_mb: usize,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self::detect()
    }
} 