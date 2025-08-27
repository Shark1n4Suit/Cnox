//! Hardware detection for SAIC inference engine

use std::collections::HashMap;

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
    /// Detect hardware capabilities safely
    pub fn detect() -> Self {
        // Wrap detection in safe error handling
        let cpu_info = Self::detect_cpu_safe();
        let gpu_info = Self::detect_gpu_safe();
        let memory_info = Self::detect_memory_safe();
        
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
    
    /// Safe CPU detection with error handling
    fn detect_cpu_safe() -> CpuInfo {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        // Safe feature detection without unsafe assembly
        let features = Self::detect_cpu_features_safe();
        
        // Safe vendor detection
        let vendor = Self::detect_vendor_safe();
        
        CpuInfo {
            vendor,
            cores,
            features,
        }
    }
    
    /// Safe CPU feature detection
    fn detect_cpu_features_safe() -> Vec<String> {
        let mut features = Vec::new();
        
        // Use cfg attributes for compile-time feature detection
        #[cfg(target_arch = "x86_64")]
        {
            // Safe feature detection using cfg flags
            if cfg!(target_feature = "avx2") {
                features.push("AVX2".to_string());
            }
            if cfg!(target_feature = "fma") {
                features.push("FMA".to_string());
            }
            if cfg!(target_feature = "sse2") {
                features.push("SSE2".to_string());
            }
            if cfg!(target_feature = "sse4.2") {
                features.push("SSE4.2".to_string());
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            features.push("NEON".to_string());
        }
        
        // Always add basic features
        if features.is_empty() {
            features.push("Basic".to_string());
        }
        
        features
    }
    
    /// Safe vendor detection
    fn detect_vendor_safe() -> String {
        // Use compile-time detection where possible
        #[cfg(target_arch = "x86_64")]
        {
            // Try to detect vendor safely
            if let Ok(vendor) = Self::try_detect_vendor_from_cpuinfo() {
                return vendor;
            }
            
            // Fallback to architecture-based detection
            "Intel".to_string()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            "ARM".to_string()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Unknown".to_string()
        }
    }
    
    /// Try to detect vendor from /proc/cpuinfo (Linux only)
    #[cfg(target_os = "linux")]
    fn try_detect_vendor_from_cpuinfo() -> Result<String, std::io::Error> {
        use std::fs;
        use std::io::BufRead;
        
        let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
        for line in cpuinfo.lines() {
            if line.starts_with("vendor_id") {
                if line.contains("GenuineIntel") {
                    return Ok("Intel".to_string());
                } else if line.contains("AuthenticAMD") {
                    return Ok("AMD".to_string());
                }
            }
        }
        Ok("Unknown".to_string())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn try_detect_vendor_from_cpuinfo() -> Result<String, std::io::Error> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "Not supported on this platform"))
    }
    
    /// Safe GPU detection
    fn detect_gpu_safe() -> Option<GpuInfo> {
        // For now, return None to avoid crashes
        // In a real implementation, this would use safe GPU detection
        None
    }
    
    /// Safe memory detection
    fn detect_memory_safe() -> usize {
        // Safe memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(memory) = Self::try_detect_memory_from_proc() {
                return memory;
            }
        }
        
        // Fallback to reasonable default
        8192 // Assume 8GB
    }
    
    /// Try to detect memory from /proc/meminfo (Linux only)
    #[cfg(target_os = "linux")]
    fn try_detect_memory_from_proc() -> Result<usize, std::io::Error> {
        use std::fs;
        use std::io::BufRead;
        
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        return Ok(kb / 1024); // Convert KB to MB
                    }
                }
            }
        }
        Ok(8192) // Default fallback
    }
    
    #[cfg(not(target_os = "linux"))]
    fn try_detect_memory_from_proc() -> Result<usize, std::io::Error> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "Not supported on this platform"))
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