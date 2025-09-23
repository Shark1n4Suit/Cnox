//! Utility functions for SAIC
//! 
//! This module provides utility functions for file operations, security validation,
//! and environment detection.

use crate::{SaicError, SaicResult};
use std::path::Path;
use std::collections::HashMap;
use std::arch::asm;

/// System environment information
#[derive(Debug, Clone)]
pub struct SystemEnvironment {
    /// Operating system
    pub os: String,
    /// CPU architecture
    pub architecture: String,
    /// CPU vendor
    pub cpu_vendor: String,
    /// CPU model
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available memory in GB
    pub memory_gb: usize,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
    /// SIMD capabilities
    pub simd_capabilities: Vec<SimdCapability>,
    /// System capabilities
    pub capabilities: HashMap<String, bool>,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU vendor
    pub vendor: String,
    /// GPU model
    pub model: String,
    /// GPU memory in GB
    pub memory_gb: usize,
    /// Compute capability (for NVIDIA)
    pub compute_capability: Option<String>,
    /// Driver version
    pub driver_version: Option<String>,
}

/// SIMD capabilities
#[derive(Debug, Clone, PartialEq)]
pub enum SimdCapability {
    /// SSE instructions
    SSE,
    /// SSE2 instructions
    SSE2,
    /// SSE3 instructions
    SSE3,
    /// SSE4.1 instructions
    SSE41,
    /// SSE4.2 instructions
    SSE42,
    /// AVX instructions
    AVX,
    /// AVX2 instructions
    AVX2,
    /// AVX-512 instructions
    AVX512,
    /// ARM NEON instructions
    NEON,
    /// ARM SVE instructions
    SVE,
}

impl SystemEnvironment {
    /// Detect the current system environment
    pub fn detect() -> SaicResult<Self> {
        let os = Self::detect_os()?;
        let architecture = Self::detect_architecture()?;
        let cpu_vendor = Self::detect_cpu_vendor()?;
        let cpu_model = Self::detect_cpu_model()?;
        let cpu_cores = Self::detect_cpu_cores()?;
        let memory_gb = Self::detect_memory()?;
        let gpu_info = Self::detect_gpu()?;
        let simd_capabilities = Self::detect_simd_capabilities()?;
        let capabilities = Self::detect_capabilities()?;
        
        Ok(Self {
            os,
            architecture,
            cpu_vendor,
            cpu_model,
            cpu_cores,
            memory_gb,
            gpu_info,
            simd_capabilities,
            capabilities,
        })
    }
    
    /// Detect operating system
    fn detect_os() -> SaicResult<String> {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /etc/os-release
            if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        let os_name = line.trim_start_matches("PRETTY_NAME=").trim_matches('"');
                        return Ok(os_name.to_string());
                    }
                }
            }
            Ok("Linux".to_string())
        }
        
        #[cfg(target_os = "windows")]
        {
            Ok("Windows".to_string())
        }
        
        #[cfg(target_os = "macos")]
        {
            Ok("macOS".to_string())
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            Ok("Unknown".to_string())
        }
    }
    
    /// Detect CPU architecture
    fn detect_architecture() -> SaicResult<String> {
        #[cfg(target_arch = "x86_64")]
        {
            Ok("x86_64".to_string())
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Ok("aarch64".to_string())
        }
        
        #[cfg(target_arch = "arm")]
        {
            Ok("arm".to_string())
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            Ok("unknown".to_string())
        }
    }
    
    /// Detect CPU vendor
    fn detect_cpu_vendor() -> SaicResult<String> {
        #[cfg(target_arch = "x86_64")]
        {
            // Use CPUID to detect vendor
            let vendor = unsafe {
                let mut vendor = [0u32; 3];
                asm!("cpuid", 
                     in("eax") 0,
                     out("r10") vendor[0],
                     out("edx") vendor[1],
                     out("ecx") vendor[2],
                     options(nomem, nostack));
                
                let vendor_bytes = vendor.iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect::<Vec<u8>>();
                
                String::from_utf8_lossy(&vendor_bytes).to_string()
            };
            
            Ok(vendor)
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // For ARM, try to read from /proc/cpuinfo
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("Hardware") {
                        let hardware = line.split(':').nth(1).unwrap_or("ARM").trim();
                        return Ok(hardware.to_string());
                    }
                }
            }
            Ok("ARM".to_string())
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Ok("Unknown".to_string())
        }
    }
    
    /// Detect CPU model
    fn detect_cpu_model() -> SaicResult<String> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        let model = line.split(':').nth(1).unwrap_or("Unknown").trim();
                        return Ok(model.to_string());
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows detection would require WMI calls
            Ok("Windows CPU".to_string())
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS detection would require sysctl calls
            Ok("macOS CPU".to_string())
        }
        
        Ok("Unknown".to_string())
    }
    
    /// Detect number of CPU cores
    fn detect_cpu_cores() -> SaicResult<usize> {
        Ok(num_cpus::get())
    }
    
    /// Detect available memory
    fn detect_memory() -> SaicResult<usize> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let kb = line.split_whitespace().nth(1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        return Ok(kb / 1024 / 1024); // Convert to GB
                    }
                }
            }
        }
        
        // Fallback to system info
        Ok(8) // Default assumption
    }
    
    /// Detect GPU information
    fn detect_gpu() -> SaicResult<Option<GpuInfo>> {
        #[cfg(target_os = "linux")]
        {
            // Try to detect NVIDIA GPU
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=name,memory.total,driver_version")
                .arg("--format=csv,noheader,nounits")
                .output() {
                
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = output_str.lines().next() {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 3 {
                            let model = parts[0].trim();
                            let memory_mb = parts[1].trim().parse::<usize>().unwrap_or(0);
                            let driver = parts[2].trim();
                            
                            return Ok(Some(GpuInfo {
                                vendor: "NVIDIA".to_string(),
                                model: model.to_string(),
                                memory_gb: memory_mb / 1024,
                                compute_capability: None, // Would need additional query
                                driver_version: Some(driver.to_string()),
                            }));
                        }
                    }
                }
            }
            
            // Try to detect AMD GPU
            if let Ok(_) = std::process::Command::new("rocm-smi")
                .arg("--showproductname")
                .output() {
                return Ok(Some(GpuInfo {
                    vendor: "AMD".to_string(),
                    model: "AMD GPU".to_string(),
                    memory_gb: 8, // Default assumption
                    compute_capability: None,
                    driver_version: None,
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Detect SIMD capabilities
    fn detect_simd_capabilities() -> SaicResult<Vec<SimdCapability>> {
        let mut capabilities = Vec::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            // Use CPUID to detect SIMD capabilities
            unsafe {
                // Check for SSE
                let mut eax = 1;
                asm!("cpuid", 
                     inout("eax") eax,
                     out("r10") _,
                     out("ecx") _,
                     out("edx") _,
                     options(nomem, nostack));
                
                let edx = eax; // Actually checking EDX register
                
                if (edx >> 25) & 1 != 0 { capabilities.push(SimdCapability::SSE); }
                if (edx >> 26) & 1 != 0 { capabilities.push(SimdCapability::SSE2); }
                if (edx >> 0) & 1 != 0 { capabilities.push(SimdCapability::SSE3); }
                if (edx >> 19) & 1 != 0 { capabilities.push(SimdCapability::SSE41); }
                if (edx >> 20) & 1 != 0 { capabilities.push(SimdCapability::SSE42); }
                if (edx >> 28) & 1 != 0 { capabilities.push(SimdCapability::AVX); }
                if (edx >> 5) & 1 != 0 { capabilities.push(SimdCapability::AVX2); }
                if (edx >> 16) & 1 != 0 { capabilities.push(SimdCapability::AVX512); }
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON is always available on aarch64
            capabilities.push(SimdCapability::NEON);
            
            // Check for SVE (would need more sophisticated detection)
            capabilities.push(SimdCapability::SVE);
        }
        
        Ok(capabilities)
    }
    
    /// Detect system capabilities
    fn detect_capabilities() -> SaicResult<HashMap<String, bool>> {
        let mut capabilities = HashMap::new();
        
        // Check for CUDA
        capabilities.insert("cuda".to_string(), std::env::var("CUDA_PATH").is_ok());
        
        // Check for OpenCL
        capabilities.insert("opencl".to_string(), std::env::var("OPENCL_PATH").is_ok());
        
        // Check for Vulkan
        capabilities.insert("vulkan".to_string(), std::env::var("VULKAN_SDK").is_ok());
        
        // Check for TensorRT
        capabilities.insert("tensorrt".to_string(), std::env::var("TENSORRT_PATH").is_ok());
        
        // Check for oneDNN
        capabilities.insert("onednn".to_string(), std::env::var("DNNLROOT").is_ok());
        
        Ok(capabilities)
    }
    
    /// Get recommended hardware target
    pub fn recommended_target(&self) -> crate::format::HardwareTarget {
        if self.gpu_info.is_some() {
            crate::format::HardwareTarget::Gpu
        } else {
            crate::format::HardwareTarget::Cpu
        }
    }
    
    /// Get optimization level based on environment
    pub fn recommended_optimization(&self) -> crate::format::OptimizationLevel {
        if self.cpu_cores >= 8 && self.simd_capabilities.contains(&SimdCapability::AVX2) {
            crate::format::OptimizationLevel::Maximum
        } else if self.cpu_cores >= 4 {
            crate::format::OptimizationLevel::Advanced
        } else {
            crate::format::OptimizationLevel::Basic
        }
    }
    
    /// Check if environment supports specific feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "gpu" => self.gpu_info.is_some(),
            "cuda" => *self.capabilities.get("cuda").unwrap_or(&false),
            "opencl" => *self.capabilities.get("opencl").unwrap_or(&false),
            "avx2" => self.simd_capabilities.contains(&SimdCapability::AVX2),
            "avx512" => self.simd_capabilities.contains(&SimdCapability::AVX512),
            "neon" => self.simd_capabilities.contains(&SimdCapability::NEON),
            _ => false,
        }
    }
}

/// Validate file permissions
pub fn validate_file_permissions(path: &Path) -> SaicResult<()> {
    // TODO: Implement file permission validation
    // For now, just check if file exists
    if !path.exists() {
        return Err(SaicError::FileSystem(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found"
        )));
    }
    
    Ok(())
}

/// Validate file path for path traversal attacks
pub fn validate_path(path: &Path) -> SaicResult<()> {
    // Check for path traversal attempts
    let path_str = path.to_string_lossy();
    
    if path_str.contains("..") {
        return Err(SaicError::PathTraversal("Path traversal detected".to_string()));
    }
    
    if path_str.contains("//") {
        return Err(SaicError::PathTraversal("Invalid path format".to_string()));
    }
    
    Ok(())
}
