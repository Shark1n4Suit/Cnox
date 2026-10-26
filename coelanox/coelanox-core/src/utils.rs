//! Utility functions for COELANOX
//!
//! This module provides utility functions for file operations, security validation,
//! and environment detection.

use crate::{CoelanoxError, CoelanoxResult};
use std::arch::asm;
use std::collections::HashMap;
use std::path::Path;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

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
    pub fn detect() -> CoelanoxResult<Self> {
        let os = Self::detect_os()?;
        let architecture = Self::detect_architecture()?;
        let cpu_vendor = Self::detect_cpu_vendor()?;
        // One sysinfo refresh for CPU brand + total RAM (Windows needs this for correct model/RAM).
        let (sysinfo_brand, ram_bytes) = Self::sysinfo_cpu_brand_and_ram_bytes();
        let cpu_model = sysinfo_brand
            .or_else(Self::cpu_model_linux_proc)
            .unwrap_or_else(|| "Unknown".to_string());
        let memory_gb = Self::memory_gb_from_ram_bytes(ram_bytes);
        let cpu_cores = Self::detect_cpu_cores()?;
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
    fn detect_os() -> CoelanoxResult<String> {
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
    fn detect_architecture() -> CoelanoxResult<String> {
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

    /// Detect CPU vendor (x86: stable `__cpuid(0)` — avoids broken inline asm on some targets).
    fn detect_cpu_vendor() -> CoelanoxResult<String> {
        #[cfg(target_arch = "x86_64")]
        {
            // `__cpuid` is `unsafe` on older Rust; on 1.95+ the intrinsic is safe and an `unsafe`
            // block triggers `unused_unsafe` under `-D warnings`. Allow keeps CI and MSRV builds aligned.
            #[allow(unused_unsafe)]
            let cp = unsafe { std::arch::x86_64::__cpuid(0) };
            let mut buf = [0u8; 12];
            buf[0..4].copy_from_slice(&cp.ebx.to_le_bytes());
            buf[4..8].copy_from_slice(&cp.edx.to_le_bytes());
            buf[8..12].copy_from_slice(&cp.ecx.to_le_bytes());
            let vendor = String::from_utf8_lossy(&buf).trim().to_string();
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

    /// Single refresh: CPU brand string and total RAM in bytes (sysinfo).
    fn sysinfo_cpu_brand_and_ram_bytes() -> (Option<String>, u64) {
        let mut sys = System::new();
        sys.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::new().with_ram()),
        );
        let brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty());
        (brand, sys.total_memory())
    }

    #[cfg(target_os = "linux")]
    fn cpu_model_linux_proc() -> Option<String> {
        let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in content.lines() {
            if line.starts_with("model name") {
                let model = line.split(':').nth(1)?.trim();
                if !model.is_empty() {
                    return Some(model.to_string());
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn cpu_model_linux_proc() -> Option<String> {
        None
    }

    fn memory_gb_from_ram_bytes(ram_bytes: u64) -> usize {
        if ram_bytes > 0 {
            return (ram_bytes / (1024 * 1024 * 1024)) as usize;
        }
        #[cfg(target_os = "linux")]
        {
            if let Some(gb) = Self::memory_gb_linux_proc() {
                return gb;
            }
        }
        8
    }

    #[cfg(target_os = "linux")]
    fn memory_gb_linux_proc() -> Option<usize> {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let kb = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
                return Some(kb / 1024 / 1024);
            }
        }
        None
    }

    /// Detect number of CPU cores
    fn detect_cpu_cores() -> CoelanoxResult<usize> {
        Ok(num_cpus::get())
    }

    /// Detect GPU information
    fn detect_gpu() -> CoelanoxResult<Option<GpuInfo>> {
        #[cfg(target_os = "linux")]
        {
            // Try to detect NVIDIA GPU
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=name,memory.total,driver_version")
                .arg("--format=csv,noheader,nounits")
                .output()
            {
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
            if std::process::Command::new("rocm-smi")
                .arg("--showproductname")
                .output()
                .is_ok()
            {
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
    fn detect_simd_capabilities() -> CoelanoxResult<Vec<SimdCapability>> {
        let mut capabilities = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            // Use CPUID to detect SIMD capabilities
            unsafe {
                // Check for SSE/AVX capabilities (CPUID function 1)
                let mut eax = 1u32;
                let mut ecx = 0u32;
                let mut edx = 0u32;

                // CPUID modifies rbx, but we don't need it - just preserve it
                asm!("push rbx; cpuid; pop rbx",
                     inout("eax") eax,
                     out("ecx") ecx,
                     out("edx") edx,
                     options(nomem, preserves_flags));
                let _ = eax;

                // Check EDX register flags (SSE, SSE2, etc.)
                if (edx >> 25) & 1 != 0 {
                    capabilities.push(SimdCapability::SSE);
                }
                if (edx >> 26) & 1 != 0 {
                    capabilities.push(SimdCapability::SSE2);
                }

                // Check ECX register flags (SSE3, SSE4, AVX, etc.)
                if ecx & 1 != 0 {
                    capabilities.push(SimdCapability::SSE3);
                }
                if (ecx >> 19) & 1 != 0 {
                    capabilities.push(SimdCapability::SSE41);
                }
                if (ecx >> 20) & 1 != 0 {
                    capabilities.push(SimdCapability::SSE42);
                }
                if (ecx >> 28) & 1 != 0 {
                    capabilities.push(SimdCapability::AVX);
                }

                // Check for AVX2 (CPUID function 7, sub-leaf 0)
                // Note: rbx is reserved by LLVM, so we use r10 and move the value
                let mut eax7 = 7u32;
                let ecx_in = 0u32; // Sub-leaf (input)
                let mut r10_result = 0u64; // Temporary register for ebx
                let mut _ecx7 = 0u32;
                let mut _edx7 = 0u32;

                asm!("push rbx; cpuid; mov {result}, rbx; pop rbx",
                     inout("eax") eax7,
                     inout("ecx") ecx_in => _ecx7,
                     out("edx") _edx7,
                     result = out(reg) r10_result,
                     options(nomem, preserves_flags));

                let _ = (eax7, _ecx7, _edx7);
                let ebx7 = (r10_result & 0xFFFFFFFF) as u32;
                if (ebx7 >> 5) & 1 != 0 {
                    capabilities.push(SimdCapability::AVX2);
                }
                if (ebx7 >> 16) & 1 != 0 {
                    capabilities.push(SimdCapability::AVX512);
                }
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
    fn detect_capabilities() -> CoelanoxResult<HashMap<String, bool>> {
        let mut capabilities = HashMap::new();

        // Check for CUDA
        capabilities.insert("cuda".to_string(), std::env::var("CUDA_PATH").is_ok());

        // Check for OpenCL
        capabilities.insert("opencl".to_string(), std::env::var("OPENCL_PATH").is_ok());

        // Check for Vulkan
        capabilities.insert("vulkan".to_string(), std::env::var("VULKAN_SDK").is_ok());

        // Check for TensorRT
        capabilities.insert(
            "tensorrt".to_string(),
            std::env::var("TENSORRT_PATH").is_ok(),
        );

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

/// Validate file (existence check; rwx permission validation not implemented).
pub fn validate_file_permissions(path: &Path) -> CoelanoxResult<()> {
    if !path.exists() {
        return Err(CoelanoxError::FileSystem(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found",
        )));
    }

    Ok(())
}

/// Validate file path for path traversal attacks
pub fn validate_path(path: &Path) -> CoelanoxResult<()> {
    // Check for path traversal attempts
    let path_str = path.to_string_lossy();

    if path_str.contains("..") {
        return Err(CoelanoxError::PathTraversal(
            "Path traversal detected".to_string(),
        ));
    }

    if path_str.contains("//") {
        return Err(CoelanoxError::PathTraversal(
            "Invalid path format".to_string(),
        ));
    }

    Ok(())
}
