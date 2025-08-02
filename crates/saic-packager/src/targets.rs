//! SAIC Platform Targeting
//! 
//! Handles platform targeting and cross-compilation support

use saic_core::{SaicError, SaicResult};

/// Target platform information
#[derive(Debug, Clone)]
pub struct TargetPlatform {
    pub name: String,
    pub architecture: String,
    pub optimization_flags: Vec<String>,
    pub memory_alignment: usize,
    pub cache_line_size: usize,
    pub supported_backends: Vec<String>,
}

/// Platform targeting utilities
pub struct PlatformTargets;

impl PlatformTargets {
    /// Get target platform for x86_64
    pub fn x86_64() -> TargetPlatform {
        TargetPlatform {
            name: "x86_64".to_string(),
            architecture: "x86_64".to_string(),
            optimization_flags: vec!["avx2".to_string(), "fma".to_string()],
            memory_alignment: 32,
            cache_line_size: 64,
            supported_backends: vec!["openvino".to_string(), "tensorrt".to_string(), "custom".to_string()],
        }
    }
    
    /// Get target platform for ARM64
    pub fn arm64() -> TargetPlatform {
        TargetPlatform {
            name: "arm64".to_string(),
            architecture: "aarch64".to_string(),
            optimization_flags: vec!["neon".to_string()],
            memory_alignment: 16,
            cache_line_size: 64,
            supported_backends: vec!["openvino".to_string(), "custom".to_string()],
        }
    }
    
    /// Get target platform for generic
    pub fn generic() -> TargetPlatform {
        TargetPlatform {
            name: "generic".to_string(),
            architecture: "generic".to_string(),
            optimization_flags: vec![],
            memory_alignment: 8,
            cache_line_size: 64,
            supported_backends: vec!["custom".to_string()],
        }
    }
    
    /// Get target platform by name
    pub fn get_target(platform_name: &str) -> SaicResult<TargetPlatform> {
        match platform_name {
            "x86_64" => Ok(Self::x86_64()),
            "arm64" => Ok(Self::arm64()),
            "generic" => Ok(Self::generic()),
            _ => Err(SaicError::InvalidFormat(
                format!("Unsupported target platform: {}", platform_name)
            )),
        }
    }
    
    /// Get all available target platforms
    pub fn available_targets() -> Vec<String> {
        vec!["x86_64".to_string(), "arm64".to_string(), "generic".to_string()]
    }
    
    /// Validate target platform for backend
    pub fn validate_target_for_backend(platform_name: &str, backend_name: &str) -> SaicResult<()> {
        let target = Self::get_target(platform_name)?;
        
        if !target.supported_backends.contains(&backend_name.to_string()) {
            return Err(SaicError::InvalidFormat(
                format!("Backend '{}' does not support target platform '{}'", backend_name, platform_name)
            ));
        }
        
        Ok(())
    }
    
    /// Get optimization flags for target platform
    pub fn get_optimization_flags(platform_name: &str) -> SaicResult<Vec<String>> {
        let target = Self::get_target(platform_name)?;
        Ok(target.optimization_flags)
    }
    
    /// Get memory alignment for target platform
    pub fn get_memory_alignment(platform_name: &str) -> SaicResult<usize> {
        let target = Self::get_target(platform_name)?;
        Ok(target.memory_alignment)
    }
    
    /// Get cache line size for target platform
    pub fn get_cache_line_size(platform_name: &str) -> SaicResult<usize> {
        let target = Self::get_target(platform_name)?;
        Ok(target.cache_line_size)
    }
}

/// Cross-compilation utilities
pub struct CrossCompilation;

impl CrossCompilation {
    /// Check if cross-compilation is supported for target
    pub fn is_supported(platform_name: &str) -> bool {
        match platform_name {
            "x86_64" => true,
            "arm64" => true,
            "generic" => true,
            _ => false,
        }
    }
    
    /// Get cross-compilation toolchain for target
    pub fn get_toolchain(platform_name: &str) -> SaicResult<String> {
        match platform_name {
            "x86_64" => Ok("native".to_string()),
            "arm64" => Ok("aarch64-linux-gnu".to_string()),
            "generic" => Ok("generic".to_string()),
            _ => Err(SaicError::InvalidFormat(
                format!("No toolchain available for platform: {}", platform_name)
            )),
        }
    }
    
    /// Get compilation flags for target platform
    pub fn get_compilation_flags(platform_name: &str) -> SaicResult<Vec<String>> {
        let target = PlatformTargets::get_target(platform_name)?;
        let mut flags = vec![];
        
        // Add architecture-specific flags
        match target.architecture.as_str() {
            "x86_64" => {
                flags.push("-march=native".to_string());
                flags.push("-mtune=native".to_string());
            },
            "aarch64" => {
                flags.push("-march=armv8-a".to_string());
                flags.push("-mtune=cortex-a72".to_string());
            },
            _ => {
                // Generic flags for unknown architectures
                flags.push("-O2".to_string());
            },
        }
        
        // Add optimization flags
        for flag in &target.optimization_flags {
            match flag.as_str() {
                "avx2" => flags.push("-mavx2".to_string()),
                "fma" => flags.push("-mfma".to_string()),
                "neon" => flags.push("-mfpu=neon".to_string()),
                _ => {},
            }
        }
        
        Ok(flags)
    }
}

/// Platform detection utilities
pub struct PlatformDetection;

impl PlatformDetection {
    /// Detect current platform
    pub fn detect_current_platform() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            "x86_64".to_string()
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            "arm64".to_string()
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "generic".to_string()
        }
    }
    
    /// Check if current platform matches target
    pub fn is_native_compilation(target_platform: &str) -> bool {
        let current = Self::detect_current_platform();
        current == target_platform
    }
    
    /// Get platform capabilities summary
    pub fn get_platform_summary() -> String {
        let current = Self::detect_current_platform();
        let is_native = Self::is_native_compilation(&current);
        
        format!(
            "Current platform: {} (native: {})",
            current,
            if is_native { "yes" } else { "no" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_64_target() {
        let target = PlatformTargets::x86_64();
        
        assert_eq!(target.name, "x86_64");
        assert_eq!(target.architecture, "x86_64");
        assert!(target.optimization_flags.contains(&"avx2".to_string()));
        assert!(target.supported_backends.contains(&"openvino".to_string()));
    }

    #[test]
    fn test_arm64_target() {
        let target = PlatformTargets::arm64();
        
        assert_eq!(target.name, "arm64");
        assert_eq!(target.architecture, "aarch64");
        assert!(target.optimization_flags.contains(&"neon".to_string()));
        assert!(target.supported_backends.contains(&"openvino".to_string()));
    }

    #[test]
    fn test_get_target() {
        let target = PlatformTargets::get_target("x86_64").unwrap();
        assert_eq!(target.name, "x86_64");
        
        let result = PlatformTargets::get_target("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_available_targets() {
        let targets = PlatformTargets::available_targets();
        
        assert!(targets.contains(&"x86_64".to_string()));
        assert!(targets.contains(&"arm64".to_string()));
        assert!(targets.contains(&"generic".to_string()));
    }

    #[test]
    fn test_validate_target_for_backend() {
        // Valid combination
        let result = PlatformTargets::validate_target_for_backend("x86_64", "openvino");
        assert!(result.is_ok());
        
        // Invalid combination
        let result = PlatformTargets::validate_target_for_backend("generic", "tensorrt");
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_compilation() {
        let supported = CrossCompilation::is_supported("x86_64");
        assert!(supported);
        
        let toolchain = CrossCompilation::get_toolchain("x86_64").unwrap();
        assert_eq!(toolchain, "native");
    }

    #[test]
    fn test_platform_detection() {
        let current = PlatformDetection::detect_current_platform();
        assert!(!current.is_empty());
        
        let summary = PlatformDetection::get_platform_summary();
        assert!(summary.contains("Current platform"));
    }
} 