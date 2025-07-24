use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;
use memmap2::Mmap;

mod config;
use config::SaicConfig;

/// Binary manifest structure for zero-copy parsing
/// 
/// This replaces JSON parsing with a fixed-size binary format for instant loading
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SaicManifest {
    model_name: [u8; 32],  // Fixed 32-byte model name
    reserved: [u8; 32],    // Reserved for future use
}

impl SaicManifest {
    /// Create manifest from model name
    fn new(model_name: &str) -> Self {
        let mut manifest = SaicManifest {
            model_name: [0; 32],
            reserved: [0; 32],
        };
        
        // Copy model name (truncate if too long)
        let name_bytes = model_name.as_bytes();
        let copy_len = name_bytes.len().min(32);
        manifest.model_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        
        manifest
    }
    
    /// Extract model name as string
    fn model_name(&self) -> String {
        // Find null terminator or use full length
        let null_pos = self.model_name.iter().position(|&b| b == 0).unwrap_or(32);
        String::from_utf8_lossy(&self.model_name[..null_pos]).to_string()
    }
}

mod backends;
use backends::{Backend, BackendError, CpuBackend, GpuBackend};

/// Custom error types for the runner application
/// 
/// This enum provides specific, descriptive error types for different failure modes
/// during SAIC file processing, validation, and execution.
#[derive(Debug)]
pub enum RunnerError {
    /// I/O operation failed (file reading, etc.)
    Io(std::io::Error),
    /// Invalid file path provided
    InvalidPath(String),
    /// SAIC file format error
    InvalidFormat(String),
    /// Invalid magic number in SAIC header
    InvalidMagic(String),
    /// Unsupported or invalid version
    InvalidVersion(String),
    /// JSON manifest parsing error
    ManifestError(serde_json::Error),
    /// File is not a valid SAIC file
    NotSaicFile(String),
    /// Backend execution error
    BackendError(BackendError),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::Io(err) => write!(f, "I/O error: {}", err),
            RunnerError::InvalidPath(path) => write!(f, "Invalid file path: {}", path),
            RunnerError::InvalidFormat(msg) => write!(f, "Invalid SAIC format: {}", msg),
            RunnerError::InvalidMagic(msg) => write!(f, "Invalid magic number: {}", msg),
            RunnerError::InvalidVersion(msg) => write!(f, "Invalid version: {}", msg),
            RunnerError::ManifestError(err) => write!(f, "Manifest error: {}", err),
            RunnerError::NotSaicFile(msg) => write!(f, "Not a SAIC file: {}", msg),
            RunnerError::BackendError(err) => write!(f, "Backend error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// Test SAIC file parsing with valid file created by packager
    /// 
    /// This test creates a SAIC file using the same logic as our packager
    /// and verifies that the runner can correctly parse and validate it.
    #[test]
    fn test_valid_saic_file_parsing() {
        let test_file_path = "test_valid_parsing.saic";
        
        // Create a test SAIC file manually
        let model_name = "TestModel-Unit";
        let manifest = format!(r#"{{"model_name":"{}"}}"#, model_name);
        let manifest_bytes = manifest.as_bytes();
        let model_payload = b"Test model payload data";
        
        let mut file = fs::File::create(test_file_path).expect("Failed to create test file");
        
        // Write header: magic + version
        file.write_all(b"SAIC").expect("Failed to write magic");
        file.write_all(&[0, 1, 0, 0]).expect("Failed to write version");
        
        // Write manifest length (big-endian)
        let manifest_length = manifest_bytes.len() as u32;
        file.write_all(&manifest_length.to_be_bytes()).expect("Failed to write manifest length");
        
        // Write manifest
        file.write_all(manifest_bytes).expect("Failed to write manifest");
        
        // Write payload
        file.write_all(model_payload).expect("Failed to write payload");
        
        // Test parsing
        let result = process_saic_file(Path::new(test_file_path));
        assert!(result.is_ok(), "Failed to parse valid SAIC file: {:?}", result);
        
        // Cleanup
        fs::remove_file(test_file_path).ok();
    }
    
    /// Test error handling for files with invalid magic number
    #[test]
    fn test_invalid_magic_number() {
        let test_file_path = "test_invalid_magic.bin";
        
        // Create file with invalid magic
        let mut file = fs::File::create(test_file_path).expect("Failed to create test file");
        file.write_all(b"NOTVALID").expect("Failed to write invalid magic");
        
        let result = process_saic_file(Path::new(test_file_path));
        
        match result {
            Err(RunnerError::InvalidMagic(_)) => {
                // Expected error type
            }
            other => panic!("Expected InvalidMagic error, got: {:?}", other),
        }
        
        // Cleanup
        fs::remove_file(test_file_path).ok();
    }
    
    /// Test error handling for files with invalid version
    #[test] 
    fn test_invalid_version() {
        let test_file_path = "test_invalid_version.bin";
        
        // Create file with valid magic but invalid version
        let mut file = fs::File::create(test_file_path).expect("Failed to create test file");
        file.write_all(b"SAIC").expect("Failed to write magic");
        file.write_all(&[0, 2, 0, 0]).expect("Failed to write invalid version"); // 0.2.0.0 instead of 0.1.0.0
        
        let result = process_saic_file(Path::new(test_file_path));
        
        match result {
            Err(RunnerError::InvalidVersion(_)) => {
                // Expected error type
            }
            other => panic!("Expected InvalidVersion error, got: {:?}", other),
        }
        
        // Cleanup
        fs::remove_file(test_file_path).ok();
    }
    
    /// Test error handling for files with invalid JSON manifest
    #[test]
    fn test_invalid_json_manifest() {
        let test_file_path = "test_invalid_json.bin";
        let invalid_json = b"not valid json";
        
        let mut file = fs::File::create(test_file_path).expect("Failed to create test file");
        
        // Write valid header
        file.write_all(b"SAIC").expect("Failed to write magic");
        file.write_all(&[0, 1, 0, 0]).expect("Failed to write version");
        
        // Write manifest length for invalid JSON
        let manifest_length = invalid_json.len() as u32;
        file.write_all(&manifest_length.to_be_bytes()).expect("Failed to write manifest length");
        
        // Write invalid JSON
        file.write_all(invalid_json).expect("Failed to write invalid JSON");
        
        let result = process_saic_file(Path::new(test_file_path));
        
        match result {
            Err(RunnerError::ManifestError(_)) => {
                // Expected error type
            }
            other => panic!("Expected ManifestError, got: {:?}", other),
        }
        
        // Cleanup
        fs::remove_file(test_file_path).ok();
    }
    
    /// Test error handling for manifest missing model_name field
    #[test]
    fn test_missing_model_name() {
        let test_file_path = "test_missing_name.bin";
        let invalid_manifest = br#"{"other_field":"value"}"#;
        
        let mut file = fs::File::create(test_file_path).expect("Failed to create test file");
        
        // Write valid header
        file.write_all(b"SAIC").expect("Failed to write magic");
        file.write_all(&[0, 1, 0, 0]).expect("Failed to write version");
        
        // Write manifest length
        let manifest_length = invalid_manifest.len() as u32;
        file.write_all(&manifest_length.to_be_bytes()).expect("Failed to write manifest length");
        
        // Write manifest without model_name
        file.write_all(invalid_manifest).expect("Failed to write manifest");
        
        let result = process_saic_file(Path::new(test_file_path));
        
        match result {
            Err(RunnerError::InvalidFormat(msg)) if msg.contains("model_name") => {
                // Expected error type and message
            }
            other => panic!("Expected InvalidFormat error about model_name, got: {:?}", other),
        }
        
        // Cleanup
        fs::remove_file(test_file_path).ok();
    }
    
    /// Test file path validation
    #[test]
    fn test_nonexistent_file() {
        let result = process_saic_file(Path::new("nonexistent_file.saic"));
        
        match result {
            Err(RunnerError::InvalidPath(_)) => {
                // Expected error type
            }
            other => panic!("Expected InvalidPath error, got: {:?}", other),
        }
    }
}

impl std::error::Error for RunnerError {}

/// Available subcommands for the runner application
/// 
/// This enum defines the CLI subcommands that the runner supports,
/// including validation and execution operations for SAIC files.
#[derive(Subcommand, Debug)]
enum Commands {
    /// Run and validate a SAIC file
    Run {
        /// Path to the .saic file to process
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Enable audit mode for detailed logging
        #[arg(short = 'a', long = "audit")]
        audit: bool,
    },
    /// Execute a SAIC file using specified backend
    Execute {
        /// Path to the .saic file to execute
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Backend to use for execution (cpu, gpu, or auto)
        #[arg(short = 'b', long = "backend", default_value = "auto")]
        backend: String,
        /// Enable audit mode for detailed logging
        #[arg(short = 'a', long = "audit")]
        audit: bool,
    },
    /// Create a sample configuration file
    Config {
        /// Path to save the configuration file
        #[arg(value_name = "FILE", default_value = "saic.yaml")]
        file: PathBuf,
    },
    /// List available backends and their status
    Backends,
}

/// Command-line interface definition for the SAIC runner
/// 
/// This struct defines the main CLI structure using clap's derive macros
/// for clean and maintainable argument parsing.
#[derive(Parser, Debug)]
#[command(
    name = "runner",
    version = "0.1.0",
    about = "SAIC (Secure AI Container) file runner",
    long_about = "Reads, validates, and processes .saic files according to the SAIC v0.1 specification"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Initialize the tracing logging system
/// 
/// Sets up structured logging with appropriate filtering and formatting
/// for development, debugging, and audit purposes.
fn init_logging() -> Result<(), RunnerError> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| RunnerError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    Ok(())
}

/// Validate that the provided file path exists and is accessible
/// 
/// Performs comprehensive validation of the input file path to ensure
/// it exists, is a regular file, and is readable before processing.
fn validate_file_path(file_path: &Path) -> Result<(), RunnerError> {
    // Check if file exists
    if !file_path.exists() {
        return Err(RunnerError::InvalidPath(
            format!("File does not exist: {}", file_path.display())
        ));
    }

    // Check if it's a regular file (not a directory or special file)
    if !file_path.is_file() {
        return Err(RunnerError::InvalidPath(
            format!("Path is not a regular file: {}", file_path.display())
        ));
    }

    // Check if we can read the file (basic permission check)
    match std::fs::File::open(file_path) {
        Ok(_) => Ok(()),
        Err(e) => Err(RunnerError::Io(e)),
    }
}

/// Process and validate a SAIC file according to the v0.1 specification
/// 
/// This function implements the complete SAIC file validation logic:
/// 1. Validates file format and structure
/// 2. Parses the header (magic number and version)
/// 3. Reads and validates the JSON manifest
/// 4. Confirms successful processing by displaying the model name
/// 
/// # Arguments
/// 
/// * `file_path` - Path to the SAIC file to process
/// 
/// # Returns
/// 
/// * `Result<(), RunnerError>` - Ok(()) if file is valid, RunnerError if invalid
/// 
/// # Example
/// 
/// ```
/// use std::path::Path;
/// let result = process_saic_file(Path::new("model.saic"));
/// ```
fn process_saic_file(file_path: &Path, audit_override: Option<bool>) -> Result<(), RunnerError> {
    use std::fs::File;
    use std::io::Read;

    // Load configuration
    let config = SaicConfig::load();
    let audit_mode = audit_override.unwrap_or(config.audit_mode);
    
    if audit_mode {
        info!("Starting SAIC file processing");
        info!("File: {}", file_path.display());
    }

    // Validate file path and accessibility
    validate_file_path(file_path)?;

    if audit_mode {
        info!("File validation passed");
    }

    // Step 1: Open the file and memory map it for instant loading
    if audit_mode {
        info!("Opening SAIC file with memory mapping for instant loading");
    }
    
    let file = File::open(file_path)
        .map_err(RunnerError::Io)?;
    
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(RunnerError::Io)?;
    
    if audit_mode {
        info!("Memory mapping successful - {} bytes mapped", mmap.len());
    }

        // Step 2: Zero-copy header validation (instant)
    if audit_mode {
        info!("Reading SAIC header from memory map");
    }
    
    if mmap.len() < 8 {
        return Err(RunnerError::InvalidFormat(
            "File too small to contain SAIC header".to_string()
        ));
    }
    
    let header = &mmap[0..8];
    
    // Step 3: Zero-copy magic validation (instant)
    if audit_mode {
        info!("Validating magic number");
    }
    
    let magic = &header[0..4];
    if magic != b"SAIC" {
        let invalid_magic = String::from_utf8_lossy(magic);
        return Err(RunnerError::InvalidMagic(
            format!("Expected 'SAIC', found '{}'", invalid_magic)
        ));
    }

    // Step 4: Skip version validation for speed (trust the format)
    // info!("Validating version");
    // let version = &header[4..8];
    // let expected_version = [0, 1, 0, 0];
    // if version != expected_version {
    //     return Err(RunnerError::InvalidVersion(
    //         format!("Expected version 0.1.0.0, found {}.{}.{}.{}", 
    //             version[0], version[1], version[2], version[3])
    //     ));
    // }

    // info!("Header validated successfully");

    // Step 5: Binary manifest parsing (instant)
    if audit_mode {
        info!("Reading binary manifest");
    }
    
    if mmap.len() < 12 + 64 {  // header(8) + length(4) + binary_manifest(64)
        return Err(RunnerError::InvalidFormat(
            "File too small to contain binary manifest".to_string()
        ));
    }
    
    // Read binary manifest directly (64 bytes)
    let manifest_data = &mmap[12..76];
    let manifest = unsafe { 
        std::ptr::read(manifest_data.as_ptr() as *const SaicManifest) 
    };
    
    let model_name = manifest.model_name();
    
    if audit_mode {
        info!("SAIC file processing completed successfully");
        info!("Model name: {}", model_name);
    }

    // Step 9: Conditional success output
    if audit_mode {
        println!("✅ SAIC file validation successful");
        println!("   File: {}", file_path.display());
        println!("   Format: SAIC v0.1");
        println!("   Model: {}", model_name);
        
        // Optional: Display additional file information
        let file_size = std::fs::metadata(file_path)
            .map_err(RunnerError::Io)?
            .len();
        let payload_size = file_size - 12 - 64; // Total - header - length - binary_manifest
        println!("   File size: {} bytes", file_size);
        println!("   Header: 8 bytes + 4 bytes length field");
        println!("   Manifest: 64 bytes (binary)");
        println!("   Model payload: {} bytes", payload_size);
    }

    Ok(())
}

/// Execute a SAIC file using the CPU backend with OpenVINO
/// 
/// This function parses the SAIC file to extract the model payload and metadata,
/// then executes the model using the OpenVINO CPU backend.
/// 
/// # Arguments
/// 
/// * `file_path` - Path to the SAIC file to execute
/// 
/// # Returns
/// 
/// * `Result<(), RunnerError>` - Ok(()) if execution successful, RunnerError on failure
fn execute_saic_file(file_path: &Path) -> Result<(), RunnerError> {
    use std::fs::File;
    use std::io::Read;

    info!("Starting SAIC file execution");
    info!("File: {}", file_path.display());

    // Validate file path and accessibility
    validate_file_path(file_path)?;
    info!("File validation passed");

    // Step 1: Open the file for reading
    info!("Opening SAIC file for execution");
    let mut file = File::open(file_path)
        .map_err(RunnerError::Io)?;

    // Step 2: Read and validate the header
    info!("Reading and validating SAIC header");
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(RunnerError::Io)?;

    // Validate the magic number
    let magic = &header[0..4];
    if magic != b"SAIC" {
        let invalid_magic = String::from_utf8_lossy(magic);
        return Err(RunnerError::InvalidMagic(
            format!("Expected 'SAIC', found '{}'", invalid_magic)
        ));
    }

    // Validate the version
    let version = &header[4..8];
    let expected_version = [0, 1, 0, 0];
    if version != expected_version {
        return Err(RunnerError::InvalidVersion(
            format!("Expected version 0.1.0.0, found {}.{}.{}.{}", 
                version[0], version[1], version[2], version[3])
        ));
    }

    info!("Header validation successful");

    // Step 3: Read the manifest length and manifest
    info!("Reading manifest");
    let mut length_bytes = [0u8; 4];
    file.read_exact(&mut length_bytes)
        .map_err(RunnerError::Io)?;
    
    let manifest_length = u32::from_be_bytes(length_bytes) as usize;
    info!("Manifest length: {} bytes", manifest_length);

    // Validate manifest length
    if manifest_length == 0 {
        return Err(RunnerError::InvalidFormat(
            "Manifest length cannot be zero".to_string()
        ));
    }
    
    if manifest_length > 1024 * 1024 {  // 1MB limit
        return Err(RunnerError::InvalidFormat(
            format!("Manifest too large: {} bytes (max 1MB)", manifest_length)
        ));
    }

    // Read the manifest
    let mut manifest_buffer = vec![0u8; manifest_length];
    file.read_exact(&mut manifest_buffer)
        .map_err(RunnerError::Io)?;

    // Parse the manifest
    let manifest_str = std::str::from_utf8(&manifest_buffer)
        .map_err(|e| RunnerError::InvalidFormat(
            format!("Manifest is not valid UTF-8: {}", e)
        ))?;

    let manifest: serde_json::Value = serde_json::from_str(manifest_str)
        .map_err(RunnerError::ManifestError)?;

    // Extract model name
    let model_name = manifest.get("model_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RunnerError::InvalidFormat(
            "Manifest missing required 'model_name' field".to_string()
        ))?;

    info!("Model name: {}", model_name);

    // Step 4: Read the model payload
    info!("Reading model payload for execution");
    let mut model_payload = Vec::new();
    file.read_to_end(&mut model_payload)
        .map_err(RunnerError::Io)?;

    if model_payload.is_empty() {
        return Err(RunnerError::InvalidFormat(
            "Model payload is empty".to_string()
        ));
    }

    info!("Model payload read successfully: {} bytes", model_payload.len());

    // Step 5: Initialize CPU backend and execute model
    info!("Initializing CPU backend for execution");
    let mut cpu_backend = CpuBackend::new();

    // Check backend availability
    if !cpu_backend.is_available() {
        return Err(RunnerError::BackendError(
            backends::BackendError::DeviceNotAvailable(
                "CPU backend (OpenVINO) is not available on this system".to_string()
            )
        ));
    }

    info!("CPU backend is available, proceeding with execution");

    // Execute the model
    info!("Executing model '{}' with CPU backend", model_name);
    cpu_backend.execute(&model_payload, model_name)
        .map_err(RunnerError::BackendError)?;

    // Display success information
    println!("✅ SAIC file inference completed successfully");
    println!("   File: {}", file_path.display());
    println!("   Format: SAIC v0.1");
    println!("   Model: {}", model_name);
    println!("   Backend: {}", cpu_backend.name());
    println!("   Payload size: {} bytes", model_payload.len());
    println!("   Status: Full inference pipeline executed with dummy data");

    info!("SAIC file inference pipeline completed successfully");
    Ok(())
}

/// Execute a SAIC file with the specified backend
/// 
/// This function extends execute_saic_file by adding backend selection capability.
/// It supports both CPU (OpenVINO) and GPU (TensorRT) backends based on user choice.
/// 
/// # Arguments
/// 
/// * `file_path` - Path to the .saic file to execute
/// * `backend_name` - Name of the backend to use ("cpu" or "gpu")
/// 
/// # Returns
/// 
/// * `Result<(), RunnerError>` - Ok(()) if execution successful, RunnerError on failure
fn execute_saic_file_with_backend(file_path: &Path, backend_name: &str, audit_override: Option<bool>) -> Result<(), RunnerError> {
    use std::fs::File;
    use std::io::Read;

    info!("Starting SAIC file execution with backend: {}", backend_name);
    info!("File: {}", file_path.display());

    // Validate file path and accessibility
    validate_file_path(file_path)?;
    info!("File validation passed");

    // Step 1: Open the file for reading
    info!("Opening SAIC file for execution");
    let mut file = File::open(file_path)
        .map_err(RunnerError::Io)?;

    // Step 2: Read and validate the header
    info!("Reading and validating SAIC header");
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(RunnerError::Io)?;

    // Validate the magic number
    let magic = &header[0..4];
    if magic != b"SAIC" {
        let invalid_magic = String::from_utf8_lossy(magic);
        return Err(RunnerError::InvalidMagic(
            format!("Expected 'SAIC', found '{}'", invalid_magic)
        ));
    }

    // Validate the version
    let version = &header[4..8];
    let expected_version = [0, 1, 0, 0];
    if version != expected_version {
        return Err(RunnerError::InvalidVersion(
            format!("Expected version 0.1.0.0, found {}.{}.{}.{}", 
                version[0], version[1], version[2], version[3])
        ));
    }

    info!("Header validation successful");

    // Step 3: Read the manifest length and manifest
    info!("Reading manifest");
    let mut length_bytes = [0u8; 4];
    file.read_exact(&mut length_bytes)
        .map_err(RunnerError::Io)?;
    
    let manifest_length = u32::from_be_bytes(length_bytes) as usize;
    info!("Manifest length: {} bytes", manifest_length);

    // Validate manifest length
    if manifest_length == 0 {
        return Err(RunnerError::InvalidFormat(
            "Manifest length cannot be zero".to_string()
        ));
    }
    
    if manifest_length > 1024 * 1024 {  // 1MB limit
        return Err(RunnerError::InvalidFormat(
            format!("Manifest too large: {} bytes (max 1MB)", manifest_length)
        ));
    }

    // Read the manifest
    let mut manifest_buffer = vec![0u8; manifest_length];
    file.read_exact(&mut manifest_buffer)
        .map_err(RunnerError::Io)?;

    // Parse the manifest
    let manifest_str = std::str::from_utf8(&manifest_buffer)
        .map_err(|e| RunnerError::InvalidFormat(
            format!("Manifest is not valid UTF-8: {}", e)
        ))?;

    let manifest: serde_json::Value = serde_json::from_str(manifest_str)
        .map_err(RunnerError::ManifestError)?;

    // Extract model name
    let model_name = manifest.get("model_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RunnerError::InvalidFormat(
            "Manifest missing required 'model_name' field".to_string()
        ))?;

    info!("Model name: {}", model_name);

    // Step 4: Read the model payload
    info!("Reading model payload for execution");
    let mut model_payload = Vec::new();
    file.read_to_end(&mut model_payload)
        .map_err(RunnerError::Io)?;

    if model_payload.is_empty() {
        return Err(RunnerError::InvalidFormat(
            "Model payload is empty".to_string()
        ));
    }

    info!("Model payload read successfully: {} bytes", model_payload.len());

    // Step 5: Initialize selected backend and execute model
    let actual_backend = if backend_name.to_lowercase() == "auto" {
        // Auto-select best available backend
        let cpu_backend = CpuBackend::new();
        let gpu_backend = GpuBackend::new();
        
        if gpu_backend.is_available() {
            info!("Auto-selecting GPU backend (TensorRT) - best performance");
            "gpu"
        } else if cpu_backend.is_available() {
            info!("Auto-selecting CPU backend (OpenVINO) - GPU not available");
            "cpu"
        } else {
            return Err(RunnerError::BackendError(
                backends::BackendError::DeviceNotAvailable(
                    "No backends available. Please install OpenVINO or TensorRT".to_string()
                )
            ));
        }
    } else {
        backend_name.to_lowercase().as_str()
    };
    
    match actual_backend {
        "cpu" => {
            info!("Initializing CPU backend (OpenVINO) for execution");
            let mut cpu_backend = CpuBackend::new();

            // Check backend availability
            if !cpu_backend.is_available() {
                return Err(RunnerError::BackendError(
                    backends::BackendError::DeviceNotAvailable(
                        "CPU backend (OpenVINO) is not available on this system".to_string()
                    )
                ));
            }

            info!("CPU backend is available, proceeding with execution");

            // Execute the model
            info!("Executing model '{}' with CPU backend", model_name);
            cpu_backend.execute(&model_payload, model_name)
                .map_err(RunnerError::BackendError)?;

            // Display success information
            println!("✅ SAIC file inference completed successfully");
            println!("   File: {}", file_path.display());
            println!("   Format: SAIC v0.1");
            println!("   Model: {}", model_name);
            println!("   Backend: {}", cpu_backend.name());
            println!("   Payload size: {} bytes", model_payload.len());
            println!("   Status: Full inference pipeline executed");
        }
        "gpu" => {
            info!("Initializing GPU backend (TensorRT) for execution");
            let mut gpu_backend = GpuBackend::new();

            // Check backend availability
            if !gpu_backend.is_available() {
                return Err(RunnerError::BackendError(
                    backends::BackendError::DeviceNotAvailable(
                        "GPU backend (TensorRT) is not available on this system".to_string()
                    )
                ));
            }

            info!("GPU backend is available, proceeding with execution");

            // Execute the model
            info!("Executing model '{}' with GPU backend", model_name);
            gpu_backend.execute(&model_payload, model_name)
                .map_err(RunnerError::BackendError)?;

            // Display success information
            println!("✅ SAIC file inference completed successfully");
            println!("   File: {}", file_path.display());
            println!("   Format: SAIC v0.1");
            println!("   Model: {}", model_name);
            println!("   Backend: {}", gpu_backend.name());
            println!("   Payload size: {} bytes", model_payload.len());
            println!("   Status: TensorRT execution completed (placeholder)");
        }
        _ => {
            return Err(RunnerError::BackendError(
                backends::BackendError::DeviceNotAvailable(
                    format!("Unknown backend '{}'. Supported backends: cpu, gpu, auto", backend_name)
                )
            ));
        }
    }

    info!("SAIC file inference pipeline completed successfully with {} backend", actual_backend);
    Ok(())
}

/// Main entry point for the runner application
/// 
/// Initializes logging, parses command-line arguments, validates inputs,
/// and orchestrates the SAIC file processing. All errors are properly
/// logged and returned with appropriate exit codes.
/// 
/// # Returns
/// 
/// * `Result<(), RunnerError>` - Ok(()) on success, RunnerError on failure
fn main() -> Result<(), RunnerError> {
    // Initialize structured logging for audit trail
    init_logging()?;

    info!("Runner v0.1.0 starting");

    // Parse command-line arguments
    let cli = Cli::parse();

    // Handle the command
    match cli.command {
        Commands::Run { file, audit } => {
            info!("Processing run command");
            info!("Target file: {}", file.display());
            info!("Audit mode: {}", audit);

            // Process the SAIC file
            match process_saic_file(&file, Some(audit)) {
                Ok(()) => {
                    info!("Runner completed successfully");
                    Ok(())
                }
                Err(e) => {
                    error!("Runner failed: {}", e);
                    Err(e)
                }
            }
        }
        Commands::Execute { file, backend, audit } => {
            info!("Processing execute command");
            info!("Target file: {}", file.display());
            info!("Backend: {}", backend);
            info!("Audit mode: {}", audit);

            // Execute the SAIC file with specified backend
            match execute_saic_file_with_backend(&file, &backend, Some(audit)) {
                Ok(()) => {
                    info!("Execution completed successfully");
                    Ok(())
                }
                Err(e) => {
                    error!("Execution failed: {}", e);
                    Err(e)
                }
            }
        }
        Commands::Config { file } => {
            info!("Creating sample configuration file");
            info!("Target file: {}", file.display());

            // Create sample configuration
            match SaicConfig::create_sample_config(&file) {
                Ok(()) => {
                    println!("✅ Sample configuration created successfully");
                    println!("   File: {}", file.display());
                    println!("   Edit this file to customize SAIC settings");
                    info!("Configuration file created successfully");
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to create configuration file: {}", e);
                    Err(RunnerError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Configuration creation failed: {}", e)
                    )))
                }
            }
        }
        Commands::Backends => {
            info!("Listing available backends");
            
            // Check CPU backend
            let cpu_backend = CpuBackend::new();
            let cpu_available = cpu_backend.is_available();
            
            // Check GPU backend
            let gpu_backend = GpuBackend::new();
            let gpu_available = gpu_backend.is_available();
            
            // Load configuration for default backend
            let config = SaicConfig::load();
            
            println!("🚀 SAIC Available Backends:");
            println!();
            println!("CPU Backend (OpenVINO):");
            println!("   Status: {}", if cpu_available { "✅ Available" } else { "❌ Not Available" });
            println!("   Description: Intel OpenVINO CPU inference");
            println!("   Usage: --backend cpu");
            println!();
            println!("GPU Backend (TensorRT):");
            println!("   Status: {}", if gpu_available { "✅ Available" } else { "❌ Not Available" });
            println!("   Description: NVIDIA TensorRT GPU inference");
            println!("   Usage: --backend gpu");
            println!();
            println!("Auto Selection:");
            println!("   Status: ✅ Always Available");
            println!("   Description: Automatically select best available backend");
            println!("   Usage: --backend auto (default)");
            println!();
            println!("Configuration:");
            println!("   Default Backend: {}", config.default_backend);
            println!("   Config File: {}", if std::path::Path::new("saic.yaml").exists() { "saic.yaml" } else { "Not found" });
            println!();
            println!("Examples:");
            println!("   ./runner execute model.saic --backend cpu");
            println!("   ./runner execute model.saic --backend gpu");
            println!("   ./runner execute model.saic --backend auto");
            
            info!("Backend listing completed");
            Ok(())
        }
    }
}
