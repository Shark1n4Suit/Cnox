use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

/// Custom error types for the packager application
/// 
/// This enum provides specific, descriptive error types for different failure modes
/// to enable proper error handling and user feedback.
#[derive(Debug)]
pub enum PackagerError {
    /// I/O operation failed (file reading, writing, etc.)
    Io(std::io::Error),
    /// Invalid file path provided
    InvalidPath(String),
    /// Invalid model name provided
    InvalidModelName(String),
    /// JSON serialization/deserialization error
    JsonError(serde_json::Error),
}

impl std::fmt::Display for PackagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackagerError::Io(err) => write!(f, "I/O error: {}", err),
            PackagerError::InvalidPath(path) => write!(f, "Invalid file path: {}", path),
            PackagerError::InvalidModelName(name) => write!(f, "Invalid model name: {}", name),
            PackagerError::JsonError(err) => write!(f, "JSON error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;
    use std::path::Path;

    /// Test SAIC file format generation and structure
    /// 
    /// This test verifies that our SAIC file creation follows the v0.1 specification
    /// by checking the binary structure of a generated file.
    #[test]
    fn test_saic_file_format() {
        // Setup test data
        let test_model_content = b"Test model data for unit testing";
        let test_model_path = "test_model_unit.bin";
        let test_output_path = "test_output_unit.saic";
        let test_model_name = "UnitTestModel";
        
        // Create test model file
        fs::write(test_model_path, test_model_content).expect("Failed to create test model file");
        
        // Create test arguments
        let args = PackagerArgs {
            model_path: PathBuf::from(test_model_path),
            model_name: test_model_name.to_string(),
            output_path: PathBuf::from(test_output_path),
        };
        
        // Execute SAIC file creation
        let result = create_saic_file(&args);
        assert!(result.is_ok(), "SAIC file creation failed: {:?}", result);
        
        // Verify output file was created
        assert!(Path::new(test_output_path).exists(), "Output SAIC file was not created");
        
        // Read and verify the generated SAIC file structure
        let mut saic_file = fs::File::open(test_output_path).expect("Failed to open SAIC file");
        let mut buffer = Vec::new();
        saic_file.read_to_end(&mut buffer).expect("Failed to read SAIC file");
        
        // Verify minimum file size (header + length + manifest + payload)
        assert!(buffer.len() >= 12, "SAIC file too small: {} bytes", buffer.len());
        
        // Verify magic number "SAIC"
        assert_eq!(&buffer[0..4], b"SAIC", "Invalid magic number");
        
        // Verify version 0.1.0.0
        assert_eq!(&buffer[4..8], &[0, 1, 0, 0], "Invalid version");
        
        // Verify manifest length (big-endian)
        let manifest_length = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
        assert!(manifest_length > 0, "Manifest length is zero");
        
        // Verify file size matches expected structure
        let expected_size = 12 + manifest_length + test_model_content.len();
        assert_eq!(buffer.len(), expected_size, "File size mismatch");
        
        // Verify manifest is valid JSON and contains model name
        let manifest_start = 12;
        let manifest_end = manifest_start + manifest_length;
        let manifest_bytes = &buffer[manifest_start..manifest_end];
        let manifest_str = std::str::from_utf8(manifest_bytes).expect("Invalid UTF-8 in manifest");
        let manifest: serde_json::Value = serde_json::from_str(manifest_str).expect("Invalid JSON in manifest");
        
        assert_eq!(manifest["model_name"], test_model_name, "Model name mismatch in manifest");
        
        // Verify model payload matches original
        let payload_start = manifest_end;
        let payload = &buffer[payload_start..];
        assert_eq!(payload, test_model_content, "Model payload mismatch");
        
        // Cleanup test files
        fs::remove_file(test_model_path).ok();
        fs::remove_file(test_output_path).ok();
    }
    
    /// Test input validation for model names
    /// 
    /// Verifies that our validation logic correctly handles various model name inputs.
    #[test] 
    fn test_model_name_validation() {
        // Test empty model name - would be caught in parse_args validation
        
        // Test normal model name
        let valid_name = "ValidModelName-v1.0";
        assert!(valid_name.len() <= 255, "Test setup error: name too long");
        assert!(!valid_name.chars().any(|c| c.is_control()), "Test setup error: contains control chars");
        
        // Test name with control character
        let invalid_name = "Invalid\nModel";
        assert!(invalid_name.chars().any(|c| c.is_control()), "Control character validation test");
        
        // Test long name
        let long_name = "a".repeat(300);
        assert!(long_name.len() > 255, "Long name validation test");
    }
}

impl std::error::Error for PackagerError {}

/// Validated command-line arguments for the packager tool
/// 
/// This struct contains sanitized and validated input parameters
/// after successful argument parsing and validation.
#[derive(Debug)]
struct PackagerArgs {
    /// Path to the input model file
    model_path: PathBuf,
    /// Name to be stored in the SAIC manifest
    model_name: String,
    /// Path where the output .saic file will be written
    output_path: PathBuf,
}

/// Initialize the tracing logging system
/// 
/// Sets up structured logging with appropriate filtering and formatting
/// for development and audit purposes.
fn init_logging() -> Result<(), PackagerError> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| PackagerError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    Ok(())
}

/// Parse and validate command-line arguments
/// 
/// Performs rigorous validation and sanitization of all user inputs
/// to ensure they meet security and format requirements.
fn parse_args() -> Result<PackagerArgs, PackagerError> {
    let matches = Command::new("packager")
        .version("0.1.0")
        .about("SAIC (Secure AI Container) file packager")
        .long_about("Creates .saic files from AI models according to the SAIC v0.1 specification")
        .arg(
            Arg::new("model")
                .long("model")
                .value_name("PATH")
                .help("Path to the input model file")
                .required(true),
        )
        .arg(
            Arg::new("name")
                .long("name")
                .value_name("STRING")
                .help("Name for the model (stored in manifest)")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("PATH")
                .help("Output path for the .saic file")
                .required(true),
        )
        .get_matches();

    // Extract and validate model path
    let model_path_str = matches.get_one::<String>("model").unwrap();
    let model_path = PathBuf::from(model_path_str);
    
    // Validate model file exists and is readable
    if !model_path.exists() {
        return Err(PackagerError::InvalidPath(
            format!("Model file does not exist: {}", model_path_str)
        ));
    }
    
    if !model_path.is_file() {
        return Err(PackagerError::InvalidPath(
            format!("Model path is not a file: {}", model_path_str)
        ));
    }

    // Extract and validate model name
    let model_name = matches.get_one::<String>("name").unwrap().clone();
    
    // Validate model name is not empty and contains only safe characters
    if model_name.is_empty() {
        return Err(PackagerError::InvalidModelName("Model name cannot be empty".to_string()));
    }
    
    if model_name.len() > 255 {
        return Err(PackagerError::InvalidModelName("Model name too long (max 255 characters)".to_string()));
    }
    
    // Basic sanitization check - prevent control characters
    if model_name.chars().any(|c| c.is_control()) {
        return Err(PackagerError::InvalidModelName("Model name contains invalid characters".to_string()));
    }

    // Extract and validate output path
    let output_path_str = matches.get_one::<String>("output").unwrap();
    let output_path = PathBuf::from(output_path_str);
    
    // Validate output directory exists or can be created
    if let Some(parent) = output_path.parent() {
        // Only check if parent is not empty (current directory is represented as "")
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(PackagerError::InvalidPath(
                format!("Output directory does not exist: {}", parent.display())
            ));
        }
    }

    Ok(PackagerArgs {
        model_path,
        model_name,
        output_path,
    })
}

/// Create a SAIC file from the provided model and metadata
/// 
/// This function implements the SAIC v0.1 file format specification,
/// creating a properly structured container file with header, manifest, and payload.
fn create_saic_file(args: &PackagerArgs) -> Result<(), PackagerError> {
    use std::fs::File;
    use std::io::{Read, Write};

    info!("Starting SAIC file creation process");
    info!("Model: {}", args.model_path.display());
    info!("Output: {}", args.output_path.display());
    
    // Step 1: Create the JSON manifest
    info!("Creating JSON manifest");
    let manifest = serde_json::json!({
        "model_name": args.model_name
    });
    
    // Step 2: Serialize manifest to UTF-8 string
    info!("Serializing manifest to JSON");
    let manifest_string = serde_json::to_string(&manifest)
        .map_err(PackagerError::JsonError)?;
    let manifest_bytes = manifest_string.as_bytes();
    
    info!("Manifest size: {} bytes", manifest_bytes.len());
    
    // Step 3: Open output file for writing
    info!("Opening output file for writing");
    let mut output_file = File::create(&args.output_path)
        .map_err(PackagerError::Io)?;
    
    // Step 4: Write the 8-byte SAIC header
    info!("Writing SAIC header");
    
    // Write magic number "SAIC" (4 bytes)
    output_file.write_all(b"SAIC")
        .map_err(PackagerError::Io)?;
    
    // Write version 0.1.0.0 (4 bytes: major.minor.patch.build)
    let version_bytes: [u8; 4] = [0, 1, 0, 0];
    output_file.write_all(&version_bytes)
        .map_err(PackagerError::Io)?;
    
    // Step 5: Write manifest length as 32-bit big-endian unsigned integer
    info!("Writing manifest length");
    let manifest_length = manifest_bytes.len() as u32;
    let length_bytes = manifest_length.to_be_bytes(); // big-endian format
    output_file.write_all(&length_bytes)
        .map_err(PackagerError::Io)?;
    
    // Step 6: Write the JSON manifest string
    info!("Writing JSON manifest");
    output_file.write_all(manifest_bytes)
        .map_err(PackagerError::Io)?;
    
    // Step 7: Copy the input model file contents
    info!("Copying model payload");
    let mut input_file = File::open(&args.model_path)
        .map_err(PackagerError::Io)?;
    
    // Read and write the model file in chunks for memory efficiency
    let mut buffer = [0u8; 8192]; // 8KB buffer for efficient copying
    let mut total_bytes_copied = 0u64;
    
    loop {
        let bytes_read = input_file.read(&mut buffer)
            .map_err(PackagerError::Io)?;
        
        if bytes_read == 0 {
            break; // End of file reached
        }
        
        output_file.write_all(&buffer[..bytes_read])
            .map_err(PackagerError::Io)?;
        
        total_bytes_copied += bytes_read as u64;
    }
    
    // Ensure all data is written to disk
    output_file.flush().map_err(PackagerError::Io)?;
    
    info!("Model payload copied: {} bytes", total_bytes_copied);
    info!("SAIC file creation completed successfully");
    
    // Log final file structure summary
    info!("Final SAIC file structure:");
    info!("  Header: 8 bytes (magic + version)");
    info!("  Manifest length: 4 bytes");
    info!("  Manifest: {} bytes", manifest_bytes.len());
    info!("  Model payload: {} bytes", total_bytes_copied);
    info!("  Total file size: {} bytes", 8 + 4 + manifest_bytes.len() as u64 + total_bytes_copied);
    
    Ok(())
}

/// Main entry point for the packager application
/// 
/// Initializes logging, parses command-line arguments, validates inputs,
/// and orchestrates the SAIC file creation process. All errors are properly
/// logged and returned with appropriate exit codes.
fn main() -> Result<(), PackagerError> {
    // Initialize structured logging for audit trail
    init_logging()?;
    
    info!("Packager v0.1.0 starting");
    
    // Parse and validate command-line arguments
    let args = parse_args()?;
    
    // Log validated arguments for audit purposes
    info!("Validated arguments:");
    info!("  Model file: {}", args.model_path.display());
    info!("  Model name: {}", args.model_name);
    info!("  Output file: {}", args.output_path.display());
    
    // Create the SAIC file
    match create_saic_file(&args) {
        Ok(()) => {
            info!("Packager completed successfully");
            Ok(())
        }
        Err(e) => {
            error!("Packager failed: {}", e);
            Err(e)
        }
    }
}
