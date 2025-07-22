# SAIC Core Library Reference

This document provides the API reference for the `saic_core` Rust crate, which contains the core logic extracted from the SAIC runner for use as a library in other applications.

## Overview

The `saic_core` crate provides the foundational components for working with SAIC containers, including parsing, validation, and backend management. This library enables developers to integrate SAIC functionality directly into their Rust applications.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
saic_core = "0.5.0"
```

## Quick Start

```rust
use saic_core::{Parser, Validator, BackendManager};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a SAIC container
    let parser = Parser::new();
    let container = parser.parse_file("model.saic")?;
    
    // Validate the container
    let validator = Validator::new();
    validator.validate(&container)?;
    
    // Initialize backend manager
    let mut backend_manager = BackendManager::new();
    backend_manager.initialize_backend("cpu")?;
    
    // Execute inference
    let result = backend_manager.execute(&container, &input_data)?;
    
    Ok(())
}
```

## Core Modules

### Parser Module

The parser module handles reading and parsing SAIC container files.

```rust
use saic_core::parser::{Parser, Container, ParseError};

pub struct Parser {
    // Parser configuration
}

impl Parser {
    /// Create a new parser instance
    pub fn new() -> Self;
    
    /// Parse a SAIC file from a path
    pub fn parse_file(&self, path: &str) -> Result<Container, ParseError>;
    
    /// Parse SAIC data from memory
    pub fn parse_bytes(&self, data: &[u8]) -> Result<Container, ParseError>;
    
    /// Parse SAIC data from a reader
    pub fn parse_reader<R: Read>(&self, reader: R) -> Result<Container, ParseError>;
}

pub struct Container {
    pub metadata: Metadata,
    pub model_data: Vec<u8>,
    pub signature: Signature,
}

pub struct Metadata {
    pub version: String,
    pub model_type: String,
    pub backend_requirements: Vec<String>,
    pub creation_date: DateTime<Utc>,
    pub author: String,
}

pub struct Signature {
    pub algorithm: String,
    pub value: Vec<u8>,
    pub public_key: Option<Vec<u8>>,
}
```

### Validator Module

The validator module provides cryptographic validation and integrity checking.

```rust
use saic_core::validator::{Validator, ValidationError, ValidationResult};

pub struct Validator {
    // Validator configuration
}

impl Validator {
    /// Create a new validator instance
    pub fn new() -> Self;
    
    /// Validate a container's integrity
    pub fn validate(&self, container: &Container) -> Result<ValidationResult, ValidationError>;
    
    /// Validate container signature
    pub fn validate_signature(&self, container: &Container) -> Result<bool, ValidationError>;
    
    /// Check container metadata
    pub fn validate_metadata(&self, metadata: &Metadata) -> Result<bool, ValidationError>;
}

pub struct ValidationResult {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}
```

### Backend Module

The backend module manages AI model execution engines.

```rust
use saic_core::backend::{BackendManager, Backend, BackendError, ExecutionResult};

pub struct BackendManager {
    // Backend manager configuration
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new() -> Self;
    
    /// Initialize a specific backend
    pub fn initialize_backend(&mut self, backend_type: &str) -> Result<(), BackendError>;
    
    /// Execute inference on a container
    pub fn execute(&self, container: &Container, input: &[f32]) -> Result<ExecutionResult, BackendError>;
    
    /// List available backends
    pub fn list_backends(&self) -> Vec<String>;
    
    /// Get backend capabilities
    pub fn get_capabilities(&self, backend_type: &str) -> Result<BackendCapabilities, BackendError>;
}

pub struct ExecutionResult {
    pub output: Vec<f32>,
    pub execution_time: Duration,
    pub memory_usage: usize,
}

pub struct BackendCapabilities {
    pub supported_models: Vec<String>,
    pub max_batch_size: usize,
    pub precision_support: Vec<String>,
}
```

### Crypto Module

The crypto module handles cryptographic operations and DNA key management.

```rust
use saic_core::crypto::{CryptoManager, DnaKey, CryptoError};

pub struct CryptoManager {
    // Crypto manager configuration
}

impl CryptoManager {
    /// Create a new crypto manager
    pub fn new() -> Self;
    
    /// Generate a new DNA key
    pub fn generate_dna_key(&self) -> Result<DnaKey, CryptoError>;
    
    /// Sign data with a DNA key
    pub fn sign(&self, data: &[u8], key: &DnaKey) -> Result<Vec<u8>, CryptoError>;
    
    /// Verify a signature
    pub fn verify(&self, data: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, CryptoError>;
}

pub struct DnaKey {
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub key_id: String,
}
```

### Config Module

The config module handles configuration management.

```rust
use saic_core::config::{Config, ConfigError};

pub struct Config {
    pub backends: BackendConfig,
    pub security: SecurityConfig,
    pub logging: LoggingConfig,
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, ConfigError>;
    
    /// Load configuration from string
    pub fn from_str(content: &str) -> Result<Self, ConfigError>;
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError>;
}

pub struct BackendConfig {
    pub cpu: CpuBackendConfig,
    pub gpu: GpuBackendConfig,
}

pub struct SecurityConfig {
    pub validate_signatures: bool,
    pub encryption_algorithm: String,
    pub allowed_origins: Vec<String>,
}
```

## Error Handling

The library uses custom error types for different operations:

```rust
use saic_core::error::{SaicError, SaicErrorKind};

pub enum SaicError {
    Parse(ParseError),
    Validation(ValidationError),
    Backend(BackendError),
    Crypto(CryptoError),
    Config(ConfigError),
    Io(std::io::Error),
}

impl std::error::Error for SaicError {}
impl std::fmt::Display for SaicError {}
```

## Examples

### Basic Container Processing

```rust
use saic_core::{Parser, Validator, BackendManager, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::from_file("config.yaml")?;
    
    // Initialize components
    let parser = Parser::new();
    let validator = Validator::new();
    let mut backend_manager = BackendManager::new();
    
    // Parse container
    let container = parser.parse_file("model.saic")?;
    
    // Validate container
    validator.validate(&container)?;
    
    // Initialize backend
    backend_manager.initialize_backend("cpu")?;
    
    // Prepare input data
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    
    // Execute inference
    let result = backend_manager.execute(&container, &input_data)?;
    
    println!("Output: {:?}", result.output);
    println!("Execution time: {:?}", result.execution_time);
    
    Ok(())
}
```

### Custom Backend Integration

```rust
use saic_core::backend::{Backend, BackendError, ExecutionResult};

pub struct CustomBackend {
    // Custom backend implementation
}

impl Backend for CustomBackend {
    fn name(&self) -> &str {
        "custom"
    }
    
    fn initialize(&mut self) -> Result<(), BackendError> {
        // Initialize custom backend
        Ok(())
    }
    
    fn execute(&self, model_data: &[u8], input: &[f32]) -> Result<ExecutionResult, BackendError> {
        // Execute custom inference
        Ok(ExecutionResult {
            output: vec![0.0; 1000],
            execution_time: Duration::from_millis(10),
            memory_usage: 1024,
        })
    }
    
    fn cleanup(&mut self) -> Result<(), BackendError> {
        // Cleanup resources
        Ok(())
    }
}
```

### DNA Key Management

```rust
use saic_core::crypto::{CryptoManager, DnaKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crypto_manager = CryptoManager::new();
    
    // Generate a new DNA key
    let dna_key = crypto_manager.generate_dna_key()?;
    
    // Sign some data
    let data = b"Hello, SAIC!";
    let signature = crypto_manager.sign(data, &dna_key)?;
    
    // Verify the signature
    let is_valid = crypto_manager.verify(data, &signature, &dna_key.public_key)?;
    
    println!("Signature valid: {}", is_valid);
    
    Ok(())
}
```

## Performance Considerations

### Memory Management

- The library uses zero-copy parsing where possible
- Large models are streamed rather than loaded entirely into memory
- Backend memory is managed efficiently with proper cleanup

### Threading

- Backend operations are thread-safe
- Multiple backends can be used concurrently
- Parsing and validation are designed for concurrent access

### Error Recovery

- Graceful handling of corrupted containers
- Automatic fallback to alternative backends
- Comprehensive error reporting for debugging

## Future Enhancements

The following features are planned for future releases:

1. **WebAssembly Backend**: Support for Wasm-based model execution
2. **Distributed Execution**: Multi-node inference capabilities
3. **Advanced Crypto**: Post-quantum cryptography support
4. **Plugin System**: Extensible backend architecture
5. **Performance Profiling**: Built-in performance analysis tools

## Contributing

To contribute to the `saic_core` library:

1. Follow the Rust coding standards
2. Add comprehensive tests for new features
3. Update this documentation for API changes
4. Ensure backward compatibility for stable releases

---

**Note**: This API reference is for the planned `saic_core` library. The actual implementation may vary as development progresses. Check the latest documentation for current API details. 