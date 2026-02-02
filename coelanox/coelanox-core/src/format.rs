//! COELANOX container format
//! 
//! This module defines the COELANOX container format structure, including headers,
//! manifests, and validation. COELANOX is a hybrid container/format that combines
//! container encapsulation with optimized binary representation.

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::path::Path;
use sha2::{Sha256, Digest};
use crate::{CoelanoxError, CoelanoxResult, COELANOX_MAGIC};

/// COELANOX binary format constants
pub const COELANOX_BINARY_VERSION: u32 = 0x00010000; // Version 1.0.0
pub const COELANOX_HEADER_SIZE: usize = 76; // Fixed header size: 4(magic) + 4(version) + 4(flags) + 8(container_size) + 8(manifest_offset) + 8(code_offset) + 32(security_hash) + 8(reserved) = 76 bytes
pub const COELANOX_MANIFEST_OFFSET: usize = COELANOX_HEADER_SIZE;
pub const COELANOX_CODE_OFFSET: usize = COELANOX_MANIFEST_OFFSET + 1024; // Approximate manifest size

/// COELANOX header with binary layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoelanoxHeader {
    /// Magic number for identification
    pub magic: [u8; 4],
    /// Version number
    pub version: u32,
    /// Flags for container features
    pub flags: u32,
    /// Container size in bytes
    pub container_size: u64,
    /// Manifest offset
    pub manifest_offset: u64,
    /// Code offset
    pub code_offset: u64,
    /// Security hash
    pub security_hash: [u8; 32],
    /// Reserved for future use
    pub reserved: [u8; 8],
}

impl CoelanoxHeader {
    /// Create a new COELANOX header
    pub fn new(container_size: u64, manifest_offset: u64, code_offset: u64) -> Self {
        Self {
            magic: COELANOX_MAGIC,
            version: COELANOX_BINARY_VERSION,
            flags: 0,
            container_size,
            manifest_offset,
            code_offset,
            security_hash: [0; 32], // Set to [0; 32] during creation, calculated during write_binary_file()
            reserved: [0; 8],
        }
    }
    
    /// Validate the header
    pub fn validate(&self) -> CoelanoxResult<()> {
        if self.magic != COELANOX_MAGIC {
            return Err(CoelanoxError::InvalidMagic);
        }
        
        if self.version == 0 {
            return Err(CoelanoxError::UnsupportedVersion("Version cannot be zero".to_string()));
        }
        
        if self.container_size == 0 {
            return Err(CoelanoxError::PayloadValidation("Container size cannot be zero".to_string()));
        }
        
        Ok(())
    }
    
    /// Get version string
    pub fn version_string(&self) -> String {
        format!("{}.{}.{}", 
            (self.version >> 16) & 0xFF,
            (self.version >> 8) & 0xFF,
            self.version & 0xFF
        )
    }
    
    /// Write header to binary stream
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> CoelanoxResult<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        writer.write_all(&self.container_size.to_le_bytes())?;
        writer.write_all(&self.manifest_offset.to_le_bytes())?;
        writer.write_all(&self.code_offset.to_le_bytes())?;
        writer.write_all(&self.security_hash)?;
        writer.write_all(&self.reserved)?;
        Ok(())
    }
    
    /// Read header from binary stream
    pub fn read_binary<R: Read>(reader: &mut R) -> CoelanoxResult<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        
        let mut flags_bytes = [0u8; 4];
        reader.read_exact(&mut flags_bytes)?;
        let flags = u32::from_le_bytes(flags_bytes);
        
        let mut container_size_bytes = [0u8; 8];
        reader.read_exact(&mut container_size_bytes)?;
        let container_size = u64::from_le_bytes(container_size_bytes);
        
        let mut manifest_offset_bytes = [0u8; 8];
        reader.read_exact(&mut manifest_offset_bytes)?;
        let manifest_offset = u64::from_le_bytes(manifest_offset_bytes);
        
        let mut code_offset_bytes = [0u8; 8];
        reader.read_exact(&mut code_offset_bytes)?;
        let code_offset = u64::from_le_bytes(code_offset_bytes);
        
        let mut security_hash = [0u8; 32];
        reader.read_exact(&mut security_hash)?;
        
        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;
        
        Ok(Self {
            magic,
            version,
            flags,
            container_size,
            manifest_offset,
            code_offset,
            security_hash,
            reserved,
        })
    }
}

/// Model format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX format
    Onnx,
    /// COELANOX container format
    Coelanox,
    /// Custom format
    Custom(String),
}

/// Hardware target
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// CPU target
    Cpu,
    /// GPU target
    Gpu,
    /// Edge device target
    Edge,
    /// Apple Neural Engine
    NeuralEngine,
    /// NPU (Neural Processing Unit)
    NPU,
    /// Custom target
    Custom(String),
}

/// Optimization level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Advanced optimization
    Advanced,
    /// Maximum optimization
    Maximum,
}

/// Backend hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHints {
    /// Preferred backend
    pub preferred_backend: String,
    /// Fallback backend
    pub fallback_backend: String,
    /// Optimization hints
    pub optimization_hints: Vec<String>,
}

/// Security metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetadata {
    /// Digital signature
    pub signature: Option<Vec<u8>>,
    /// Checksum
    pub checksum: Option<Vec<u8>>,
    /// Encryption metadata
    pub encryption: Option<EncryptionMetadata>,
    /// Access control
    pub access_control: Option<AccessControl>,
}

/// Encryption metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// Encryption algorithm
    pub algorithm: String,
    /// Key ID
    pub key_id: String,
    /// Initialization vector
    pub iv: Vec<u8>,
}

/// Access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    /// Required permissions
    pub required_permissions: Vec<String>,
    /// Usage limits
    pub usage_limits: Option<UsageLimits>,
}

/// Usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimits {
    /// Maximum executions
    pub max_executions: Option<u64>,
    /// Expiration time
    pub expiration_time: Option<u64>,
    /// Rate limit
    pub rate_limit: Option<u64>,
}

/// Backend selection mode for AOT binary selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendSelectionMode {
    /// Slim mode: Selected backends only
    Slim {
        /// Primary backend name
        primary: String,
        /// Additional backend names
        additional: Vec<String>,
    },
    /// Fat mode: All available backends
    Fat,
    /// Auto mode: Auto-detected backends
    Auto,
}

/// Backend metadata for AOT binary selection
/// 
/// This metadata stores information about which backends were selected at package time
/// and how they are mapped to device IDs for runtime selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetadata {
    /// Primary backend (device ID 0)
    pub primary_backend: String,
    /// Additional backends (device IDs 1, 2, 3...)
    pub additional_backends: Vec<String>,
    /// Device ID mapping (device_id -> backend_name)
    pub device_id_mapping: std::collections::HashMap<usize, String>,
    /// Backend selection mode (slim, fat, auto)
    pub selection_mode: BackendSelectionMode,
}

impl BackendMetadata {
    /// Create new BackendMetadata for slim mode
    pub fn new_slim(primary: String, additional: Vec<String>) -> Self {
        let mut device_id_mapping = std::collections::HashMap::new();
        device_id_mapping.insert(0, primary.clone());
        
        for (idx, backend_name) in additional.iter().enumerate() {
            device_id_mapping.insert(idx + 1, backend_name.clone());
        }
        
        Self {
            primary_backend: primary.clone(),
            additional_backends: additional.clone(),
            device_id_mapping,
            selection_mode: BackendSelectionMode::Slim {
                primary,
                additional,
            },
        }
    }
    
    /// Create new BackendMetadata for fat mode
    pub fn new_fat(all_backends: Vec<String>) -> Self {
        let mut device_id_mapping = std::collections::HashMap::new();
        let primary = all_backends.first().cloned().unwrap_or_else(|| "cpu".to_string());
        let additional = all_backends.iter().skip(1).cloned().collect();
        
        device_id_mapping.insert(0, primary.clone());
        for (idx, backend_name) in all_backends.iter().skip(1).enumerate() {
            device_id_mapping.insert(idx + 1, backend_name.clone());
        }
        
        Self {
            primary_backend: primary,
            additional_backends: additional,
            device_id_mapping,
            selection_mode: BackendSelectionMode::Fat,
        }
    }
    
    /// Create new BackendMetadata for auto mode
    pub fn new_auto(detected_backends: Vec<String>) -> Self {
        let mut device_id_mapping = std::collections::HashMap::new();
        let primary = detected_backends.first().cloned().unwrap_or_else(|| "cpu".to_string());
        let additional = detected_backends.iter().skip(1).cloned().collect();
        
        device_id_mapping.insert(0, primary.clone());
        for (idx, backend_name) in detected_backends.iter().skip(1).enumerate() {
            device_id_mapping.insert(idx + 1, backend_name.clone());
        }
        
        Self {
            primary_backend: primary,
            additional_backends: additional,
            device_id_mapping,
            selection_mode: BackendSelectionMode::Auto,
        }
    }
    
    /// Get backend name for a device ID
    pub fn get_backend_for_device(&self, device_id: usize) -> Option<&String> {
        self.device_id_mapping.get(&device_id)
    }
    
    /// Validate device ID is in range
    pub fn validate_device_id(&self, device_id: usize) -> CoelanoxResult<()> {
        let max_id = self.device_id_mapping.keys().max().copied().unwrap_or(0);
        if device_id > max_id {
            return Err(CoelanoxError::DeviceIdOutOfRange(device_id, max_id));
        }
        Ok(())
    }
    
    /// Get backend for device ID with validation
    pub fn get_backend_for_device_checked(&self, device_id: usize) -> CoelanoxResult<&String> {
        self.validate_device_id(device_id)?;
        self.device_id_mapping.get(&device_id)
            .ok_or_else(|| CoelanoxError::DeviceIdOutOfRange(device_id, 
                self.device_id_mapping.keys().max().copied().unwrap_or(0)))
    }
}

/// COELANOX manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoelanoxManifest {
    /// Model name
    pub name: String,
    /// Model format
    pub format: ModelFormat,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Model size in bytes
    pub model_size: usize,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Model description
    pub description: Option<String>,
    /// Model tags
    pub tags: Vec<String>,
    /// Model metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Backend metadata (AOT selection)
    pub backend_metadata: Option<BackendMetadata>,
    /// NVMe backing data offset in container (for unified file approach)
    pub nvme_backing_offset: Option<u64>,
    /// NVMe backing data size in bytes
    pub nvme_backing_size: Option<usize>,
}

impl CoelanoxManifest {
    /// Create a new COELANOX manifest
    pub fn new(
        name: String,
        format: ModelFormat,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        model_size: usize,
    ) -> Self {
        let now = chrono::Utc::now().timestamp() as u64;
        
        Self {
            name,
            format,
            input_shapes,
            output_shapes,
            model_size,
            created_at: now,
            modified_at: now,
            description: None,
            tags: Vec::new(),
            metadata: std::collections::HashMap::new(),
            backend_metadata: None,
            nvme_backing_offset: None,
            nvme_backing_size: None,
        }
    }
    
    /// Validate the manifest
    pub fn validate(&self) -> CoelanoxResult<()> {
        if self.name.is_empty() {
            return Err(CoelanoxError::ManifestParse("Model name cannot be empty".to_string()));
        }
        
        if self.model_size == 0 {
            return Err(CoelanoxError::ManifestParse("Model size cannot be zero".to_string()));
        }
        
        if self.input_shapes.is_empty() {
            return Err(CoelanoxError::ManifestParse("Input shapes cannot be empty".to_string()));
        }
        
        if self.output_shapes.is_empty() {
            return Err(CoelanoxError::ManifestParse("Output shapes cannot be empty".to_string()));
        }
        
        Ok(())
    }
    
    /// Write manifest to binary stream
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> CoelanoxResult<()> {
        // Serialize to JSON for now (can be optimized later)
        let json_data = serde_json::to_vec(self)
            .map_err(|e| CoelanoxError::ManifestParse(format!("Failed to serialize manifest: {}", e)))?;
        
        // Write length prefix
        writer.write_all(&(json_data.len() as u32).to_le_bytes())?;
        
        // Write JSON data
        writer.write_all(&json_data)?;
        
        Ok(())
    }
    
    /// Read manifest from binary stream
    pub fn read_binary<R: Read>(reader: &mut R) -> CoelanoxResult<Self> {
        // Read length prefix
        let mut length_bytes = [0u8; 4];
        reader.read_exact(&mut length_bytes)?;
        let length = u32::from_le_bytes(length_bytes) as usize;
        
        // Read JSON data
        let mut json_data = vec![0u8; length];
        reader.read_exact(&mut json_data)?;
        
        // Deserialize from JSON
        let manifest: CoelanoxManifest = serde_json::from_slice(&json_data)
            .map_err(|e| CoelanoxError::ManifestParse(format!("Failed to deserialize manifest: {}", e)))?;
        
        Ok(manifest)
    }
}

/// Compression types supported by COELANOX
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// Zstandard compression
    Zstd,
    /// LZ4 compression
    Lz4,
    /// Deflate compression (fallback)
    Deflate,
}

impl CompressionType {
    /// Get compression type from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "zstd" => CompressionType::Zstd,
            "lz4" => CompressionType::Lz4,
            "deflate" => CompressionType::Deflate,
            _ => CompressionType::None,
        }
    }
    
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionType::None => "none",
            CompressionType::Zstd => "zstd",
            CompressionType::Lz4 => "lz4",
            CompressionType::Deflate => "deflate",
        }
    }
}

/// COELANOX container with binary format support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoelanoxContainer {
    /// Container header
    pub header: CoelanoxHeader,
    /// Container manifest
    pub manifest: CoelanoxManifest,
    /// Optimized machine code
    pub optimized_code: Vec<u8>,
    /// Compressed model weights
    pub compressed_weights: Vec<u8>,
    /// Compression type used for weights
    pub compression_type: CompressionType,
    /// IR data (serialized OptimizedIR using bincode)
    pub ir_data: Vec<u8>,
    /// NVMe backing data (pre-allocated memory layout)
    pub nvme_backing_data: Option<Vec<u8>>,
    /// Backend hints
    pub backend_hints: Option<BackendHints>,
    /// Security metadata
    pub security: Option<SecurityMetadata>,
}

impl CoelanoxContainer {
    /// Create a new COELANOX container
    pub fn new(manifest: CoelanoxManifest, optimized_code: Vec<u8>, compressed_weights: Vec<u8>, compression_type: CompressionType, ir_data: Vec<u8>) -> Self {
        Self::new_with_nvme(manifest, optimized_code, compressed_weights, compression_type, ir_data, None)
    }
    
    /// Create a new COELANOX container with NVMe backing data
    pub fn new_with_nvme(
        manifest: CoelanoxManifest,
        optimized_code: Vec<u8>,
        compressed_weights: Vec<u8>,
        compression_type: CompressionType,
        ir_data: Vec<u8>,
        nvme_backing_data: Option<Vec<u8>>,
    ) -> Self {
        let manifest_size = serde_json::to_vec(&manifest).unwrap_or_default().len();
        let code_offset = COELANOX_HEADER_SIZE as u64 + manifest_size as u64;
        
        // Calculate weights offset (with length prefix: 4 bytes)
        let weights_offset = code_offset + optimized_code.len() as u64;
        
        // Calculate IR offset (weights have 4-byte length prefix)
        let ir_offset = weights_offset + 4 + compressed_weights.len() as u64;
        
        // Calculate NVMe backing offset (IR has 4-byte length prefix)
        let nvme_backing_offset = ir_offset + 4 + ir_data.len() as u64;
        
        // Calculate total size including NVMe backing if present
        let nvme_size = nvme_backing_data.as_ref().map(|d| d.len()).unwrap_or(0) as u64;
        let total_size = if nvme_size > 0 {
            // NVMe backing has 8-byte length prefix (u64 for large files)
            nvme_backing_offset + 8 + nvme_size
        } else {
            ir_offset + 4 + ir_data.len() as u64
        };
        
        Self {
            header: CoelanoxHeader::new(total_size, COELANOX_HEADER_SIZE as u64, code_offset),
            manifest,
            optimized_code,
            compressed_weights,
            compression_type,
            ir_data,
            nvme_backing_data,
            backend_hints: None,
            security: None,
        }
    }
    
    /// Validate the container
    pub fn validate(&self) -> CoelanoxResult<()> {
        // Validate header
        self.header.validate()?;
        
        // Validate manifest
        self.manifest.validate()?;
        
        // Validate code size matches manifest
        if self.optimized_code.len() != self.manifest.model_size {
            return Err(CoelanoxError::PayloadValidation(format!(
                "Code size mismatch: expected {}, got {}",
                self.manifest.model_size,
                self.optimized_code.len()
            )));
        }
        
        Ok(())
    }
    
    /// Get total container size
    pub fn total_size(&self) -> usize {
        self.header.container_size as usize
    }
    
    /// Decompress weights (runtime - no external dependencies)
    pub fn decompress_weights(&self) -> CoelanoxResult<Vec<f32>> {
        match self.compression_type {
            CompressionType::Zstd => {
                self.decompress_zstd(&self.compressed_weights)
            }
            CompressionType::Lz4 => {
                self.decompress_lz4(&self.compressed_weights)
            }
            CompressionType::Deflate => {
                self.decompress_deflate(&self.compressed_weights)
            }
            CompressionType::None => {
                // Convert bytes to f32
                let mut weights = Vec::new();
                for chunk in self.compressed_weights.chunks(4) {
                    if chunk.len() == 4 {
                        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        weights.push(f32::from_le_bytes(bytes));
                    }
                }
                Ok(weights)
            }
        }
    }
    
    /// Built-in Zstd decompression (no external dependencies)
    fn decompress_zstd(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        // For now, use a simple fallback
        // TODO: Implement built-in Zstd decompression algorithm
        self.decompress_deflate(data)
    }
    
    /// Built-in LZ4 decompression (no external dependencies)
    fn decompress_lz4(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        // For now, use a simple fallback
        // TODO: Implement built-in LZ4 decompression algorithm
        self.decompress_deflate(data)
    }
    
    /// Built-in Deflate decompression (no external dependencies)
    fn decompress_deflate(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        // Simple fallback - assume data is uncompressed f32 values
        let mut weights = Vec::new();
        for chunk in data.chunks(4) {
            if chunk.len() == 4 {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                weights.push(f32::from_le_bytes(bytes));
            }
        }
        Ok(weights)
    }
    
    /// Write container to binary file
    /// 
    /// This method writes the container to a file and calculates the SHA-256 security hash.
    /// The hash is calculated on the entire container (after compression) and embedded in the header.
    pub fn write_binary_file(&self, path: &std::path::Path) -> CoelanoxResult<()> {
        // Write entire container to in-memory buffer first
        let mut buffer = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut buffer);
            self.write_binary(&mut cursor)?;
        }
        
        // Calculate SHA-256 hash of entire buffer (entire container as stored)
        let mut hasher = Sha256::new();
        hasher.update(&buffer);
        let calculated_hash: [u8; 32] = hasher.finalize().into();
        
        // Update header in buffer with calculated hash
        // Hash is at offset 36-67 (32 bytes) in the header
        let hash_offset = 36;
        if buffer.len() < hash_offset + 32 {
            return Err(CoelanoxError::Internal(
                "Buffer too small to contain header with hash".to_string()
            ));
        }
        
        buffer[hash_offset..hash_offset + 32].copy_from_slice(&calculated_hash);
        
        // Write final buffer to file
        std::fs::write(path, &buffer)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        Ok(())
    }
    
    /// Write container to binary stream
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> CoelanoxResult<()> {
        // Write header
        self.header.write_binary(writer)?;
        
        // Write manifest
        self.manifest.write_binary(writer)?;
        
        // Write optimized code
        writer.write_all(&self.optimized_code)?;
        
        // Write compressed weights with length prefix
        writer.write_all(&(self.compressed_weights.len() as u32).to_le_bytes())?;
        writer.write_all(&self.compressed_weights)?;
        
        // Write IR data with length prefix
        writer.write_all(&(self.ir_data.len() as u32).to_le_bytes())?;
        writer.write_all(&self.ir_data)?;
        
        // Write NVMe backing data with length prefix (if present)
        if let Some(ref nvme_backing) = self.nvme_backing_data {
            // Use u64 for length prefix to support large files
            writer.write_all(&(nvme_backing.len() as u64).to_le_bytes())?;
            writer.write_all(nvme_backing)?;
        }
        
        // TODO: Write backend hints and security metadata if present
        
        Ok(())
    }
    
    /// Read container from binary file
    pub fn read_binary_file(path: &std::path::Path) -> CoelanoxResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| CoelanoxError::FileSystem(e))?;
        
        Self::read_binary(&mut file)
    }
    
    /// Read container from binary stream
    pub fn read_binary<R: Read>(reader: &mut R) -> CoelanoxResult<Self> {
        // Read header
        let header = CoelanoxHeader::read_binary(reader)?;
        
        // Read manifest
        let manifest = CoelanoxManifest::read_binary(reader)?;
        
        // Read optimized code
        let mut optimized_code = vec![0u8; manifest.model_size];
        reader.read_exact(&mut optimized_code)?;
        
        // Read compressed weights length
        let mut weights_len_bytes = [0u8; 4];
        reader.read_exact(&mut weights_len_bytes)?;
        let weights_len = u32::from_le_bytes(weights_len_bytes) as usize;
        
        // Read compressed weights
        let mut compressed_weights = vec![0u8; weights_len];
        reader.read_exact(&mut compressed_weights)?;
        
        // Read IR data length
        let mut ir_len_bytes = [0u8; 4];
        reader.read_exact(&mut ir_len_bytes)?;
        let ir_len = u32::from_le_bytes(ir_len_bytes) as usize;
        
        // Read IR data
        let mut ir_data = vec![0u8; ir_len];
        reader.read_exact(&mut ir_data)?;
        
        // Read NVMe backing data if present (check manifest for size)
        // We read sequentially since we're already at the right position after IR data
        let nvme_backing_data = if let Some(nvme_size) = manifest.nvme_backing_size {
            if nvme_size > 0 {
                // Read NVMe backing length prefix (u64)
                let mut nvme_len_bytes = [0u8; 8];
                reader.read_exact(&mut nvme_len_bytes)?;
                let nvme_len = u64::from_le_bytes(nvme_len_bytes) as usize;
                
                // Validate size matches manifest
                if nvme_len != nvme_size {
                    return Err(CoelanoxError::PayloadValidation(
                        format!("NVMe backing size mismatch: manifest says {}, file has {}", nvme_size, nvme_len)
                    ));
                }
                
                // Read NVMe backing data
                let mut nvme_backing = vec![0u8; nvme_len];
                reader.read_exact(&mut nvme_backing)?;
                Some(nvme_backing)
            } else {
                None
            }
        } else {
            None
        };
        
        // TODO: Read backend hints and security metadata if present
        // TODO: Determine compression_type from manifest or metadata
        
        Ok(Self {
            header,
            manifest,
            optimized_code,
            compressed_weights,
            compression_type: CompressionType::None, // TODO: Read from manifest metadata
            ir_data,
            nvme_backing_data,
            backend_hints: None,
            security: None,
        })
    }
    
    /// Convert from JSON format to binary format
    pub fn to_binary(&self) -> CoelanoxResult<Vec<u8>> {
        let mut buffer = Vec::new();
        self.write_binary(&mut buffer)?;
        Ok(buffer)
    }
    
    /// Convert from binary format to JSON format
    pub fn from_binary(data: &[u8]) -> CoelanoxResult<Self> {
        let mut cursor = std::io::Cursor::new(data);
        Self::read_binary(&mut cursor)
    }
}

/// COELANOX validator
pub struct CoelanoxValidator;

impl CoelanoxValidator {
    /// Validate COELANOX file
    pub fn validate_file(path: &std::path::Path) -> CoelanoxResult<()> {
        // TODO: Implement full file validation
        // For now, just check if file exists
        if !path.exists() {
            return Err(CoelanoxError::FileSystem(io::Error::new(
                io::ErrorKind::NotFound,
                "File not found"
            )));
        }
        
        Ok(())
    }
    
    /// Validate COELANOX data
    pub fn validate_data(data: &[u8]) -> CoelanoxResult<()> {
        // TODO: Implement full data validation
        // For now, just check minimum size
        if data.len() < 4 {
            return Err(CoelanoxError::Format("Data too small".to_string()));
        }
        
        // Check magic number
        if !crate::validate_magic(data) {
            return Err(CoelanoxError::InvalidMagic);
        }
        
        Ok(())
    }
    
    /// Parse header from data
    pub fn parse_header(_data: &[u8]) -> CoelanoxResult<CoelanoxHeader> {
        // TODO: Implement header parsing
        // For now, return dummy header
        Ok(CoelanoxHeader::new(0, 0, 0))
    }
}

/// Verify container security hash (pre-deployment check)
/// 
/// This function reads a container file from disk (cold storage), calculates its SHA-256 hash,
/// and compares it with the hash stored in the container header. This is used for pre-deployment
/// verification before loading the container into the memory tier.
/// 
/// # Arguments
/// * `container_path` - Path to the container file to verify
/// 
/// # Returns
/// * `Ok(())` if hash matches (container is valid)
/// * `Err(CoelanoxError::SecurityValidation)` if hash mismatch (container may be tampered)
/// * `Err(CoelanoxError::FileSystem)` if file cannot be read
pub fn verify_security_hash(container_path: &Path) -> CoelanoxResult<()> {
    // Read entire container file from cold storage
    let container_data = std::fs::read(container_path)
        .map_err(|e| CoelanoxError::FileSystem(e))?;
    
    // Verify minimum size (must be at least header size)
    if container_data.len() < COELANOX_HEADER_SIZE {
        return Err(CoelanoxError::SecurityValidation(
            "Container file too small to contain valid header".to_string()
        ));
    }
    
    // Extract stored hash from header
    // Hash is at offset 36-67 (32 bytes) in the header
    // Header layout: magic(4) + version(4) + flags(4) + container_size(8) + manifest_offset(8) + code_offset(8) + security_hash(32) + reserved(8)
    let hash_offset = 36; // Start of security_hash field
    if container_data.len() < hash_offset + 32 {
        return Err(CoelanoxError::SecurityValidation(
            "Container file too small to contain security hash".to_string()
        ));
    }
    
    let stored_hash: [u8; 32] = container_data[hash_offset..hash_offset + 32]
        .try_into()
        .map_err(|_| CoelanoxError::SecurityValidation(
            "Failed to extract security hash from header".to_string()
        ))?;
    
    // Create a copy of container data with hash set to [0; 32] for hash calculation
    // The stored hash is the hash of the file with hash = [0; 32]
    let mut data_for_hashing = container_data.to_vec();
    let zero_hash = [0u8; 32];
    data_for_hashing[hash_offset..hash_offset + 32].copy_from_slice(&zero_hash);
    
    // Calculate SHA-256 hash of file with hash bytes set to zero
    let mut hasher = Sha256::new();
    hasher.update(&data_for_hashing);
    let calculated_hash: [u8; 32] = hasher.finalize().into();
    
    // Compare hashes
    if calculated_hash != stored_hash {
        return Err(CoelanoxError::SecurityValidation(
            "Container hash mismatch - file may have been tampered with".to_string()
        ));
    }
    
    Ok(())
}
