//! COELANOX container format
//! 
//! This module defines the COELANOX container format structure, including headers,
//! manifests, and validation. COELANOX is a hybrid container/format that combines
//! container encapsulation with optimized binary representation.

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::path::Path;
use sha2::{Sha256, Digest};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand_core::OsRng;
use crate::{CoelanoxError, CoelanoxResult, COELANOX_MAGIC};

/// COELANOX binary format constants
pub const COELANOX_BINARY_VERSION: u32 = 0x00010000; // Version 1.0.0
pub const COELANOX_HEADER_SIZE: usize = 76; // Fixed header size: 4(magic) + 4(version) + 4(flags) + 8(container_size) + 8(manifest_offset) + 8(code_offset) + 32(security_hash) + 8(reserved) = 76 bytes
/// Byte offset of the 32-byte security_hash field in the header (4+4+4+8+8+8 = 36).
pub const COELANOX_HEADER_SECURITY_HASH_OFFSET: usize = 36;
pub const COELANOX_MANIFEST_OFFSET: usize = COELANOX_HEADER_SIZE;
pub const COELANOX_CODE_OFFSET: usize = COELANOX_MANIFEST_OFFSET + 1024; // Approximate manifest size

/// Header flag: container must run with audit enabled every time (compliance). Runtime defaults audit output to a default folder if not specified.
pub const COELANOX_FLAG_AUDIT_REQUIRED: u32 = 1 << 0;
/// Header flag: container has Ed25519 signature appended (64 bytes after content). Hash covers content only.
pub const COELANOX_FLAG_SIGNED: u32 = 1 << 1;

/// Ed25519 signature length in bytes.
pub const ED25519_SIGNATURE_LEN: usize = 64;

/// Returns true if the header flags indicate audit is required for every run.
#[inline]
pub fn header_audit_required(flags: u32) -> bool {
    (flags & COELANOX_FLAG_AUDIT_REQUIRED) != 0
}

/// Returns true if the header flags indicate the container has an Ed25519 signature appended.
#[inline]
pub fn header_signed(flags: u32) -> bool {
    (flags & COELANOX_FLAG_SIGNED) != 0
}

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
    /// Create a new COELANOX header with no flags.
    pub fn new(container_size: u64, manifest_offset: u64, code_offset: u64) -> Self {
        Self::new_with_flags(container_size, manifest_offset, code_offset, 0)
    }

    /// Create a new COELANOX header with the given flags (e.g. COELANOX_FLAG_AUDIT_REQUIRED).
    pub fn new_with_flags(container_size: u64, manifest_offset: u64, code_offset: u64, flags: u32) -> Self {
        Self {
            magic: COELANOX_MAGIC,
            version: COELANOX_BINARY_VERSION,
            flags,
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

/// IR semantics version embedded in packaged containers. Bump when UniversalIR / scalar contract changes incompatibly.
pub const COELANOX_IR_SEMANTICS_VERSION: &str = "1";

/// Provenance recorded at package time (translator + flags). Helps explain “same model, different behavior.”
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackagingFingerprint {
    pub translator_name: String,
    pub translator_version: String,
    pub use_demo_translator: bool,
    pub fallback_only: bool,
    pub target_str: String,
    pub optimization: String,
    pub packager_crate_version: String,
}

/// Reject load when container declares an IR semantics version newer than this runtime understands.
/// Element count for one tensor shape (product of dimensions). Errors on overflow.
pub fn shape_element_count(shape: &[usize]) -> CoelanoxResult<usize> {
    let mut p = 1usize;
    for &d in shape {
        p = p.checked_mul(d).ok_or_else(|| {
            CoelanoxError::ManifestParse(
                "Tensor shape dimension product overflow (untrusted or malformed manifest)".to_string(),
            )
        })?;
    }
    Ok(p)
}

/// Sum of element counts for every tensor in `shapes` (e.g. manifest input or output list).
pub fn shape_sets_total_elements(shapes: &[Vec<usize>]) -> CoelanoxResult<usize> {
    let mut sum = 0usize;
    for shape in shapes {
        let n = shape_element_count(shape)?;
        sum = sum.checked_add(n).ok_or_else(|| {
            CoelanoxError::ManifestParse(
                "Total tensor element count overflow (untrusted or malformed manifest)".to_string(),
            )
        })?;
    }
    Ok(sum)
}

pub fn validate_ir_semantics_version(manifest: &CoelanoxManifest) -> CoelanoxResult<()> {
    match &manifest.ir_semantics_version {
        None => Ok(()),
        Some(v) if v == COELANOX_IR_SEMANTICS_VERSION => Ok(()),
        Some(v) => Err(CoelanoxError::Runtime(format!(
            "IR semantics version mismatch: container has '{}', this runtime supports '{}'. Repackage with a matching toolchain or use a compatible runtime.",
            v, COELANOX_IR_SEMANTICS_VERSION
        ))),
    }
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
    /// CLFMM (memory movement) data size in bytes; present when container embeds memory movement
    #[serde(default)]
    pub clfmm_size: Option<usize>,
    /// CLFMP (memory protection) data size in bytes; present when container embeds memory protection
    #[serde(default)]
    pub clfmp_size: Option<usize>,
    /// Execution plan size in bytes (CLF mini-executor); present when container has multi-blob code
    #[serde(default)]
    pub execution_plan_size: Option<usize>,
    /// Workspace required in elements (f32) for CLF plan execution. Set at package time; runtime can cap via --max-workspace-mb or env.
    #[serde(default)]
    pub workspace_required_elements: Option<u64>,
    /// UniversalIR semantics contract version this container was built for (`COELANOX_IR_SEMANTICS_VERSION`).
    #[serde(default)]
    pub ir_semantics_version: Option<String>,
    /// Translator + packaging flags at build time.
    #[serde(default)]
    pub packaging_fingerprint: Option<PackagingFingerprint>,
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
            clfmm_size: None,
            clfmp_size: None,
            execution_plan_size: None,
            workspace_required_elements: None,
            ir_semantics_version: Some(COELANOX_IR_SEMANTICS_VERSION.to_string()),
            packaging_fingerprint: None,
        }
    }
    
    /// Validate the manifest
    pub fn validate(&self) -> CoelanoxResult<()> {
        if self.name.is_empty() {
            return Err(CoelanoxError::ManifestParse("Model name cannot be empty".to_string()));
        }
        
        // model_size may be 0 for scalar/interpreted fallback (no generated code blob)
        if self.input_shapes.is_empty() {
            return Err(CoelanoxError::ManifestParse("Input shapes cannot be empty".to_string()));
        }
        
        if self.output_shapes.is_empty() {
            return Err(CoelanoxError::ManifestParse("Output shapes cannot be empty".to_string()));
        }

        let _ = shape_sets_total_elements(&self.input_shapes)?;
        let _ = shape_sets_total_elements(&self.output_shapes)?;
        
        Ok(())
    }
    
    /// Write manifest to binary stream
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> CoelanoxResult<()> {
        // JSON manifest format; binary manifest variant not implemented
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
    /// Compressed model weights. Wrapped in Arc+Mutex so runtime can release after first decompress
    /// to avoid holding both compressed and decompressed in memory (~500MB+ savings for BERT).
    #[serde(skip)]
    pub compressed_weights: std::sync::Arc<std::sync::Mutex<Option<Vec<u8>>>>,
    /// Compression type used for weights
    pub compression_type: CompressionType,
    /// IR data (serialized OptimizedIR using bincode)
    pub ir_data: Vec<u8>,
    /// NVMe backing data (pre-allocated memory layout)
    pub nvme_backing_data: Option<Vec<u8>>,
    /// CLFMM (memory movement) blobs; container pulls memory via these. OS-independent.
    pub clfmm_data: Option<Vec<u8>>,
    /// CLFMP (memory protection) blobs; container configures protection via these.
    pub clfmp_data: Option<Vec<u8>>,
    /// Execution plan for CLF mini-executor (multi-blob dispatch); present when code is kernel blobs in graph order
    pub execution_plan: Option<Vec<u8>>,
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
        Self::new_with_hal_data(manifest, optimized_code, compressed_weights, compression_type, ir_data, nvme_backing_data, None, None)
    }

    /// Create a new COELANOX container with NVMe, CLFMM, CLFMP, and optional execution plan data
    pub fn new_with_hal_data(
        manifest: CoelanoxManifest,
        optimized_code: Vec<u8>,
        compressed_weights: Vec<u8>,
        compression_type: CompressionType,
        ir_data: Vec<u8>,
        nvme_backing_data: Option<Vec<u8>>,
        clfmm_data: Option<Vec<u8>>,
        clfmp_data: Option<Vec<u8>>,
    ) -> Self {
        Self::new_with_hal_data_and_plan(
            manifest, optimized_code, compressed_weights, compression_type, ir_data,
            nvme_backing_data, clfmm_data, clfmp_data, None, 0,
        )
    }

    /// Create a new COELANOX container with NVMe, CLFMM, CLFMP, and optional execution plan (CLF mini-executor).
    /// `header_flags` is passed to the container header (e.g. COELANOX_FLAG_AUDIT_REQUIRED for compliance).
    pub fn new_with_hal_data_and_plan(
        manifest: CoelanoxManifest,
        optimized_code: Vec<u8>,
        compressed_weights: Vec<u8>,
        compression_type: CompressionType,
        ir_data: Vec<u8>,
        nvme_backing_data: Option<Vec<u8>>,
        clfmm_data: Option<Vec<u8>>,
        clfmp_data: Option<Vec<u8>>,
        execution_plan: Option<Vec<u8>>,
        header_flags: u32,
    ) -> Self {
        let manifest_size = serde_json::to_vec(&manifest).unwrap_or_default().len();
        let code_offset = COELANOX_HEADER_SIZE as u64 + manifest_size as u64;
        
        // Calculate weights offset (with length prefix: 4 bytes)
        let weights_offset = code_offset + optimized_code.len() as u64;
        
        // Calculate IR offset (weights have 4-byte length prefix)
        let ir_offset = weights_offset + 4 + compressed_weights.len() as u64;
        
        // Calculate NVMe backing offset (IR has 4-byte length prefix)
        let nvme_backing_offset = ir_offset + 4 + ir_data.len() as u64;
        
        // Calculate total size including NVMe backing and optional CLFMM/CLFMP/execution_plan
        let nvme_size = nvme_backing_data.as_ref().map(|d| d.len()).unwrap_or(0) as u64;
        let base_after_nvme = if nvme_size > 0 {
            nvme_backing_offset + 8 + nvme_size
        } else {
            ir_offset + 4 + ir_data.len() as u64
        };

        let clfmm_size = clfmm_data.as_ref().map(|d| d.len());
        let clfmp_size = clfmp_data.as_ref().map(|d| d.len());
        let execution_plan_size = execution_plan.as_ref().map(|d| d.len());
        let mut manifest = manifest;
        manifest.clfmm_size = clfmm_size;
        manifest.clfmp_size = clfmp_size;
        manifest.execution_plan_size = execution_plan_size;

        let total_size = base_after_nvme
            + clfmm_data.as_ref().map(|d| 8 + d.len() as u64).unwrap_or(0)
            + clfmp_data.as_ref().map(|d| 8 + d.len() as u64).unwrap_or(0)
            + execution_plan.as_ref().map(|d| 8 + d.len() as u64).unwrap_or(0);

        Self {
            header: CoelanoxHeader::new_with_flags(total_size, COELANOX_HEADER_SIZE as u64, code_offset, header_flags),
            manifest,
            optimized_code,
            compressed_weights: std::sync::Arc::new(std::sync::Mutex::new(Some(compressed_weights))),
            compression_type,
            ir_data,
            nvme_backing_data,
            clfmm_data,
            clfmp_data,
            execution_plan,
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

    /// Size of compressed weights in bytes (0 if already released to save memory).
    pub fn compressed_weights_len(&self) -> usize {
        self.compressed_weights
            .lock()
            .map(|g| g.as_ref().map(|v| v.len()).unwrap_or(0))
            .unwrap_or(0)
    }
    
    /// Load weights into a Vec<f32>. Consumes the compressed buffer on first call so the runtime
    /// can avoid holding both compressed and decompressed in memory (~500MB+ savings for BERT).
    /// Subsequent calls return an error if weights were already released; use warm cache instead.
    pub fn decompress_weights(&self) -> CoelanoxResult<Vec<f32>> {
        let data = self
            .compressed_weights
            .lock()
            .map_err(|_| CoelanoxError::Runtime("compressed_weights mutex poisoned".to_string()))?
            .take()
            .ok_or_else(|| {
                CoelanoxError::Runtime(
                    "Weights already decompressed and released; use warm cache".to_string(),
                )
            })?;
        match self.compression_type {
            CompressionType::Zstd => self.decompress_zstd(&data),
            CompressionType::Lz4 => self.decompress_lz4(&data),
            CompressionType::Deflate => self.decompress_deflate(&data),
            CompressionType::None => {
                if data.len() % 4 != 0 {
                    return Err(CoelanoxError::PayloadValidation(
                        "Uncompressed weights length is not a multiple of 4".to_string(),
                    ));
                }
                let n = data.len() / 4;
                let mut weights = Vec::with_capacity(n);
                weights.extend(
                    data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())),
                );
                Ok(weights)
            }
        }
    }
    
    /// Zstd decompression. Currently falls back to raw-f32 passthrough; native Zstd not implemented.
    fn decompress_zstd(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        self.decompress_deflate(data)
    }

    /// LZ4 decompression. Currently falls back to raw-f32 passthrough; native LZ4 not implemented.
    fn decompress_lz4(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        self.decompress_deflate(data)
    }

    /// Fallback: treats data as raw little-endian f32 chunks (no actual deflate). Used when CompressionType::None or as passthrough for Zstd/LZ4 until native decompressors exist.
    fn decompress_deflate(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        let mut weights = Vec::new();
        for chunk in data.chunks(4) {
            if chunk.len() == 4 {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                weights.push(f32::from_le_bytes(bytes));
            }
        }
        Ok(weights)
    }
    
    /// Write container to binary file.
    ///
    /// Computes SHA-256 hash and embeds it in the header. When `signing_key` is `Some`,
    /// sets COELANOX_FLAG_SIGNED and appends a 64-byte Ed25519 signature over the hash.
    /// The signing key must be 32 bytes (Ed25519 seed).
    pub fn write_binary_file(&self, path: &std::path::Path) -> CoelanoxResult<()> {
        self.write_binary_file_with_signing(path, None)
    }

    /// Write container to binary file with optional Ed25519 signing.
    ///
    /// When `signing_key` is `Some`, the container is signed and the signature is appended.
    /// The key must be exactly 32 bytes (Ed25519 secret key seed).
    pub fn write_binary_file_with_signing(
        &self,
        path: &std::path::Path,
        signing_key: Option<&[u8; 32]>,
    ) -> CoelanoxResult<()> {
        // Build header with signing flag if we will sign
        let sign_flags = if signing_key.is_some() {
            self.header.flags | COELANOX_FLAG_SIGNED
        } else {
            self.header.flags
        };
        let mut header = self.header.clone();
        header.flags = sign_flags;

        // Write container to buffer (header with hash=0)
        let mut buffer = Vec::new();
        {
            let cw = self
                .compressed_weights
                .lock()
                .map_err(|_| CoelanoxError::Runtime("compressed_weights mutex poisoned".to_string()))?;
            let cw_ref = cw
                .as_ref()
                .ok_or_else(|| CoelanoxError::Runtime("Cannot write container: weights already released".to_string()))?;
            let mut cursor = std::io::Cursor::new(&mut buffer);
            header.write_binary(&mut cursor)?;
            self.manifest.write_binary(&mut cursor)?;
            cursor.write_all(&self.optimized_code)?;
            cursor.write_all(&(cw_ref.len() as u32).to_le_bytes())?;
            cursor.write_all(cw_ref)?;
            cursor.write_all(&(self.ir_data.len() as u32).to_le_bytes())?;
            cursor.write_all(&self.ir_data)?;
            if let Some(ref nvme) = self.nvme_backing_data {
                cursor.write_all(&(nvme.len() as u64).to_le_bytes())?;
                cursor.write_all(nvme)?;
            }
            if let Some(ref d) = self.clfmm_data {
                cursor.write_all(&(d.len() as u64).to_le_bytes())?;
                cursor.write_all(d)?;
            }
            if let Some(ref d) = self.clfmp_data {
                cursor.write_all(&(d.len() as u64).to_le_bytes())?;
                cursor.write_all(d)?;
            }
            if let Some(ref p) = self.execution_plan {
                cursor.write_all(&(p.len() as u64).to_le_bytes())?;
                cursor.write_all(p)?;
            }
        }

        // Hash: header (hash zeroed) + rest
        let mut hasher = Sha256::new();
        hasher.update(&buffer[..COELANOX_HEADER_SECURITY_HASH_OFFSET]);
        hasher.update(&[0u8; 32]);
        hasher.update(&buffer[COELANOX_HEADER_SECURITY_HASH_OFFSET + 32..]);
        let calculated_hash: [u8; 32] = hasher.finalize().into();

        buffer[COELANOX_HEADER_SECURITY_HASH_OFFSET..COELANOX_HEADER_SECURITY_HASH_OFFSET + 32]
            .copy_from_slice(&calculated_hash);

        if let Some(key_bytes) = signing_key {
            let signing_key = SigningKey::from_bytes(key_bytes);
            let signature: Signature = signing_key.sign(&calculated_hash);
            buffer.extend_from_slice(&signature.to_bytes());
        }

        std::fs::write(path, &buffer).map_err(CoelanoxError::FileSystem)?;
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
        let cw = self
            .compressed_weights
            .lock()
            .map_err(|_| CoelanoxError::Runtime("compressed_weights mutex poisoned".to_string()))?;
        let cw_ref = cw
            .as_ref()
            .ok_or_else(|| CoelanoxError::Runtime("Cannot write container: weights already released".to_string()))?;
        writer.write_all(&(cw_ref.len() as u32).to_le_bytes())?;
        writer.write_all(cw_ref)?;
        
        // Write IR data with length prefix
        writer.write_all(&(self.ir_data.len() as u32).to_le_bytes())?;
        writer.write_all(&self.ir_data)?;
        
        // Write NVMe backing data with length prefix (if present)
        if let Some(ref nvme_backing) = self.nvme_backing_data {
            writer.write_all(&(nvme_backing.len() as u64).to_le_bytes())?;
            writer.write_all(nvme_backing)?;
        }

        // Write CLFMM data with length prefix (if present)
        if let Some(ref clfmm) = self.clfmm_data {
            writer.write_all(&(clfmm.len() as u64).to_le_bytes())?;
            writer.write_all(clfmm)?;
        }

        // Write CLFMP data with length prefix (if present)
        if let Some(ref clfmp) = self.clfmp_data {
            writer.write_all(&(clfmp.len() as u64).to_le_bytes())?;
            writer.write_all(clfmp)?;
        }

        // Write execution plan with length prefix (if present; CLF mini-executor)
        if let Some(ref plan) = self.execution_plan {
            writer.write_all(&(plan.len() as u64).to_le_bytes())?;
            writer.write_all(plan)?;
        }
        
        // Future: write backend_hints, security metadata if present in manifest
        
        Ok(())
    }
    
    /// Read container from binary file
    pub fn read_binary_file(path: &std::path::Path) -> CoelanoxResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(CoelanoxError::FileSystem)?;
        
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
        let mut compressed_weights_raw = vec![0u8; weights_len];
        reader.read_exact(&mut compressed_weights_raw)?;
        let compressed_weights =
            std::sync::Arc::new(std::sync::Mutex::new(Some(compressed_weights_raw)));
        
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

        // Read CLFMM data if present (manifest.clfmm_size)
        let clfmm_data = if let Some(clfmm_size) = manifest.clfmm_size {
            if clfmm_size > 0 {
                let mut len_bytes = [0u8; 8];
                reader.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;
                if len != clfmm_size {
                    return Err(CoelanoxError::PayloadValidation(
                        format!("CLFMM size mismatch: manifest says {}, file has {}", clfmm_size, len)
                    ));
                }
                let mut data = vec![0u8; len];
                reader.read_exact(&mut data)?;
                Some(data)
            } else {
                None
            }
        } else {
            None
        };

        // Read CLFMP data if present (manifest.clfmp_size)
        let clfmp_data = if let Some(clfmp_size) = manifest.clfmp_size {
            if clfmp_size > 0 {
                let mut len_bytes = [0u8; 8];
                reader.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;
                if len != clfmp_size {
                    return Err(CoelanoxError::PayloadValidation(
                        format!("CLFMP size mismatch: manifest says {}, file has {}", clfmp_size, len)
                    ));
                }
                let mut data = vec![0u8; len];
                reader.read_exact(&mut data)?;
                Some(data)
            } else {
                None
            }
        } else {
            None
        };

        // Read execution plan if present (manifest.execution_plan_size)
        let execution_plan = if let Some(plan_size) = manifest.execution_plan_size {
            if plan_size > 0 {
                let mut len_bytes = [0u8; 8];
                reader.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;
                if len != plan_size {
                    return Err(CoelanoxError::PayloadValidation(
                        format!("Execution plan size mismatch: manifest says {}, file has {}", plan_size, len)
                    ));
                }
                let mut data = vec![0u8; len];
                reader.read_exact(&mut data)?;
                Some(data)
            } else {
                None
            }
        } else {
            None
        };
        
        // Future: read backend_hints, security metadata, compression_type from manifest
        
        Ok(Self {
            header,
            manifest,
            optimized_code,
            compressed_weights,
            compression_type: CompressionType::None, // Manifest has no field; weights assumed uncompressed on read
            ir_data,
            nvme_backing_data,
            clfmm_data,
            clfmp_data,
            execution_plan,
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
    /// Validate COELANOX file (existence check; full header/manifest/hash validation not implemented).
    pub fn validate_file(path: &std::path::Path) -> CoelanoxResult<()> {
        if !path.exists() {
            return Err(CoelanoxError::FileSystem(io::Error::new(
                io::ErrorKind::NotFound,
                "File not found"
            )));
        }
        
        Ok(())
    }
    
    /// Validate COELANOX data (min size + magic; full validation not implemented).
    pub fn validate_data(data: &[u8]) -> CoelanoxResult<()> {
        if data.len() < 4 {
            return Err(CoelanoxError::Format("Data too small".to_string()));
        }
        
        // Check magic number
        if !crate::validate_magic(data) {
            return Err(CoelanoxError::InvalidMagic);
        }
        
        Ok(())
    }
    
    /// Parse header from data (stub: returns dummy header; full parsing in read_binary).
    pub fn parse_header(_data: &[u8]) -> CoelanoxResult<CoelanoxHeader> {
        Ok(CoelanoxHeader::new(0, 0, 0))
    }
}

/// Verify container security hash from in-memory data (e.g. mmap slice).
/// When the container has COELANOX_FLAG_SIGNED, the last 64 bytes are the signature;
/// the hash covers only the content (first len-64 bytes).
pub fn verify_security_hash_from_slice(container_data: &[u8]) -> CoelanoxResult<()> {
    if container_data.len() < COELANOX_HEADER_SIZE {
        return Err(CoelanoxError::SecurityValidation(
            "Container file too small to contain valid header".to_string()
        ));
    }
    if container_data.len() < COELANOX_HEADER_SECURITY_HASH_OFFSET + 32 {
        return Err(CoelanoxError::SecurityValidation(
            "Container file too small to contain security hash".to_string()
        ));
    }
    let flags = u32::from_le_bytes(
        container_data[8..12].try_into().map_err(|_| {
            CoelanoxError::SecurityValidation("Failed to read header flags".to_string())
        })?,
    );
    let content_len = if header_signed(flags) {
        if container_data.len() < COELANOX_HEADER_SIZE + ED25519_SIGNATURE_LEN {
            return Err(CoelanoxError::SecurityValidation(
                "Signed container too small for signature trailer".to_string()
            ));
        }
        container_data.len() - ED25519_SIGNATURE_LEN
    } else {
        container_data.len()
    };
    let stored_hash: [u8; 32] = container_data[COELANOX_HEADER_SECURITY_HASH_OFFSET
        ..COELANOX_HEADER_SECURITY_HASH_OFFSET + 32]
        .try_into()
        .map_err(|_| CoelanoxError::SecurityValidation(
            "Failed to extract security hash from header".to_string()
        ))?;
    let mut hasher = Sha256::new();
    hasher.update(&container_data[..COELANOX_HEADER_SECURITY_HASH_OFFSET]);
    hasher.update(&[0u8; 32]);
    hasher.update(&container_data[COELANOX_HEADER_SECURITY_HASH_OFFSET + 32..content_len]);
    let calculated_hash: [u8; 32] = hasher.finalize().into();
    if calculated_hash != stored_hash {
        return Err(CoelanoxError::SecurityValidation(
            "Container hash mismatch - file may have been tampered with".to_string()
        ));
    }
    Ok(())
}

/// Verify Ed25519 signature on a signed container.
/// Call after verify_security_hash_from_slice. Fails if container is not signed or signature is invalid.
pub fn verify_signature_from_slice(
    container_data: &[u8],
    public_key: &[u8; 32],
) -> CoelanoxResult<()> {
    if container_data.len() < COELANOX_HEADER_SIZE {
        return Err(CoelanoxError::SecurityValidation(
            "Container too small for signature verification".to_string()
        ));
    }
    let flags = u32::from_le_bytes(
        container_data[8..12].try_into().map_err(|_| {
            CoelanoxError::SecurityValidation("Failed to read header flags".to_string())
        })?,
    );
    if !header_signed(flags) {
        return Err(CoelanoxError::SecurityValidation(
            "Container is not signed; cannot verify signature".to_string()
        ));
    }
    if container_data.len() < COELANOX_HEADER_SIZE + ED25519_SIGNATURE_LEN {
        return Err(CoelanoxError::SecurityValidation(
            "Signed container too small for signature".to_string()
        ));
    }
    let hash: [u8; 32] = container_data[COELANOX_HEADER_SECURITY_HASH_OFFSET
        ..COELANOX_HEADER_SECURITY_HASH_OFFSET + 32]
        .try_into()
        .map_err(|_| CoelanoxError::SecurityValidation("Invalid hash in header".to_string()))?;
    let sig_bytes = &container_data[container_data.len() - ED25519_SIGNATURE_LEN..];
    let signature = Signature::from_bytes(sig_bytes.try_into().map_err(|_| {
        CoelanoxError::SecurityValidation("Invalid signature length".to_string())
    })?);
    let verifying_key = VerifyingKey::from_bytes(public_key)
        .map_err(|_| CoelanoxError::SecurityValidation("Invalid public key".to_string()))?;
    verifying_key
        .verify(&hash, &signature)
        .map_err(|_| CoelanoxError::SecurityValidation(
            "Signature verification failed - container may have been tampered or key mismatch".to_string()
        ))?;
    Ok(())
}

/// Verify container security hash (pre-deployment check).
///
/// When `trusted_public_key` is `Some`, also verifies the Ed25519 signature if the container is signed.
pub fn verify_security_hash(container_path: &Path) -> CoelanoxResult<()> {
    verify_security_hash_with_trusted_key(container_path, None)
}

/// Verify container integrity and optionally the Ed25519 signature.
///
/// When `trusted_public_key` is `Some` and the container has COELANOX_FLAG_SIGNED,
/// verifies the signature over the hash. When signed but key is None, only hash is verified (integrity, not provenance).
pub fn verify_security_hash_with_trusted_key(
    container_path: &Path,
    trusted_public_key: Option<&[u8; 32]>,
) -> CoelanoxResult<()> {
    let container_data = std::fs::read(container_path).map_err(CoelanoxError::FileSystem)?;
    verify_security_hash_from_slice(&container_data)?;
    if let Some(key) = trusted_public_key {
        if container_data.len() >= COELANOX_HEADER_SIZE {
            let flags = u32::from_le_bytes(container_data[8..12].try_into().unwrap_or([0; 4]));
            if header_signed(flags) {
                verify_signature_from_slice(&container_data, key)?;
            }
        }
    }
    Ok(())
}

/// Check if a container file has the signed flag (COELANOX_FLAG_SIGNED).
/// Reads only the header; does not verify hash or signature.
pub fn is_container_signed(path: &Path) -> CoelanoxResult<bool> {
    let data = std::fs::read(path).map_err(CoelanoxError::FileSystem)?;
    Ok(is_container_signed_from_slice(&data))
}

/// Check if container data has the signed flag. Requires at least 12 bytes (magic + version + flags).
pub fn is_container_signed_from_slice(data: &[u8]) -> bool {
    if data.len() < 12 {
        return false;
    }
    let flags = u32::from_le_bytes(data[8..12].try_into().unwrap_or([0; 4]));
    header_signed(flags)
}

/// Generate a new Ed25519 keypair for container signing.
/// Returns (secret_key_32_bytes, public_key_32_bytes).
pub fn generate_signing_keypair() -> ([u8; 32], [u8; 32]) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let secret = signing_key.to_bytes();
    let public = signing_key.verifying_key().to_bytes();
    (secret, public)
}
