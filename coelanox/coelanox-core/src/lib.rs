//! COELANOX Core Library
//!
//! This is the core library containing shared data structures, utilities, and error types
//! used throughout the COELANOX ecosystem.
//!
//! ## Frontend SDK
//!
//! `coelanox-core` is the **Frontend SDK** for model translators (compiler frontends).
//! Implement [`CoelanoxTranslator`] to add support for new model formats (ONNX, PyTorch, etc.).
//! See [`plugins`] and [`ir`] for the stable API.

pub mod format;
pub mod utils;
pub mod error;
pub mod ir;
pub mod plugins;
pub mod scalar;
pub mod scratch;
pub mod optimization;
pub mod memory_hal;
pub mod protection_hal;
pub mod hal_mapping;

pub use error::{CoelanoxError, CoelanoxResult};
pub use memory_hal::{
    MemoryBackendRegistry, MemoryCapabilities, MemoryHAL, MemoryRegion, MemoryTarget, MemoryTier,
    RegionOptions,
};
pub use protection_hal::{
    ProtectionBackendRegistry, ProtectionCapabilities, ProtectionHAL, ProtectionTarget,
    RegionPermissions,
};
pub use hal_mapping::RegionKind;
pub use plugins::CoelanoxTranslator;

/// COELANOX magic number for file identification
pub const COELANOX_MAGIC: [u8; 4] = *b"COEL";

/// Validate magic number in data
pub fn validate_magic(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == COELANOX_MAGIC
}