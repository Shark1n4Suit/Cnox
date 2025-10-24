//! COELANOX Core Library
//! 
//! This is the core library containing shared data structures, utilities, and error types
//! used throughout the COELANOX ecosystem.

pub mod format;
pub mod utils;
pub mod error;
pub mod ir;
pub mod plugins;

pub use error::{CoelanoxError, CoelanoxResult};

/// COELANOX magic number for file identification
pub const COELANOX_MAGIC: [u8; 4] = *b"COEL";

/// Validate magic number in data
pub fn validate_magic(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == COELANOX_MAGIC
}