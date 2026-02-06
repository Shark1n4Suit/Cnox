//! CLF (Coelanox Library File) reader API
//!
//! This crate defines the contract for loading a `.clf` file and reading its payload
//! (e.g. kernel blobs or weight array). The default implementation is a stub that
//! returns empty data so the Coelanox workspace builds without an external CLF repo.
//!
//! **To use your own implementation:** Replace this crate with your open-source repo
//! (git submodule or path dependency). See [docs/CLF_INTEGRATION.md](../../../docs/CLF_INTEGRATION.md).

use std::path::Path;
use thiserror::Error;

/// Errors produced by the CLF reader.
#[derive(Debug, Error)]
pub enum ClfError {
    #[error("CLF file not found or not readable: {0}")]
    Io(String),
    #[error("Invalid CLF format: {0}")]
    InvalidFormat(String),
    #[error("CLF blob/op not found: op_id={0}")]
    BlobNotFound(u32),
}

/// Result type for CLF operations.
pub type ClfResult<T> = Result<T, ClfError>;

/// CLF reader: load a `.clf` file and read its payload.
///
/// **Contract (for your implementation):**
/// - [ClfReader::open](ClfReader::open) loads the file at the given path.
/// - [ClfReader::read_blob_array](ClfReader::read_blob_array) returns the full payload
///   as one contiguous byte array (e.g. all kernel blobs concatenated, or the main
///   weight/kernel array).
/// - [ClfReader::get_blob](ClfReader::get_blob) returns a single blob by op_id when
///   the packager needs per-op blobs in IR order.
pub trait ClfReader: Send + Sync {
    /// Load a CLF file from the given path.
    fn open(path: &Path) -> ClfResult<Self>
    where
        Self: Sized;

    /// Read the full blob array as one contiguous byte array.
    /// Used when the consumer needs the entire payload (e.g. machine code for the container).
    fn read_blob_array(&self) -> ClfResult<Vec<u8>>;

    /// Get a single blob by op_id. Returns `None` if the op_id is not present.
    /// Used by the packager to concatenate blobs in IR node order.
    fn get_blob(&self, op_id: u32) -> Option<Vec<u8>>;

    /// List all op_ids present in the CLF (for validation or iteration).
    fn list_op_ids(&self) -> Vec<u32>;
}

/// Stub implementation: builds without an external CLF repo.
/// Replace this crate with your repo to provide a real implementation.
pub struct StubClfReader;

impl ClfReader for StubClfReader {
    fn open(_path: &Path) -> ClfResult<Self> {
        Ok(StubClfReader)
    }

    fn read_blob_array(&self) -> ClfResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn get_blob(&self, _op_id: u32) -> Option<Vec<u8>> {
        None
    }

    fn list_op_ids(&self) -> Vec<u32> {
        Vec::new()
    }
}

/// Default entry point: open a CLF file and return a reader.
/// Uses the stub implementation until you replace this crate with your own.
/// Your implementation can return `Box::new(YourReader::open(path)?)`.
pub fn open(path: &Path) -> ClfResult<Box<dyn ClfReader>> {
    StubClfReader::open(path).map(|r| Box::new(r) as Box<dyn ClfReader>)
}
