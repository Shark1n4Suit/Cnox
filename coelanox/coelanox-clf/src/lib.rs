//! CLF (Coelanox Library File) in-tree reader.
//!
//! Implements the public CLF spec: parse .clfc/.clf header and manifest, get blob by op_id.
//! Op registry matches the canonical op_id table so the packager can map IR ops to blobs.
//! CLF (kernel format and producer) remains a separate project; this crate only reads.

pub mod format;
pub mod op_registry;
pub mod reader;

pub use format::{ClfHeader, ClfKind, ManifestEntry, CLF_MAGIC, CLF_VERSION, SIG_BLOCK_LEN, SIG_MAGIC};
pub use op_registry::{op_type_to_clf_id, OpType};
pub use reader::{ClfError, ClfReader, ClfReaderFromBytes, MissingOpIdPolicy};
