//! CLF binary format types and constants (in-tree reader; matches public CLF spec).
//! All multi-byte fields are little-endian.

/// Magic bytes at the start of every CLF file: "CLF1".
pub const CLF_MAGIC: [u8; 4] = [0x43, 0x4C, 0x46, 0x31];

/// Current format version; readers reject version > CLF_VERSION.
pub const CLF_VERSION: u8 = 2;

/// CLF file kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ClfKind {
    Compute = 0,
    MemoryMovement = 1,
    MemoryProtection = 2,
    Executor = 3,
}

impl ClfKind {
    pub const fn default_for_v1() -> Self {
        Self::Compute
    }
    pub fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::MemoryMovement,
            2 => Self::MemoryProtection,
            3 => Self::Executor,
            _ => Self::Compute,
        }
    }
}

/// Signature block: "SIG0" + 32-byte hash.
pub const SIG_MAGIC: [u8; 4] = [0x53, 0x49, 0x47, 0x30];
pub const SIG_HASH_LEN: usize = 32;
pub const SIG_BLOCK_LEN: usize = 4 + SIG_HASH_LEN;

/// Parsed CLF header.
#[derive(Debug, Clone)]
pub struct ClfHeader {
    pub version: u8,
    pub vendor: String,
    pub target: String,
    pub blob_alignment: u8,
    pub kind: ClfKind,
    pub header_end: u64,
}

/// Manifest entry: op_id → (offset, size) into blob store.
#[derive(Debug, Clone, Copy)]
pub struct ManifestEntry {
    pub op_id: u32,
    pub offset: u32,
    pub size: u32,
}

impl ManifestEntry {
    pub const ENTRY_SIZE: usize = 4 + 4 + 4;
}
