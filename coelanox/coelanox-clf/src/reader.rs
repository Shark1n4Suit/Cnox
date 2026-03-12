//! CLF reader: open .clf, parse header and manifest, expose get_blob(op_id). In-tree implementation of the public CLF spec.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use thiserror::Error;

use crate::format::{
    ClfHeader, ClfKind, ManifestEntry, CLF_MAGIC, CLF_VERSION, SIG_BLOCK_LEN, SIG_MAGIC,
};

/// Policy when an op_id is missing in the CLF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingOpIdPolicy {
    Fail,
    Skip,
}

/// Errors produced by the CLF reader.
#[derive(Debug, Error)]
pub enum ClfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic: expected CLF1")]
    InvalidMagic,
    #[error("unsupported version: {0} (supported: {1})")]
    UnsupportedVersion(u8, u8),
    #[error("invalid vendor or target: UTF-8 error")]
    InvalidVendorUtf8,
    #[error("signature missing or invalid")]
    SignatureInvalid,
    #[error("missing op_id {0} in CLF (policy: Fail)")]
    MissingOpId(u32),
    #[error("CLF kind mismatch: expected {expected:?}, got {actual:?}")]
    KindMismatch {
        expected: ClfKind,
        actual: ClfKind,
    },
}

/// CLF reader: parses header and manifest, provides get_blob(op_id).
#[derive(Debug)]
pub struct ClfReader {
    pub header: ClfHeader,
    manifest: HashMap<u32, ManifestEntry>,
    reader: BufReader<File>,
    blob_store_offset: u64,
    blob_store_len: u64,
}

impl ClfReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ClfError> {
        Self::open_with_expected_kind(path, None)
    }

    pub fn open_with_expected_kind<P: AsRef<Path>>(
        path: P,
        expected_kind: Option<ClfKind>,
    ) -> Result<Self, ClfError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != CLF_MAGIC {
            return Err(ClfError::InvalidMagic);
        }

        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        let version = version_byte[0];
        if version > CLF_VERSION {
            return Err(ClfError::UnsupportedVersion(version, CLF_VERSION));
        }

        let mut vendor_len_buf = [0u8; 4];
        reader.read_exact(&mut vendor_len_buf)?;
        let vendor_len = u32::from_le_bytes(vendor_len_buf) as usize;
        let mut vendor_bytes = vec![0u8; vendor_len];
        if vendor_len > 0 {
            reader.read_exact(&mut vendor_bytes)?;
        }
        let vendor = String::from_utf8(vendor_bytes).map_err(|_| ClfError::InvalidVendorUtf8)?;

        let mut target_len_buf = [0u8; 4];
        reader.read_exact(&mut target_len_buf)?;
        let target_len = u32::from_le_bytes(target_len_buf) as usize;
        let mut target_bytes = vec![0u8; target_len];
        if target_len > 0 {
            reader.read_exact(&mut target_bytes)?;
        }
        let target = String::from_utf8(target_bytes).map_err(|_| ClfError::InvalidVendorUtf8)?;
        let mut blob_align_byte = [0u8; 1];
        reader.read_exact(&mut blob_align_byte)?;
        let blob_alignment = blob_align_byte[0];

        let kind = if version >= 2 {
            let mut kind_byte = [0u8; 1];
            reader.read_exact(&mut kind_byte)?;
            ClfKind::from_byte(kind_byte[0])
        } else {
            ClfKind::default_for_v1()
        };

        let header_end = reader.stream_position()?;
        let header = ClfHeader {
            version,
            vendor,
            target,
            blob_alignment,
            kind,
            header_end,
        };

        if let Some(expected) = expected_kind {
            if header.kind != expected {
                return Err(ClfError::KindMismatch {
                    expected,
                    actual: header.kind,
                });
            }
        }

        let mut num_entries_buf = [0u8; 4];
        reader.read_exact(&mut num_entries_buf)?;
        let num_entries = u32::from_le_bytes(num_entries_buf) as usize;
        let mut manifest = HashMap::with_capacity(num_entries);
        for _ in 0..num_entries {
            let mut entry_buf = [0u8; ManifestEntry::ENTRY_SIZE];
            reader.read_exact(&mut entry_buf)?;
            let op_id = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
            let offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());
            let size = u32::from_le_bytes(entry_buf[8..12].try_into().unwrap());
            manifest.insert(op_id, ManifestEntry { op_id, offset, size });
        }

        let blob_store_offset = reader.stream_position()?;
        let file_len = reader.seek(SeekFrom::End(0))?;
        let has_sig = file_len >= (SIG_BLOCK_LEN as u64)
            && {
                reader.seek(SeekFrom::End(-(SIG_BLOCK_LEN as i64)))?;
                let mut sig_magic = [0u8; 4];
                reader.read_exact(&mut sig_magic).is_ok() && sig_magic == SIG_MAGIC
            };
        let blob_store_len = if has_sig {
            file_len
                .saturating_sub(SIG_BLOCK_LEN as u64)
                .saturating_sub(blob_store_offset)
        } else {
            file_len.saturating_sub(blob_store_offset)
        };

        reader.seek(SeekFrom::Start(blob_store_offset))?;

        Ok(ClfReader {
            header,
            manifest,
            reader,
            blob_store_offset,
            blob_store_len,
        })
    }

    pub fn get_blob(&mut self, op_id: u32) -> Result<Option<Vec<u8>>, ClfError> {
        let entry = match self.manifest.get(&op_id) {
            Some(e) => e,
            None => return Ok(None),
        };
        let start = self.blob_store_offset + u64::from(entry.offset);
        let end = start + u64::from(entry.size);
        if end > self.blob_store_offset + self.blob_store_len {
            return Err(ClfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "manifest entry extends past blob store",
            )));
        }
        self.reader.seek(SeekFrom::Start(start))?;
        let mut blob = vec![0u8; entry.size as usize];
        self.reader.read_exact(&mut blob)?;
        Ok(Some(blob))
    }

    pub fn op_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.manifest.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn build_code_section(
        &mut self,
        op_ids: &[u32],
        policy: MissingOpIdPolicy,
    ) -> Result<Vec<u8>, ClfError> {
        let mut out = Vec::new();
        for &op_id in op_ids {
            match self.get_blob(op_id)? {
                Some(blob) => out.extend_from_slice(&blob),
                None => {
                    if policy == MissingOpIdPolicy::Fail {
                        return Err(ClfError::MissingOpId(op_id));
                    }
                }
            }
        }
        Ok(out)
    }
}

/// CLF reader from bytes (e.g. container-embedded CLFMM/CLFMP).
#[derive(Debug)]
pub struct ClfReaderFromBytes {
    pub header: ClfHeader,
    manifest: HashMap<u32, ManifestEntry>,
    data: Vec<u8>,
    blob_store_offset: u64,
}

impl ClfReaderFromBytes {
    pub fn open(data: &[u8], expected_kind: Option<ClfKind>) -> Result<Self, ClfError> {
        let mut cursor = Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if magic != CLF_MAGIC {
            return Err(ClfError::InvalidMagic);
        }
        let mut version_byte = [0u8; 1];
        cursor.read_exact(&mut version_byte)?;
        let version = version_byte[0];
        if version > CLF_VERSION {
            return Err(ClfError::UnsupportedVersion(version, CLF_VERSION));
        }
        let mut vendor_len_buf = [0u8; 4];
        cursor.read_exact(&mut vendor_len_buf)?;
        let vendor_len = u32::from_le_bytes(vendor_len_buf) as usize;
        let mut vendor_bytes = vec![0u8; vendor_len];
        if vendor_len > 0 {
            cursor.read_exact(&mut vendor_bytes)?;
        }
        let vendor = String::from_utf8(vendor_bytes).map_err(|_| ClfError::InvalidVendorUtf8)?;
        let mut target_len_buf = [0u8; 4];
        cursor.read_exact(&mut target_len_buf)?;
        let target_len = u32::from_le_bytes(target_len_buf) as usize;
        let mut target_bytes = vec![0u8; target_len];
        if target_len > 0 {
            cursor.read_exact(&mut target_bytes)?;
        }
        let target = String::from_utf8(target_bytes).map_err(|_| ClfError::InvalidVendorUtf8)?;
        let mut blob_align_byte = [0u8; 1];
        cursor.read_exact(&mut blob_align_byte)?;
        let blob_alignment = blob_align_byte[0];
        let kind = if version >= 2 {
            let mut kind_byte = [0u8; 1];
            cursor.read_exact(&mut kind_byte)?;
            ClfKind::from_byte(kind_byte[0])
        } else {
            ClfKind::default_for_v1()
        };
        let header_end = cursor.stream_position()?;
        let header = ClfHeader {
            version,
            vendor,
            target,
            blob_alignment,
            kind,
            header_end,
        };
        if let Some(expected) = expected_kind {
            if header.kind != expected {
                return Err(ClfError::KindMismatch {
                    expected,
                    actual: header.kind,
                });
            }
        }
        let mut num_entries_buf = [0u8; 4];
        cursor.read_exact(&mut num_entries_buf)?;
        let num_entries = u32::from_le_bytes(num_entries_buf) as usize;
        let mut manifest = HashMap::with_capacity(num_entries);
        for _ in 0..num_entries {
            let mut entry_buf = [0u8; ManifestEntry::ENTRY_SIZE];
            cursor.read_exact(&mut entry_buf)?;
            let op_id = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap());
            let offset = u32::from_le_bytes(entry_buf[4..8].try_into().unwrap());
            let size = u32::from_le_bytes(entry_buf[8..12].try_into().unwrap());
            manifest.insert(op_id, ManifestEntry { op_id, offset, size });
        }
        let blob_store_offset = cursor.stream_position()?;
        Ok(Self {
            header,
            manifest,
            data: data.to_vec(),
            blob_store_offset,
        })
    }

    pub fn get_blob(&self, op_id: u32) -> Result<Option<Vec<u8>>, ClfError> {
        let entry = match self.manifest.get(&op_id) {
            Some(e) => e,
            None => return Ok(None),
        };
        let start = (self.blob_store_offset + u64::from(entry.offset)) as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(ClfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "manifest entry extends past blob store",
            )));
        }
        Ok(Some(self.data[start..end].to_vec()))
    }
}
