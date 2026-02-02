//! Memory HAL (Hardware Abstraction Layer) for COELANOX
//!
//! Defines the abstract memory contract: allocate/deallocate regions, move memory,
//! create backing storage, and capabilities. Backends (NVMe, SD, OS, etc.) implement
//! this trait. Callers use the registry to obtain a backend and never depend on
//! concrete hardware. Mirrors the OpType + BackendTranslator pattern.

use crate::optimization::MaxShapeMetadata;
use crate::CoelanoxResult;
use std::collections::HashMap;

/// Memory backend target (registry key). Like HardwareTarget for codegen.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryTarget {
    /// NVMe-backed storage (pre-allocated backing, DMA-capable)
    Nvme,
    /// SD card storage
    Sd,
    /// OS-provided memory (mmap, VirtualAlloc, heap)
    Os,
    /// Custom target (e.g. vendor-specific)
    Custom(String),
}

/// Memory tier for tiering (fast vs slow). Backends report supported tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Fast tier (e.g. DRAM, L1)
    Fast,
    /// Slow tier (e.g. NVMe, SD)
    Slow,
}

/// Options for allocating a region. Used by allocate_region.
#[derive(Debug, Clone)]
pub struct RegionOptions {
    /// Requested size in bytes
    pub size: usize,
    /// Alignment in bytes (e.g. 4096 for page)
    pub alignment: usize,
    /// Preferred tier (backend may ignore if not supported)
    pub tier: MemoryTier,
    /// If true, region must be executable (e.g. for JIT code)
    pub executable: bool,
}

/// Capabilities reported by a memory backend. Used for tiering and feature detection.
#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    /// Tiers this backend supports
    pub supported_tiers: Vec<MemoryTier>,
    /// True if backend can allocate executable regions
    pub supports_executable: bool,
    /// True if backend can create backing (create_backing) for container packaging
    pub supports_create_backing: bool,
    /// Maximum single region size in bytes (0 = no limit or unknown)
    pub max_region_size: usize,
    /// Backend name for logging
    pub name: String,
}

/// Opaque handle to an allocated region. Backends return types implementing this trait.
/// Callers use as_ptr/size for copy or FFI; deallocate via MemoryHAL::deallocate_region.
pub trait MemoryRegion: Send + Sync {
    /// Pointer to the start of the region. Valid until deallocate_region is called.
    fn as_ptr(&self) -> *mut u8;
    /// Size in bytes of the region.
    fn size(&self) -> usize;
}

/// Memory HAL trait: abstract contract for memory backends.
/// No hardware-specific types; backends implement allocate, deallocate, move, create_backing.
pub trait MemoryHAL: Send + Sync {
    /// Allocate a region with the given options. Returns a boxed MemoryRegion for cleanup.
    fn allocate_region(
        &self,
        size: usize,
        alignment: usize,
        options: &RegionOptions,
    ) -> CoelanoxResult<Box<dyn MemoryRegion>>;

    /// Deallocate a region previously returned by allocate_region. Same backend must be used.
    fn deallocate_region(&self, region: Box<dyn MemoryRegion>) -> CoelanoxResult<()>;

    /// Copy bytes from src to dst. Both regions must be valid and size must not exceed either.
    fn move_memory(
        &self,
        src: &dyn MemoryRegion,
        dst: &dyn MemoryRegion,
        size: usize,
    ) -> CoelanoxResult<()>;

    /// Create backing data for container packaging (e.g. NVMe backing). Same semantics as
    /// NvmeAllocator::create_nvme_backing: zero-initialized buffer sized from metadata.
    fn create_backing(&self, metadata: &MaxShapeMetadata) -> CoelanoxResult<Vec<u8>>;

    /// Report backend capabilities (tiers, executable, create_backing, max size).
    fn get_capabilities(&self) -> MemoryCapabilities;

    /// Backend name for logging and registry display.
    fn name(&self) -> &str;
}

/// Registry of memory backends keyed by MemoryTarget. Callers get a backend and use the trait only.
#[derive(Default)]
pub struct MemoryBackendRegistry {
    backends: HashMap<MemoryTarget, Box<dyn MemoryHAL>>,
}

impl MemoryBackendRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register a backend for the given target. Replaces existing if present.
    pub fn register(&mut self, target: MemoryTarget, backend: Box<dyn MemoryHAL>) {
        self.backends.insert(target, backend);
    }

    /// Get a reference to the backend for the given target.
    pub fn get(&self, target: &MemoryTarget) -> Option<&dyn MemoryHAL> {
        self.backends.get(target).map(|b| b.as_ref())
    }

    /// Get a backend that supports the given tier. Returns first match.
    pub fn get_for_tier(&self, tier: MemoryTier) -> Option<(&MemoryTarget, &dyn MemoryHAL)> {
        self.backends.iter().find_map(|(t, b)| {
            if b.get_capabilities().supported_tiers.contains(&tier) {
                Some((t, b.as_ref()))
            } else {
                None
            }
        })
    }

    /// List all registered targets.
    pub fn list_supported(&self) -> Vec<MemoryTarget> {
        self.backends.keys().cloned().collect()
    }

    /// Check if a target is registered.
    pub fn is_supported(&self, target: &MemoryTarget) -> bool {
        self.backends.contains_key(target)
    }

    /// Default for storage (create_backing). Prefer Nvme, then Os, then first available.
    /// Returns owned target so caller can store it (e.g. for cleanup).
    pub fn default_for_storage(&self) -> Option<(MemoryTarget, &dyn MemoryHAL)> {
        if let Some(b) = self.get(&MemoryTarget::Nvme) {
            return Some((MemoryTarget::Nvme, b));
        }
        if let Some(b) = self.get(&MemoryTarget::Os) {
            return Some((MemoryTarget::Os, b));
        }
        if let Some(b) = self.get(&MemoryTarget::Sd) {
            return Some((MemoryTarget::Sd, b));
        }
        self.backends
            .iter()
            .next()
            .map(|(t, b)| (t.clone(), b.as_ref()))
    }

    /// Default for executable allocation. Prefer Os (mmap/VirtualAlloc), then first that supports executable.
    /// Returns owned target so caller can store it for cleanup.
    pub fn default_for_executable(&self) -> Option<(MemoryTarget, &dyn MemoryHAL)> {
        if let Some(b) = self.get(&MemoryTarget::Os) {
            if b.get_capabilities().supports_executable {
                return Some((MemoryTarget::Os, b));
            }
        }
        self.backends.iter().find_map(|(t, b)| {
            if b.get_capabilities().supports_executable {
                Some((t.clone(), b.as_ref()))
            } else {
                None
            }
        })
    }

    /// Remove and return the backend for the given target.
    pub fn remove(&mut self, target: &MemoryTarget) -> Option<Box<dyn MemoryHAL>> {
        self.backends.remove(target)
    }
}
