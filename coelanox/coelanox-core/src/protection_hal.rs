//! Protection HAL (Hardware Abstraction Layer) for COELANOX
//!
//! Defines the abstract protection contract: configure region permissions (R/W/X),
//! enable/disable hardware enforcement. Backends (ARM MPU, x86 page tables, RISC-V PMP,
//! ESP32, etc.) implement this trait. Callers use the registry and never depend on
//! concrete hardware. Independent of Memory HAL; used for AOT lock (code=X, weights=R, intermediates=RW).

use crate::memory_hal::MemoryRegion;
use crate::CoelanoxResult;
use std::collections::HashMap;

/// Protection backend target (registry key). Like MemoryTarget / HardwareTarget.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProtectionTarget {
    /// ARM MPU (Memory Protection Unit)
    ArmMpu,
    /// x86 page tables (MMU)
    X86PageTables,
    /// RISC-V PMP (Physical Memory Protection)
    RiscvPmp,
    /// ESP32 protection mechanism
    Esp32,
    /// Custom target (e.g. vendor-specific)
    Custom(String),
}

/// Permissions for a protection region. Used for AOT: code=X, weights=R, intermediates=RW.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionPermissions {
    /// Region is readable
    pub read: bool,
    /// Region is writable
    pub write: bool,
    /// Region is executable
    pub execute: bool,
}

impl RegionPermissions {
    /// Code / DAG: execute only (or RX in practice). Not writable after load.
    pub fn code() -> Self {
        Self {
            read: true,
            write: false,
            execute: true,
        }
    }

    /// Weights: read-only. Loaded once, never written by inference.
    pub fn weights() -> Self {
        Self {
            read: true,
            write: false,
            execute: false,
        }
    }

    /// Intermediates: read-write. Activations and temp buffers.
    pub fn intermediates() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
        }
    }
}

/// Capabilities reported by a protection backend.
#[derive(Debug, Clone)]
pub struct ProtectionCapabilities {
    /// Maximum number of regions that can be configured (0 = unknown)
    pub max_regions: usize,
    /// True if backend can enforce execute (X) permission
    pub supports_execute: bool,
    /// Minimum alignment for region base (bytes)
    pub min_alignment: usize,
    /// Backend name for logging
    pub name: String,
}

/// Protection HAL trait: abstract contract for protection backends.
/// No hardware-specific types; backends implement configure, enable, disable.
pub trait ProtectionHAL: Send + Sync {
    /// Configure a region with the given permissions. Call after memory is placed.
    /// Uses the same region (base/size) as provided by the Memory HAL.
    fn configure_region(
        &self,
        region: &dyn MemoryRegion,
        permissions: RegionPermissions,
    ) -> CoelanoxResult<()>;

    /// Enable hardware enforcement for all configured regions (lock phase).
    fn enable_protection(&self) -> CoelanoxResult<()>;

    /// Disable hardware enforcement (unlock phase). Call before releasing memory.
    fn disable_protection(&self) -> CoelanoxResult<()>;

    /// Report backend capabilities.
    fn get_capabilities(&self) -> ProtectionCapabilities;

    /// Backend name for logging and registry display.
    fn name(&self) -> &str;
}

/// Registry of protection backends keyed by ProtectionTarget.
#[derive(Default)]
pub struct ProtectionBackendRegistry {
    backends: HashMap<ProtectionTarget, Box<dyn ProtectionHAL>>,
}

impl ProtectionBackendRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register a backend for the given target. Replaces existing if present.
    pub fn register(&mut self, target: ProtectionTarget, backend: Box<dyn ProtectionHAL>) {
        self.backends.insert(target, backend);
    }

    /// Get a reference to the backend for the given target.
    pub fn get(&self, target: &ProtectionTarget) -> Option<&dyn ProtectionHAL> {
        self.backends.get(target).map(|b| b.as_ref())
    }

    /// List all registered targets.
    pub fn list_supported(&self) -> Vec<ProtectionTarget> {
        self.backends.keys().cloned().collect()
    }

    /// Check if a target is registered.
    pub fn is_supported(&self, target: &ProtectionTarget) -> bool {
        self.backends.contains_key(target)
    }

    /// Default protection backend. Prefer first available; used for lock/unlock when only one backend is registered.
    pub fn default_backend(&self) -> Option<(ProtectionTarget, &dyn ProtectionHAL)> {
        self.backends
            .iter()
            .next()
            .map(|(t, b)| (t.clone(), b.as_ref()))
    }

    /// Remove and return the backend for the given target.
    pub fn remove(&mut self, target: &ProtectionTarget) -> Option<Box<dyn ProtectionHAL>> {
        self.backends.remove(target)
    }
}
