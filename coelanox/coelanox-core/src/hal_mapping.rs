//! HAL mapping: region kinds to Memory HAL and Protection HAL.
//!
//! Connects OpType/CLF (code) and container regions (code, weights, intermediates)
//! to Memory HAL (allocation/backing) and Protection HAL (permissions).
//! See docs/ARCHITECTURE.md "HAL mapping" section.

use crate::protection_hal::RegionPermissions;

/// Region kind for AOT lock: code (X), weights (R), intermediates (RW).
/// Maps to Memory HAL (which allocates/backs) and Protection HAL (which sets permissions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Machine code from CLF or translator. Memory HAL: executable allocation. Protection: code (RX).
    Code,
    /// Static weights. Memory HAL: backing / pre-alloc layout. Protection: read-only (R).
    Weights,
    /// Activations and temp buffers. Memory HAL: backing / pre-alloc layout. Protection: read-write (RW).
    Intermediates,
}

impl RegionKind {
    /// Protection HAL permissions for this region kind (code=X, weights=R, intermediates=RW).
    #[must_use]
    pub fn to_permissions(self) -> RegionPermissions {
        match self {
            RegionKind::Code => RegionPermissions::code(),
            RegionKind::Weights => RegionPermissions::weights(),
            RegionKind::Intermediates => RegionPermissions::intermediates(),
        }
    }

    /// Whether Memory HAL should allocate this region as executable (only Code).
    #[must_use]
    pub fn executable(self) -> bool {
        matches!(self, RegionKind::Code)
    }
}
