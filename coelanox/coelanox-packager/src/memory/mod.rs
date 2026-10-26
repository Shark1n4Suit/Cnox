//! Memory management utilities for COELANOX
//!
//! This module provides memory-related utilities including NVMe pre-allocation,
//! memory tiering, and buffer management.

pub mod nvme_allocator;

pub use nvme_allocator::*;
