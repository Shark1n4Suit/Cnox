//! Authoritative Scalar Backend for COELANOX
//!
//! This module provides the single authoritative scalar implementation used for:
//! - Fallback backend execution (100% uptime guarantee)
//! - Vendor backend verification (reference implementation)
//! - Golden reference for correctness
//!
//! All scalar operations are pure Rust implementations with no SIMD dependencies.
//! This ensures portability and serves as a reliable reference for testing vendor backends.

pub mod activations;
pub mod arithmetic;
pub mod comparisons;
pub mod logical;
pub mod mapping;
pub mod math;
pub mod matrix_ops;
pub mod misc;
pub mod normalization;
pub mod pooling;
pub mod reductions;
pub mod tensor_ops;

// Re-export scalar operations
pub use activations::*;
pub use arithmetic::*;
pub use comparisons::*;
pub use logical::*;
pub use mapping::execute_scalar_op;
pub use math::*;
pub use matrix_ops::*;
pub use misc::*;
pub use normalization::*;
pub use pooling::*;
pub use reductions::*;
pub use tensor_ops::*;
