//! Authoritative Scalar Backend for COELANOX
//! 
//! This module provides the single authoritative scalar implementation used for:
//! - Fallback backend execution (100% uptime guarantee)
//! - Vendor backend verification (reference implementation)
//! - Golden reference for correctness
//! 
//! All scalar operations are pure Rust implementations with no SIMD dependencies.
//! This ensures portability and serves as a reliable reference for testing vendor backends.

pub mod arithmetic;
pub mod activations;
pub mod math;
pub mod pooling;
pub mod normalization;
pub mod reductions;
pub mod matrix_ops;
pub mod tensor_ops;
pub mod comparisons;
pub mod logical;
pub mod misc;
pub mod mapping;

// Re-export commonly used operations
// Note: Operations are placeholders and will be implemented in Task 2.2
pub use arithmetic::*;
pub use activations::*;
pub use math::*;
pub use pooling::*;
pub use normalization::*;
pub use reductions::*;
pub use matrix_ops::*;
pub use tensor_ops::*;
pub use comparisons::*;
pub use logical::*;
pub use misc::*;
pub use mapping::execute_scalar_op;

