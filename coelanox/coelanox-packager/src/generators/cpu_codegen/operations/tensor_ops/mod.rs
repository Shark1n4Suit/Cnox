//! Tensor shape and structure operations
//! Reshape, Transpose, Permute, Slice

pub mod reshape;
pub mod transpose;
pub mod permute;
pub mod slice;

// Re-export the generate functions
pub use reshape::generate_reshape;
pub use transpose::generate_transpose;
pub use permute::generate_permute;
pub use slice::generate_slice;

