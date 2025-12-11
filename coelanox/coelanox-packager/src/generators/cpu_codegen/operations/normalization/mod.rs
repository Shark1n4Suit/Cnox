//! Normalization operations
//! BatchNorm, LayerNorm

pub mod batchnorm;
pub mod layernorm;

// Re-export the generate functions
pub use batchnorm::generate_batchnorm;
pub use layernorm::generate_layernorm;

