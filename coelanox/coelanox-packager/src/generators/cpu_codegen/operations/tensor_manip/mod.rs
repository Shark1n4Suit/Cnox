//! Tensor manipulation operations
//! Concatenate, Split, Gather, Scatter

pub mod concatenate;
pub mod split;
pub mod gather;
pub mod scatter;

// Re-export the generate functions
pub use concatenate::generate_concatenate;
pub use split::generate_split;
pub use gather::generate_gather;
pub use scatter::generate_scatter;

