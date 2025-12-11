//! Miscellaneous operations
//! Dropout, Broadcast, Expand, Custom

pub mod dropout;
pub mod broadcast;
pub mod expand;
pub mod custom;

// Re-export the generate functions
pub use dropout::generate_dropout;
pub use broadcast::generate_broadcast;
pub use expand::generate_expand;
pub use custom::generate_custom;

