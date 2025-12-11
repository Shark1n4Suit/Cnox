//! Logical operations
//! And, Or, Not

pub mod and;
pub mod or;
pub mod not;

// Re-export the generate functions
pub use and::generate_and;
pub use or::generate_or;
pub use not::generate_not;

