//! Reduction operations
//! ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd

pub mod reducesum;
pub mod reducemean;
pub mod reducemax;
pub mod reducemin;
pub mod reduceprod;

// Re-export the generate functions
pub use reducesum::generate_reducesum;
pub use reducemean::generate_mean;
pub use reducemax::generate_reducemax;
pub use reducemin::generate_reducemin;
pub use reduceprod::generate_reduceprod;

