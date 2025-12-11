//! Pooling operations
//! MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool

pub mod maxpool;
pub mod avgpool;
pub mod globalmaxpool;
pub mod globalavgpool;

// Re-export the generate functions
pub use maxpool::generate_maxpool;
pub use avgpool::generate_avgpool;
pub use globalmaxpool::generate_globalmaxpool;
pub use globalavgpool::generate_globalavgpool;

