//! Linear algebra operations
//! MatMul, Gemm

pub mod matmul;
pub mod gemm;

// Re-export the generate functions
pub use matmul::generate_matmul;
pub use gemm::generate_gemm;

