//! Arithmetic operations
//! Add, Subtract, Multiply, Divide

pub mod add;
pub mod subtract;
pub mod multiply;
pub mod divide;

// Re-export the generate functions
pub use add::generate_add;
pub use subtract::generate_subtract;
pub use multiply::generate_multiply;
pub use divide::generate_divide;

