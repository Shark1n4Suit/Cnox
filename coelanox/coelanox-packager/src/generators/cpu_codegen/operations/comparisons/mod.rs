//! Comparison operations
//! Element-wise comparison operations: Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual

mod helpers;
pub mod equal;
pub mod notequal;
pub mod greater;
pub mod greaterequal;
pub mod less;
pub mod lessequal;

pub use equal::generate_equal;
pub use notequal::generate_notequal;
pub use greater::generate_greater;
pub use greaterequal::generate_greaterequal;
pub use less::generate_less;
pub use lessequal::generate_lessequal;

