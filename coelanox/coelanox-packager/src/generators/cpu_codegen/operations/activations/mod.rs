//! Activation operations
//! Element-wise activation functions for neural networks

pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod gelu;
pub mod swish;
pub mod softmax;
pub mod logsoftmax;
pub mod sqrt;
pub mod pow;
pub mod cos;
pub mod sin;
pub mod exp;
pub mod log;

pub use relu::generate_relu;
pub use sigmoid::generate_sigmoid;
pub use tanh::generate_tanh;
pub use gelu::generate_gelu;
pub use swish::generate_swish;
pub use softmax::generate_softmax;
pub use logsoftmax::generate_logsoftmax;
pub use sqrt::generate_sqrt;
pub use pow::generate_pow;
pub use cos::generate_cos;
pub use sin::generate_sin;
pub use exp::generate_exp;
pub use log::generate_log;

