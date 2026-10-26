//! Model translators: convert external formats to UniversalIR.

mod bert;
mod onnx;
mod resnet_tiny_mnist;

pub use bert::BertTranslator;
pub use onnx::OnnxTranslator;
pub use resnet_tiny_mnist::ResNetTinyMnistTranslator;
