//! Model translators: convert external formats to UniversalIR.

mod bert;
mod resnet_tiny_mnist;

pub use bert::BertTranslator;
pub use resnet_tiny_mnist::ResNetTinyMnistTranslator;
