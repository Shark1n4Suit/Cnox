# Frontend API Reference

API for building translators that convert AI model formats to COELANOX Universal IR.

## Overview

The Frontend API enables framework creators to build translators for their AI model formats. Translators convert models to COELANOX's Universal IR for deployment across all platforms and hardware.

## TranslatorPlugin Trait

```rust
use coelanox_core::plugins::TranslatorPlugin;

pub trait TranslatorPlugin: Send + Sync + 'static {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>;
    fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR>;
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>;
    fn supported_formats(&self) -> Vec<ModelFormat>;
    
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn author(&self) -> &str;
    fn description(&self) -> &str;
    
    fn configure(&mut self, config: &PluginConfig) -> CoelanoxResult<()>;
    fn metadata(&self) -> PluginMetadata;
    fn capabilities(&self) -> PluginCapabilities;
}
```

## Required Methods

### translate()

Converts model data to Universal IR using the Builder API.

```rust
fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
    let model = parse_format(model_data)?;
    let mut builder = UniversalIRBuilder::new();
    
    builder.set_model_name(&model.name);
    builder.set_description(&model.description);
    // ... add operations and tensors ...
    
    builder.build()
}
```

### translate_from_path()

Reads model from file and translates.

```rust
fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR> {
    let data = std::fs::read(input_path)?;
    self.translate(&data)
}
```

### extract_weights()

Extracts weight tensors from model format.

```rust
fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
    let model = parse_format(model_data)?;
    Ok(extract_all_weights(&model))
}
```

### supported_formats()

Returns supported format identifiers.

```rust
fn supported_formats(&self) -> Vec<ModelFormat> {
    vec![ModelFormat::ONNX, ModelFormat::TensorFlowLite]
}
```

## Universal IR Construction

Use `UniversalIRBuilder` to construct IR. Do not construct IR directly.

```rust
let mut builder = UniversalIRBuilder::new();
builder.set_model_name("my_model");
builder.set_input_shapes(&input_shapes);
// ... add operations via builder methods ...
let ir = builder.build()?;
```

## Configuration

```rust
#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub verbose: bool,
    pub max_model_size: Option<usize>,
    pub optimization_settings: HashMap<String, String>,
    pub experimental_features: bool,
}
```

## Metadata

```rust
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub max_model_size: Option<usize>,
    pub supported_data_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PluginCapabilities {
    pub supports_sharded_models: bool,
    pub supports_custom_shapes: bool,
    pub supports_weight_extraction: bool,
    pub supports_model_validation: bool,
}
```

## Examples

See `examples/` directory for complete translator implementations.

## Reference

- [Getting Started Guide](./getting-started.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./best-practices.md)
