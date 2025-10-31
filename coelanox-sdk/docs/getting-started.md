# Getting Started with COELANOX SDK

This guide will help you get started building translators and backends for COELANOX.

## 📋 Prerequisites

- Rust 1.70+ installed
- COELANOX SDK cloned
- Understanding of your target format/hardware

## 🎯 Quick Start

### **Building a Translator**

A translator converts a specific AI model format to COELANOX's Universal IR.

#### **1. Create Your Translator**

```rust
use coelanox_core::plugins::TranslatorPlugin;
use coelanox_core::ir::UniversalIR;

pub struct MyFormatTranslator;

impl TranslatorPlugin for MyFormatTranslator {
    fn translate(&self, data: &[u8]) -> Result<UniversalIR> {
        // Parse your format
        // Convert to Universal IR
        // Return the IR
    }
    
    fn supported_formats(&self) -> Vec<&str> {
        vec!["myformat"]
    }
    
    fn validate(&self, data: &[u8]) -> Result<()> {
        // Validate your format
    }
}
```

#### **2. Test Your Translator**

```rust
#[test]
fn test_my_format_translation() {
    let translator = MyFormatTranslator;
    let model_data = include_bytes!("fixtures/model.myformat");
    
    let ir = translator.translate(model_data).unwrap();
    assert!(!ir.nodes.is_empty());
}
```

### **Building a Backend**

A backend executes Universal IR on specific hardware.

#### **1. Create Your Backend**

```rust
use coelanox_core::plugins::BackendPlugin;
use coelanox_core::ir::UniversalIR;
use coelanox_runtime::backends::ExecutionResult;

pub struct MyHardwareBackend;

impl BackendPlugin for MyHardwareBackend {
    fn execute(&self, ir: &UniversalIR) -> Result<ExecutionResult> {
        // Execute IR on your hardware
        // Return results
    }
    
    fn supported_hardware(&self) -> Vec<HardwareTarget> {
        vec![HardwareTarget::Custom("MyHardware")]
    }
    
    fn optimize(&self, ir: &mut UniversalIR) -> Result<()> {
        // Optimize IR for your hardware
    }
}
```

#### **2. Test Your Backend**

```rust
#[test]
fn test_my_hardware_backend() {
    let backend = MyHardwareBackend;
    let ir = create_sample_ir();
    
    let result = backend.execute(&ir).unwrap();
    assert!(result.success);
}
```

## 📚 Next Steps

- [Architecture Overview](./ARCHITECTURE.md)
- [Frontend API Reference](./frontend-api.md)
- [Backend API Reference](./backend-api.md)
- [Best Practices](./best-practices.md)

## 🤝 Examples

See the `examples/` directory for complete working implementations.

## ❓ Questions?

Open an issue or check the main [README](../README.md) for more information.

