# COELANOX SDK

**The universal SDK for AI model deployment across all platforms and hardware.**

## 🎯 Overview

The COELANOX SDK provides everything you need to build translators and backends for the COELANOX universal AI deployment platform.

## 📦 SDK Components

### **Core Libraries**
- `coelanox-core` - Universal IR, container format, utilities
- `coelanox-packager` - Model packaging and compression
- `coelanox-runtime` - Execution engine

### **Tools**
- CLI tools for packaging and execution
- Development utilities
- Testing frameworks

### **Examples**
- Sample translator implementations
- Sample backend implementations
- Complete workflow examples

### **Templates**
- Translator project templates
- Backend project templates
- Complete project starters

### **Documentation**
- API reference
- Getting started guides
- Architecture documentation
- Best practices

## 🚀 Quick Start

### **For Framework Creators (Build Translators)**
```bash
# Create a new translator
cargo generate --template translator-template

# Implement the TranslatorPlugin trait
# Map your format to Universal IR
# Test with sample models
```

### **For Hardware Vendors (Build Backends)**
```bash
# Create a new backend
cargo generate --template backend-template

# Implement the BackendPlugin trait
# Optimize for your hardware
# Benchmark performance
```

## 📚 Architecture

### **Frontend API (Framework Creators)**
Convert AI models to Universal IR - **Open Access**

```rust
pub trait TranslatorPlugin {
    fn translate(&self, data: &[u8]) -> Result<UniversalIR>;
    fn supported_formats(&self) -> Vec<&str>;
    fn validate(&self, data: &[u8]) -> Result<()>;
}
```

**See**: [Frontend API Docs](./docs/frontend-api.md)

### **Backend API (Hardware Partners)**
Partner-only API for verified hardware vendors

```rust
pub trait BackendTranslator {
    fn translate_operation(&self, op: &CoelanoxAiOperation) -> Result<String>;
    fn supported_hardware(&self) -> Vec<HardwareTarget>;
    fn optimize_code(&self, code: &str, target: HardwareTarget) -> Result<String>;
}
```

**Partner Access**: [Backend API Docs](./docs/backend-api.md)

*Backend API requires hardware vendor partnership and security approval.*

## Access Control

- **Frontend API**: Open to all developers
- **Backend API**: Partner-only (verified hardware vendors)

## 🎯 Standards

All translators and backends MUST follow the COELANOX standards:

1. **Universal IR** - Convert to/from Universal IR only
2. **Plugin Trait** - Implement required trait methods
3. **Error Handling** - Return proper error types
4. **Validation** - Validate all inputs/outputs
5. **Testing** - Include comprehensive tests
6. **Documentation** - Document all public APIs

## Documentation

- [Getting Started](./docs/getting-started.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Frontend API](./docs/frontend-api.md)
- [Backend API](./docs/backend-api.md) *(Partner-Only)*
- [Partner Onboarding](./docs/PARTNER_ONBOARDING.md) *(Hardware Vendors)*
- [Best Practices](./docs/best-practices.md)

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to contribute translators and backends.

## 📄 License

See [LICENSE](../LICENSE) for licensing information.

## 🎯 Vision

COELANOX SDK enables:
- **Framework creators** to add COELANOX support
- **Hardware vendors** to add hardware optimization
- **Developers** to deploy AI models universally

**Universal AI deployment starts here.**

