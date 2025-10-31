# COELANOX SDK Architecture

High-level architecture documentation for COELANOX SDK developers.

## 🎯 Overview

COELANOX SDK provides a standardized interface for building translators and backends that work with COELANOX's universal AI deployment platform.

## 🏗️ Core Principles

### **1. Opaque IR Design**
- **Universal IR is opaque** - internal structure is not exposed
- **Builder pattern** - use builder API to construct IR
- **Helper functions** - COELANOX provides translation utilities
- **Validation** - IR is automatically validated

### **2. Plugin Architecture**
- **Translators** - convert formats to Universal IR
- **Backends** - execute IR on hardware
- **Modular** - each plugin is independent
- **Standardized** - all plugins follow same interface

### **3. Hardware Abstraction**
- **Universal operations** - same operations work on all hardware
- **Hardware-specific optimization** - backends optimize for their hardware
- **Performance hints** - CSV-based optimization data
- **Native support** - future hardware can implement native COELANOX operations

## 📊 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   COELANOX Core (Closed Source)             │
├─────────────────────────────────────────────────────────────┤
│  • Universal IR (Opaque)                                    │
│  • Container Format                                          │
│  • Builder API                                               │
│  • Validation System                                         │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Uses Builder API
                            │
┌─────────────────────────────────────────────────────────────┐
│              Translators (SDK - Open Interface)              │
├─────────────────────────────────────────────────────────────┤
│  ONNX Translator │ TFLite │ PyTorch │ TensorFlow │ Custom   │
│  • Parse format                                             │
│  • Extract metadata                                          │
│  • Build IR via Builder API                                  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Receives Compiled IR
                            │
┌─────────────────────────────────────────────────────────────┐
│              Backends (SDK - Open Interface)                 │
├─────────────────────────────────────────────────────────────┤
│  CPU │ GPU │ NPU │ TPU │ Custom Hardware                    │
│  • Translate to hardware code                                │
│  • Optimize for hardware                                    │
│  • Execute on hardware                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### **1. Translation Phase**
1. User provides model in format X (ONNX, TFLite, etc.)
2. Translator parses format X
3. Translator uses Builder API to construct Universal IR
4. Universal IR is validated and stored in container

### **2. Execution Phase**
1. Container loads Universal IR
2. Backend receives compiled IR
3. Backend translates IR operations to hardware code
4. Hardware executes operations
5. Results returned to user

## API Surface

### Frontend API (Public)
- `TranslatorPlugin` trait
- `UniversalIRBuilder` 
- Format parsing utilities
- Weight extraction helpers
- **Access**: Open to all developers

### Backend API (Partner-Only)
- `BackendTranslator` trait
- `CoelanoxAiOperation` definitions
- Performance estimation APIs
- Hardware optimization interfaces
- **Access**: Verified hardware partners only

## 🔑 Key Design Decisions

### **1. Opaque IR**
**Why**: Protects intellectual property, prevents reverse engineering, ensures compatibility

**How**: IR is constructed via builder API, internal structure never exposed

### **2. Builder Pattern**
**Why**: Flexibility, validation, extensibility

**How**: `UniversalIRBuilder` provides safe construction, validates input

### **3. Plugin System**
**Why**: Modularity, extensibility, ecosystem growth

**How**: Standard trait interfaces, clear registration, validation

### **4. Hardware Abstraction**
**Why**: Universal operations, hardware-specific optimization

**How**: Standardized operations, CSV optimization data, native operation support

## 📚 Documentation Structure

1. **Getting Started** - basic setup and concepts
2. **API Reference** - complete trait documentation
3. **Examples** - working code samples
4. **Best Practices** - recommended patterns
5. **Architecture** - this document

## 🔐 Security & IP Protection

- **Closed-source core** - protects critical IP
- **Open SDK** - enables ecosystem growth
- **Standardized interfaces** - ensures compatibility
- **Validation** - prevents invalid IR

## 🚀 Future Expansion

- **Native operations** - hardware can implement COELANOX operations natively
- **More formats** - additional translators
- **More hardware** - additional backends
- **Performance optimization** - CSV-based tuning data

