# Coelanox SAIC
**Fast, Secure, Small** - The next-generation AI deployment standard

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/coelanox/saic/actions)
[![Version](https://img.shields.io/badge/version-v0.5.0--beta-orange.svg)](https://github.com/coelanox/saic/releases)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20|%20macOS%20|%20Windows-lightgrey.svg)](https://github.com/coelanox/saic)

## Introduction

Traditional AI deployment suffers from bloated containers, security vulnerabilities, and performance overhead. **SAIC (Secure AI Container)** solves these problems with a revolutionary approach: a binary container format specifically designed for AI models that prioritizes security, performance, and minimal footprint. SAIC enables **Fast** inference execution, **Secure** validated deployments, and **Small** optimized containers.

## 🚀 Key Features

- **⚡ High Performance**: Optimized backends (OpenVINO CPU, TensorRT GPU) with zero-overhead execution
- **📦 Minimal Overhead**: Binary format with ~50 bytes overhead vs. gigabytes in traditional containers  
- **🔒 Security-First Design**: Cryptographic validation, sandboxed execution, and tamper detection
- **🎯 Extensible Backend Architecture**: Support for CPU, GPU, and future accelerators (Apple Silicon, AMD, Intel Arc)
- **🛠️ Developer-Friendly**: Simple CLI tools with comprehensive logging and error handling
- **🏗️ Production Ready**: Rust-built for memory safety, performance, and reliability

## ⚡ Quick Start

Get started with SAIC in 3 simple steps:

```bash
# 1. Download the SAIC runner
curl -fsSL https://install.coelanox.com/saic | sh

# 2. Download a pre-packaged model
curl -O https://models.coelanox.com/examples/mobilenet.saic

# 3. Run inference
saic-runner execute mobilenet.saic --backend cpu
```

## 📋 Installation

### Linux & macOS
```bash
# Install via our official installer
curl -fsSL https://install.coelanox.com/saic | sh

# Or install manually from GitHub releases
wget https://github.com/coelanox/saic/releases/latest/download/saic-linux.tar.gz
tar -xzf saic-linux.tar.gz
sudo mv saic-runner saic-packager /usr/local/bin/
```

### Windows
```powershell
# Install via PowerShell
iwr -useb https://install.coelanox.com/saic.ps1 | iex

# Or download from releases
# Visit: https://github.com/coelanox/saic/releases/latest
```

### Build from Source
```bash
git clone https://github.com/coelanox/saic.git
cd saic
cargo build --release --features full-inference
```

## 📖 Basic Usage

### Creating SAIC Packages
```bash
# Package an ONNX model into SAIC format
saic-packager --model model.onnx --name "MyModel" --output model.saic

# Package with configuration file
saic-packager --config saic.config.yaml --output model.saic
```

### Running SAIC Models
```bash
# Validate a SAIC file
saic-runner run model.saic

# Execute with CPU backend (OpenVINO)
saic-runner execute model.saic --backend cpu

# Execute with GPU backend (TensorRT)
saic-runner execute model.saic --backend gpu

# List available backends
saic-runner --backends
```

### Example Configuration (saic.config.yaml)
```yaml
model:
  name: "ResNet50"
  version: "1.0"
  source: "./resnet50.onnx"
  
metadata:
  description: "Image classification model"
  author: "Your Organization"
  license: "MIT"
  
build:
  optimization_level: 3
  target_backends: ["cpu", "gpu"]
```

## 🏗️ Project Status

SAIC is currently in **Beta (v0.5)** and ready for community testing and feedback. The core functionality is stable, with active development focused on:

- 🔧 Additional backend support (AMD ROCm, Apple Core ML, Intel Arc)
- 📚 Extended documentation and tutorials
- 🧪 Comprehensive testing across diverse hardware
- 🎯 Performance optimization and benchmarking

**We welcome contributions, bug reports, and feature requests!**

## 📁 Repository Structure

```
saic/
├── packager/          # SAIC file creation tool
├── runner/           # SAIC file execution engine  
├── docs/            # Documentation and guides
├── reports/         # Performance analysis and benchmarks
├── examples/        # Sample models and configurations
└── tests/          # Integration and unit tests
```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- 📋 Reporting bugs and requesting features
- 🔧 Setting up the development environment  
- 🧪 Running tests and benchmarks
- 📝 Code style and review process

## 📊 Performance

SAIC delivers exceptional performance across diverse hardware:

| Backend | Hardware | Throughput | Memory | Latency |
|---------|----------|------------|--------|---------|
| OpenVINO CPU | Intel x64 | 1,200 inf/sec | 45MB | 0.8ms |
| TensorRT GPU | NVIDIA RTX | 8,500 inf/sec | 120MB | 0.12ms |

*Benchmarks: ResNet50 inference, batch size 1, mixed precision*

## 🔗 Links

- **Documentation**: [docs.coelanox.com/saic](https://docs.coelanox.com/saic)
- **Examples**: [github.com/coelanox/saic-examples](https://github.com/coelanox/saic-examples)
- **Discussions**: [GitHub Discussions](https://github.com/coelanox/saic/discussions)
- **Discord**: [Join our community](https://discord.gg/coelanox)

## 📄 License

This project is licensed under the **Business Source License 1.1** (BSL 1.1). See the [LICENSE](LICENSE) file for details.

The BSL allows free use for development, testing, and non-production purposes. Commercial production use requires a license agreement. After 4 years, the software automatically becomes available under the Apache 2.0 license.

---

<div align="center">

**[🌐 Website](https://coelanox.com)** • **[📚 Docs](https://docs.coelanox.com)** • **[💬 Community](https://discord.gg/coelanox)** • **[🐦 Twitter](https://twitter.com/coelanox)**

*Built with ❤️ by the Coelanox team*

</div> 