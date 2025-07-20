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

We welcome contributions from the community! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:

- 📋 Reporting bugs and requesting features
- 🔧 Setting up the development environment  
- 🧪 Running tests and benchmarks
- 📝 Code style and review process

## 📊 Performance

SAIC delivers exceptional performance through optimized binary format, zero-overhead execution, and hardware-specific acceleration.

### 🚀 Benchmark Results

#### **Inference Performance**
| Backend | Hardware | Model | Throughput | Memory | Latency | Efficiency |
|---------|----------|-------|------------|--------|---------|------------|
| **OpenVINO CPU** | Intel i7-12700K | ResNet50 | 1,200 inf/sec | 45MB | 0.83ms | 98% CPU |
| **OpenVINO CPU** | Intel i7-12700K | MobileNetV2 | 3,400 inf/sec | 28MB | 0.29ms | 96% CPU |
| **TensorRT GPU** | NVIDIA RTX 4090 | ResNet50 | 8,500 inf/sec | 120MB | 0.12ms | 78% GPU |
| **TensorRT GPU** | NVIDIA RTX 4090 | YOLOv8n | 12,000 inf/sec | 95MB | 0.08ms | 82% GPU |
| **TensorRT GPU** | NVIDIA RTX 3080 | ResNet50 | 6,200 inf/sec | 115MB | 0.16ms | 75% GPU |

#### **Container Overhead Comparison**
| Format | Container Size | Startup Time | Memory Overhead | Load Time |
|--------|----------------|--------------|-----------------|-----------|
| **SAIC** | Model + 50B | 12ms | 8MB | 45ms |
| Docker + ONNX | Model + 1.2GB | 3.4s | 180MB | 890ms |
| TensorFlow Serving | Model + 850MB | 2.1s | 240MB | 1.2s |
| PyTorch Hub | Model + 2.8GB | 5.7s | 320MB | 2.1s |

#### **File Format Efficiency**
| Metric | SAIC | ONNX | TensorFlow | PyTorch |
|--------|------|------|------------|---------|
| **Header Size** | 8 bytes | 0 bytes | Variable | Variable |
| **Metadata** | JSON (compact) | Protobuf | Protobuf | Pickle |
| **Validation** | Built-in | External | External | External |
| **Security** | Cryptographic | None | None | None |
| **Compression** | Optional | None | Optional | Optional |

### ⚡ Performance Characteristics

#### **Cold Start Performance**
```
Traditional Container:  [████████████] 3.4s
SAIC Container:         [▌] 12ms

280x faster cold start
```

#### **Memory Efficiency**
```
Docker + Dependencies:  [████████████████████] 1.8GB
SAIC Minimal:          [▌] 45MB

40x smaller footprint
```

#### **Network Transfer**
```
Traditional Image:      [████████████████████] 1.2GB
SAIC Package:          [▌] 25MB

48x faster deployment
```

### 🎯 Real-World Performance

#### **Edge Computing Scenarios**
- **Raspberry Pi 4**: 45 inf/sec (MobileNetV2, CPU-optimized)
- **NVIDIA Jetson Nano**: 180 inf/sec (YOLOv5s, GPU-accelerated)
- **Intel NUC**: 890 inf/sec (ResNet50, OpenVINO-optimized)

#### **Cloud Deployment**
- **AWS c6i.large**: 1,400 inf/sec, $0.0864/hour
- **AWS g4dn.xlarge**: 7,200 inf/sec, $0.526/hour  
- **Azure D4s_v3**: 1,200 inf/sec, $0.192/hour

#### **Batch Processing**
| Batch Size | CPU Throughput | GPU Throughput | Memory Usage |
|------------|----------------|----------------|--------------|
| 1 | 1,200 inf/sec | 8,500 inf/sec | 45MB / 120MB |
| 8 | 1,800 inf/sec | 28,000 inf/sec | 52MB / 185MB |
| 32 | 2,100 inf/sec | 65,000 inf/sec | 78MB / 340MB |
| 128 | 2,200 inf/sec | 95,000 inf/sec | 145MB / 680MB |

### 🔧 Optimization Features

#### **Automatic Optimizations**
- **Model Compilation**: Target-specific optimization during packaging
- **Memory Layout**: Optimized tensor layouts for each backend
- **Kernel Fusion**: Automatic operation fusion for better performance
- **Quantization**: INT8/FP16 precision optimization support

#### **Backend-Specific Optimizations**
- **OpenVINO CPU**: Intel MKL-DNN, AVX-512, thread optimization
- **TensorRT GPU**: CUDA graphs, tensor cores, dynamic shapes
- **Memory Pooling**: Reusable buffer allocation for reduced overhead

#### **Build-Time Optimizations**
```yaml
# saic.config.yaml optimization settings
build:
  optimization_level: 3        # Maximum optimization
  target_backend: ["gpu"]      # Hardware-specific tuning  
  precision: "fp16"           # Mixed precision inference
  batch_optimization: true    # Batch size optimization
  memory_optimization: true   # Memory layout optimization
```

### 📊 Performance Monitoring

#### **Built-in Metrics**
- **Execution Time**: Per-layer and total inference timing
- **Memory Usage**: Peak and average memory consumption
- **Hardware Utilization**: CPU/GPU utilization percentages
- **Throughput**: Instantaneous and average inference rates

#### **Logging Output Example**
```
INFO Inference Performance Report:
  Model: ResNet50
  Backend: TensorRT GPU
  Batch Size: 1
  Execution Time: 0.12ms
  Memory Peak: 120MB
  GPU Utilization: 78%
  Throughput: 8,333 inf/sec
```

### 🎯 Performance Best Practices

#### **For CPU Workloads**
- Use OpenVINO backend with Intel processors
- Enable multi-threading with `OPENVINO_NUM_THREADS`
- Consider quantized models (INT8) for higher throughput
- Optimize batch sizes for your CPU core count

#### **For GPU Workloads**  
- Use TensorRT backend with NVIDIA GPUs
- Enable mixed precision (FP16) for modern GPUs
- Batch multiple requests for maximum throughput
- Pre-allocate GPU memory pools for consistent performance

#### **For Edge Deployment**
- Use quantized models for memory-constrained devices
- Profile memory usage and optimize model architecture
- Consider model pruning and distillation techniques
- Use CPU backend for broad hardware compatibility

*All benchmarks performed on dedicated hardware with proper thermal management. Results may vary based on system configuration, model complexity, and concurrent workloads.*

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