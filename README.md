# COELANOX (Secure AI Container)

<div align="center">
  <img src="assets/coelanox-coelanox_logo.png" alt="COELANOX Logo" width="300"/>
  <br/>
  <em>The Universal Binary Container for AI Models</em>
</div>

<div align="center">

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](https://github.com/coelanox/coelanox/blob/main/LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://rustup.rs/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](https://github.com/coelanox/coelanox#platform-support)
[![Security](https://img.shields.io/badge/Security-Hash%20%7C%20Validation-blue.svg)](https://github.com/coelanox/coelanox#security)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/coelanox/coelanox/actions)

[![GitHub stars](https://img.shields.io/github/stars/coelanox/coelanox?style=social)](https://github.com/coelanox/coelanox)
[![GitHub forks](https://img.shields.io/github/forks/coelanox/coelanox?style=social)](https://github.com/coelanox/coelanox)
[![GitHub issues](https://img.shields.io/github/issues/coelanox/coelanox)](https://github.com/coelanox/coelanox/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/coelanox/coelanox)](https://github.com/coelanox/coelanox/pulls)

</div>

---

## What is COELANOX?

**COELANOX** is a **universal binary container** for AI models. It lets you package any model (ONNX, TensorFlow, PyTorch, or custom binary) into a single `.cnox` file, verify its integrity, and run it across different backends with a consistent CLI.

### What It Does

- **Package** — Turn a source model into a COELANOX container: translate to a universal IR, optimize for target hardware, compress weights, and embed a security hash.
- **Validate** — Check a `.cnox` container’s integrity (SHA-256) before deployment or execution, without loading the full model.
- **Run** — Load and execute a container: verify hash, load into the runtime, and run inference with the chosen backend (CPU, GPU, or fallback).

The design is **format-agnostic** (the container doesn’t depend on a single framework), **security-first** (hash verification at package and run time), and **backend-pluggable** (OpenVINO, TensorRT, and a scalar fallback).

## Architecture Overview

### Universal Container Design

COELANOX uses a **universal container** approach: the container format is independent of any single framework. The runtime interprets the payload via pluggable backends.

```
┌─────────────────────────────────────────────────────────────┐
│                    COELANOX Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  📦 Universal Container (.cnox)                             │
│  ├── Header: magic, version, security hash                  │
│  ├── Manifest: JSON metadata (name, format, hints)          │
│  └── Payload: IR, machine code, compressed weights          │
├─────────────────────────────────────────────────────────────┤
│  🔧 Backend System                                           │
│  ├── OpenVINO Backend: Intel CPU (in development)           │
│  ├── TensorRT Backend: NVIDIA GPU (in development)          │
│  └── Fallback Backend: Scalar execution (available)         │
├─────────────────────────────────────────────────────────────┤
│  🔒 Security                                                 │
│  ├── SHA-256 hash computed at package time                  │
│  ├── Validate command: standalone integrity check           │
│  └── Run command: verify-then-execute (defense in depth)     │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
coelanox/
├── coelanox-core/       # Format, IR, security hash, scalar ops
├── coelanox-packager/   # Package: IR, optimize, compress, hash
├── coelanox-runtime/    # Backend manager, load & execute
├── coelanox-cli/        # CLI: package, validate, run
└── tests/               # Integration tests
```

## What Coelanox Provides

### Universal Container
- **Format-agnostic** — Designed to support ONNX, TensorFlow, PyTorch, and custom binary formats via a single container type.
- **Single file** — One `.cnox` file holds header, manifest, IR, machine code, and compressed weights.
- **Cross-platform** — Targets Linux, Windows, and macOS.

### Three-Command Flow
- **Package** — Load a source model, translate to universal IR, optimize for target hardware, compress weights, compute SHA-256, and write a `.cnox` file.
- **Validate** — Read a `.cnox` file and verify its integrity using the embedded hash (no execution).
- **Run** — Verify the container, load it into the runtime, and execute with the selected backend.

### Backend System
- **Pluggable backends** — OpenVINO (Intel CPU) and TensorRT (NVIDIA GPU) are in development; a scalar fallback backend is available today.
- **Backend manager** — Runtime selects and manages backends; execution can fall back if a preferred backend is unavailable.

### Security
- **Security hash** — SHA-256 of the container is stored in the header and checked at validate and run time.
- **Memory safety** — Implemented in Rust with ownership and bounds checking.
- **Path and input safety** — Path traversal and input validation to reduce file-system and parsing risks.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/coelanox/coelanox.git
cd coelanox

# Build all components
cargo build --release

# Install the CLI (optional)
cargo install --path coelanox-cli
```

### Create Your First COELANOX Container

```bash
# Package a model (e.g. ONNX) into a .cnox container
cargo run --release -- package --input model.onnx --output model.cnox --format onnx

# Validate the container (integrity check only)
cargo run --release -- validate --container model.cnox

# Run the container (verify, load, execute)
cargo run --release -- run --container model.cnox --backend cpu
```

### CLI Commands

| Command | Purpose |
|---------|---------|
| `coelanox package --input <path> --output <path>` | Create a `.cnox` container from a source model. |
| `coelanox validate --container <path>` | Verify container integrity (SHA-256). |
| `coelanox run --container <path> [--backend cpu\|gpu]` | Verify, load, and execute a container. |

Use `--format onnx` (or `pytorch`, `tensorflow`) with `package` and `--target cpu` (or `gpu`) for optimization target. See `coelanox --help` for full options.

## Platform Support

### **Operating Systems**

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ **Full Support** | Primary development platform |
| **Windows** | ✅ **Full Support** | Native Windows builds |
| **macOS** | ✅ **Full Support** | Universal binary support |

### **Hardware Support**

| Hardware | Backend | Status |
|----------|---------|--------|
| **Intel CPU** | OpenVINO | 🔄 In development |
| **NVIDIA GPU** | TensorRT | 🔄 In development |
| **ARM CPU** | Fallback | ✅ Available |
| **Edge / generic** | Fallback | ✅ Available |

### **Model Formats**

| Format | Status | Backend Support |
|--------|--------|-----------------|
| **ONNX** | ✅ **Supported** | OpenVINO, TensorRT, Fallback |
| **TensorFlow** | ✅ **Supported** | OpenVINO, Fallback |
| **PyTorch** | ✅ **Supported** | OpenVINO, Fallback |
| **Custom Binary** | ✅ **Supported** | Fallback (universal) |

## 🏗️ **Technical Architecture**

### **COELANOX File Format**

A `.cnox` file contains:

- **Header** — Magic number, version, and SHA-256 security hash.
- **Manifest** — JSON metadata (model name, format, optimization hints).
- **Payload** — Universal IR, optional machine code for target hardware, and compressed weights (e.g. Zstd/LZ4).

The security hash is computed over the full container at package time and verified by `validate` and before `run`.

### **Backend System**

The runtime (`coelanox-runtime`) manages pluggable backends. Each backend can load a container, execute it, and report capabilities. The backend manager selects an appropriate backend (e.g. OpenVINO for CPU, TensorRT for GPU) and falls back to the scalar backend when needed.

## 🔒 **Security Features**

### **Security Architecture**

COELANOX implements comprehensive security layers:

```
Security Layers:
├── File System Security
│   ├── Path Traversal Protection
│   ├── Symlink Attack Prevention
│   ├── File Permission Validation
│   └── Size Limit Enforcement
├── Memory Safety
│   ├── Rust Ownership System
│   ├── Bounds Checking
│   ├── Zero-Copy Operations
│   └── Safe Unsafe Blocks
└── Format Validation
    ├── Magic Number Check
    ├── Version Validation
    ├── Structure Verification
    └── Manifest Parsing
```

### **Security Validations**

| Security Check | Implementation | Protection |
|----------------|---------------|------------|
| **Path Traversal** | Blocks `..` patterns | Directory traversal attacks |
| **Symlink Attacks** | Prevents symlink access | Symbolic link exploits |
| **File Permissions** | Read-only detection | Unauthorized access |
| **File Type** | Regular file only | Device/socket mapping |
| **Size Limits** | 1GB file, 1MB manifest | DoS attacks |
| **JSON Validation** | UTF-8, size limits | Malicious manifests |

## 🛠️ **Development**

### **Building from Source**

```bash
# Prerequisites: Rust (rustup.rs)
# Clone and build
git clone https://github.com/coelanox/coelanox.git
cd coelanox
cargo build --release

# Run tests
cargo test
```

### **Project Structure**

```
coelanox/
├── coelanox-core/       # Format, IR, security hash, scalar ops
├── coelanox-packager/   # Package: IR, optimize, compress, hash
├── coelanox-runtime/    # Backend manager, load & execute
├── coelanox-cli/        # CLI: package, validate, run
├── tests/               # Integration tests
├── docs/                # Documentation
└── assets/              # Logos and images
```

### **Contributing**

Contributions are welcome. Open an issue or pull request to discuss changes.

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — System design, three-command flow, security hash, backend manager
- **[Security](docs/SECURITY.md)** — Security features and validation
- **[COELANOX Reference](COELANOX_REFERENCE.md)** — Quick reference
- **[Production MVP Roadmap](roadmaps/PRODUCTION_MVP_ROADMAP.md)** — Roadmap

## 🤝 **Community**

- **📧 [GitHub Issues](https://github.com/coelanox/coelanox/issues)** - Bug reports & feature requests
- **💬 [GitHub Discussions](https://github.com/coelanox/coelanox/discussions)** - Community discussions
- **📖 [Documentation](https://docs.coelanox.com/coelanox)** - Complete documentation
- **🐦 [Twitter](https://twitter.com/coelanox)** - Latest updates

## 📄 **License**

COELANOX is licensed under the [BSL-1.1 License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Intel OpenVINO** - For CPU optimization backends
- **NVIDIA TensorRT** - For GPU acceleration backends
- **Rust Community** - For the excellent language and ecosystem
- **Open Source Community** - For inspiration and collaboration

---

<div align="center">

**COELANOX**: The Universal Binary Container for AI Models

[![GitHub stars](https://img.shields.io/github/stars/coelanox/coelanox?style=social)](https://github.com/coelanox/coelanox)
[![GitHub forks](https://img.shields.io/github/forks/coelanox/coelanox?style=social)](https://github.com/coelanox/coelanox)

</div> 
