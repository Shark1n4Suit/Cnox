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

**COELANOX** is a **universal binary container** for AI models. You package a source model into a single `.cnox` file (universal IR, optimized code, compressed weights, SHA-256 hash), verify its integrity with a standalone command, and run inference via the CLI. The runtime executes using a scalar fallback backend (pure Rust, from `coelanox-core`); vendor backends for packaging are discovered at package time via the BackendManager.

### What the CLI Does (from `coelanox-cli`)

- **package** — `--input`, `--output`, `--model-type` (cv, nlp, audio, tabular, custom), `--target` (cpu, gpu, edge, auto, or hardware-specific), `--optimization`, `--fat` / `--auto` / `--backend-add`. Requires a translator plugin for the detected format (e.g. ONNX); default plugins do not register translators (see `coelanox-packager` plugins).
- **validate** — `--file <path>`: read a `.cnox` file and check structure/manifest validity.
- **verify** — `--file <path>` [--verbose]: verify SHA-256 integrity (standalone, no execution).
- **run** — `--file`, `--output`, optional `--input` (JSON): verify hash, load container, run inference (scalar fallback), write output JSON.
- **info** — `--file`: print container manifest and metadata.
- **benchmark** — `--file`, `--iterations`: run repeated inference and report timing.
- **env** — print system environment (CPU, GPU, memory, SIMD, recommended target).
- **debug** — `--file`, optional `--input`/`--output`, `--gdb`, `--dump-registers`, `--trace-limit`: run with diagnostics and optional GDB.

## Architecture Overview

### Universal Container Design

The `.cnox` format is framework-agnostic: one file holds header, manifest, IR, machine code, and compressed weights. The packager translates source models (via translator plugins) to universal IR, optimizes for a target, compresses weights (Zstd/LZ4 when enabled), and writes a SHA-256 hash in the header. The runtime loads and executes using the scalar fallback (no vendor runtime backends in-tree).

```
┌─────────────────────────────────────────────────────────────┐
│                    COELANOX Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  📦 Container (.cnox) — coelanox-core/format.rs             │
│  ├── Header: 76 bytes (magic COEL, version, security_hash)  │
│  ├── Manifest: JSON (name, format, shapes, backend_metadata)│
│  └── Payload: IR, machine code, compressed weights          │
├─────────────────────────────────────────────────────────────┤
│  🔧 Package time (coelanox-packager)                         │
│  ├── Translator plugins → Universal IR (ONNX target; no      │
│  │   default translator registered in-tree)                 │
│  ├── BackendManager: discover vendor BackendTranslators     │
│  │   (.so / .clf) for code generation                        │
│  └── Optimize IR, generate code, compress, hash, write     │
├─────────────────────────────────────────────────────────────┤
│  ▶ Runtime (coelanox-runtime)                                │
│  └── Scalar fallback (coelanox-core scalar ops) — available  │
├─────────────────────────────────────────────────────────────┤
│  🔒 Security (coelanox-core format.rs, runtime path check)  │
│  ├── SHA-256 over full container at package; verify_security_hash() │
│  ├── verify command: standalone integrity check             │
│  └── run: verify then load then execute                     │
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
- **Single file** — One `.cnox` file holds a 76-byte header (magic `COEL`, version, security hash), JSON manifest (name, format, input/output shapes, backend metadata), and payload (IR, machine code, compressed weights). See `coelanox-core/src/format.rs`.
- **Format target** — Packager is built for ONNX (format detection and weight extraction); translator plugins are pluggable. No ONNX translator is registered by default (see `coelanox-packager/src/plugins/mod.rs`). TensorFlow/PyTorch detection was removed; focus is ONNX.
- **Cross-platform** — Targets Linux, Windows, and macOS.

### Commands (from `coelanox-cli` main.rs)
- **package** — Requires a translator plugin for the detected format. Translates to universal IR, optimizes for target (cpu/gpu/edge/auto or hardware-specific), compresses weights, computes SHA-256, writes `.cnox`.
- **validate** — Reads `.cnox` and checks structure/manifest validity (no hash check; use **verify** for that).
- **verify** — Standalone SHA-256 integrity check (`coelanox-core::format::verify_security_hash`).
- **run** — Verifies hash, loads container, runs inference with scalar fallback, writes output JSON.
- **info** / **benchmark** / **env** / **debug** — Inspect container, benchmark, show environment, or run with diagnostics.

### Backend System
- **Package time** — `coelanox-runtime/backend_manager.rs`: discovers vendor BackendTranslators (dynamic libs or CLF) for code generation. No vendor translators are shipped in-tree.
- **Runtime** — Execution uses the scalar fallback in `coelanox-runtime/fallback` (delegates to `coelanox-core::scalar::execute_scalar_op`). No OpenVINO/TensorRT runtime in the repo.

### Security
- **Security hash** — SHA-256 over full container; stored in header; verified by `verify` and before `run` (see `coelanox-core/src/format.rs` and `docs/SECURITY.md`).
- **Path validation** — `coelanox_core::utils::validate_path` and runtime `validate_container_path` (path traversal protection).
- **Memory safety** — Rust; bounds checking; safe usage of unsafe where needed.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/coelanox/coelanox.git
cd coelanox

# Build all components (from repo root; coelanox/ is the workspace)
cargo build --release

# Install the CLI (optional; run from workspace root, e.g. coelanox/)
cargo install --path coelanox-cli
```

### CLI Commands (from `coelanox-cli` main.rs)

| Command | Flags | Purpose |
|---------|-------|---------|
| **package** | `--input`, `--output`, `--model-type` (cv, nlp, audio, tabular, custom), `--target` (cpu, gpu, edge, auto, or h100, a100, amd-gpu, intel-npu, etc.), `--optimization`, `--fat` / `--auto` / `--backend-add` | Create a `.cnox` from a source model. Requires a translator plugin for the detected format; default plugins do not register any translator. |
| **validate** | `--file <path>` | Read `.cnox` and check structure/manifest validity. |
| **verify** | `--file <path>` [--verbose] | Standalone SHA-256 integrity check. |
| **run** | `--file <path>`, `--output <path>`, optional `--input` (JSON), `--backend` (default cpu) | Verify hash, load container, run inference, write output JSON. |
| **info** | `--file <path>` | Print container manifest and metadata. |
| **benchmark** | `--file <path>`, `--iterations` (default 100) | Run repeated inference and report timing. |
| **env** | (none) | Print system environment and recommended target. |
| **debug** | `--file`, optional `--input`/`--output`, `--gdb`, `--dump-registers`, `--trace-limit` | Run with diagnostics or GDB. |

Run from the workspace root: `cargo run --release -- <command> ...` or after `cargo install`: `coelanox <command> ...`. Use `coelanox --help` and `coelanox <command> --help` for full options.

## Platform Support

- **Operating systems** — Linux, Windows, macOS (Rust targets).
- **Runtime execution** — Scalar fallback only (in-tree). No OpenVINO/TensorRT runtime; vendor BackendTranslators are for package-time code generation and are discovered by BackendManager (`.so` / `.clf`), not shipped in-tree.
- **Model format** — Packager targets ONNX (format detection and weight extraction in `coelanox-packager`). Translator plugins are pluggable; **no ONNX translator is registered by default** (`coelanox-packager/src/plugins/mod.rs`). TensorFlow/PyTorch detection was removed in the codebase.

## Technical Architecture

### COELANOX File Format (`coelanox-core/src/format.rs`)

- **Header** — 76 bytes: magic `COEL` (4), version (4), flags (4), container_size (8), manifest_offset (8), code_offset (8), security_hash (32), reserved (8). Hash is computed over full container in `write_binary_file` and verified by `verify_security_hash(path)`.
- **Manifest** — JSON: name, format, input_shapes, output_shapes, model_size, timestamps, backend_metadata, etc. Written with length prefix.
- **Payload** — Machine code, compressed weights (Zstd/LZ4 when packaging feature enabled), IR data.

### Backend System

- **Package time** — `coelanox-runtime/backend_manager.rs`: discovers BackendTranslators (dynamic libs or CLF) for code generation. No in-tree vendor implementations.
- **Runtime** — `coelanox-runtime` loads containers and executes via scalar fallback (`fallback/mod.rs` → `coelanox_core::scalar::execute_scalar_op`). No other execution backends in the repo.

## Security

- **Integrity** — SHA-256 over full container at package time; verified by `verify` and before `run` (`coelanox-core/src/format.rs`, `verify_security_hash()`). See `docs/SECURITY.md`.
- **Path validation** — `coelanox_core::utils::validate_path` (path traversal); runtime `validate_container_path` when loading containers (`coelanox-runtime/src/runtime.rs`).
- **Format** — Magic `COEL`, version check, manifest parsing in `format.rs`.
- **Memory safety** — Rust; no unchecked unsafe for user-facing data paths.

## Development

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
