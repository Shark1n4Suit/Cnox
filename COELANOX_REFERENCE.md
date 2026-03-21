# COELANOX & .cnox Reference Document

**Quick reference: what COELANOX is and what `.cnox` files contain.**

---

## What is COELANOX?

**COELANOX** (Secure AI Container) is a universal binary container system for AI models:

1. **Universal Format**: Convert models (ONNX, TensorFlow, PyTorch) into a single `.cnox` container
2. **AOT Compilation**: Converts to optimized machine code for target hardware
3. **Universal IR**: Graph-based intermediate representation abstracting model formats
4. **Hardware Abstraction**: Backend translators map Coelanox ops to hardware-specific code
5. **"De Beers of AI" Strategy**: Standardized operations that hardware vendors implement

### Architecture (summary)

```
Input Model → Translator → Universal IR → Optimizer → Code Generator → .cnox → Runtime Execution
```

**Components:** `coelanox-core` (IR, format, scalar ops), `coelanox-packager` (translate, optimize, compress), `coelanox-orchestrator` (load, execute, backends), `coelanox-cli` (commands). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## What is a `.cnox` File?

A **`.cnox` file** contains:

1. **Header** (76 bytes): Magic `COEL`, version, flags, sizes, SHA-256 security hash
2. **Manifest** (JSON): Model name, input/output shapes, backend metadata
3. **Machine Code**: Pre-compiled for target (CLF blobs or scalar path)
4. **Compressed Weights**: Zstd/LZ4
5. **IR Data**: Serialized OptimizedIR (bincode)
6. **Optional**: NVMe backing, CLFMM, CLFMP

See [docs/CNOX_INTERNALS.md](docs/CNOX_INTERNALS.md) for byte-level layout.

---

## How It Works

**Packaging:** Translate → Optimize IR → Generate code (CLF or scalar) → Compress weights → Build container → SHA-256 hash.

**Execution:** Verify hash → Load → Decompress → Execute (plan/SIMD or scalar fallback).

---

## Key Code Locations

| Area | Location |
|------|----------|
| Container format | `coelanox-core/src/format.rs` |
| Universal IR | `coelanox-core/src/ir.rs` |
| Packaging | `coelanox-packager/src/lib.rs` |
| Runtime | `coelanox-orchestrator/src/runtime.rs` |
| Backend trait | `coelanox-packager/src/backend_trait/` |

---

## Implementation Status

See [docs/FEATURES.md](docs/FEATURES.md) for implemented and planned features.
