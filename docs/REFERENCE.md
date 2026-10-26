# COELANOX Reference

What COELANOX is, what `.cnox` files contain, and implementation status.

---

## What is COELANOX?

**COELANOX** (Secure AI Container) is a universal binary container system for AI models:

1. **Universal Format** — Convert models (ONNX; BERT/ResNet demos) into a single `.cnox` container
2. **AOT Compilation** — Optimized machine code for target hardware (CLF or scalar)
3. **Universal IR** — Graph-based intermediate representation
4. **Hardware Abstraction** — Backend translators map ops to hardware-specific code

**Pipeline:** `Input Model → Translator → Universal IR → Optimize → Code/weights → .cnox → Runtime`

---

## What is a `.cnox` File?

| Section | Description |
|---------|-------------|
| **Header** (76 bytes) | Magic COEL, version, flags, sizes, SHA-256 hash |
| **Manifest** (JSON) | Model name, input/output shapes, backend metadata |
| **Machine Code** | Pre-compiled (CLF blobs or empty for scalar) |
| **Weights** | Zstd/LZ4 at package time, or uncompressed |
| **IR Data** | Serialized OptimizedIR (bincode) |
| **Optional** | NVMe backing, CLFMM, CLFMP |

---

## Implementation status

| Area | Implemented | Planned |
|------|-------------|---------|
| Container | .cnox, SHA-256, verify, run | — |
| Model import | BERT, ResNet, ONNX (opset 13) | Generic PyTorch/TF translators |
| Execution | Scalar + CLF/plan, backend discovery | Vendor SDK, AMD reference |
| IR & ops | Universal IR, OpType set, CLF op_ids | Fusion, more ops |
| Security | Hash, Ed25519 signing, path/size limits | Encryption, code signing |
| Config | Env + file, runbooks, workspace cap | — |
| Deployment | CLI, extract (same-arch runtime) | extract --arm (cross-arch) |

---

## Key code locations

| Area | File |
|------|------|
| Container format | `coelanox-core/src/format.rs` |
| Universal IR | `coelanox-core/src/ir.rs` |
| Packaging | `coelanox-packager/src/lib.rs` |
| Runtime | `coelanox-orchestrator/src/runtime.rs` |

See [ARCHITECTURE](ARCHITECTURE.md) for full codebase map.
