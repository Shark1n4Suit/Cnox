# Troubleshooting — SDK integration

Symptoms and fixes for **package**, **verify**, translators, and CLF. For deployment and env vars see [OPERATIONS.md](../OPERATIONS.md).

---

## Packaging

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `ONNX operator 'X' is not implemented` / `Custom` error | IR still contains **`OpType::Custom`** | Lower the op in the translator, add a decomposition in `onnx.rs`, or reject the op at package time with a clear message. |
| `compute_topological_order` failed | Cycle or missing producer | Ensure every intermediate tensor has a defining node; no cycles. |
| No plugin for format | Translator not built or not registered | Enable the right **Cargo feature** (`translator-onnx`, …) and check **`PluginManager::load_default_plugins`**. |
| Wrong translator picked (e.g. BERT vs ONNX) | Demo mode | Adjust **`--use-demo-translator`** and **`-m`** / format; see [CLI_REFERENCE.md](../CLI_REFERENCE.md). |

---

## CLF and execution

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `ClfError::InvalidMagic` | File is not CLF | Confirm magic **`CLF1`**, correct file path. |
| `ClfError::MissingOpId` | Blob section built with **`Fail`** policy and graph needs an op | Ship a `.clfc` that includes every **op_id** in the plan, or use policy **Skip** only if runtime can handle missing steps (unusual). |
| `KindMismatch` | Wrong CLF kind | Use **`open_with_expected_kind`** with **`ClfKind::Compute`** for compute paths; CLFMM/CLFMP use different kinds. |
| Wrong numerical result on CPU | Shape / attribute mismatch | IR **`shape`** attributes must match actual buffer sizes the planner assumes; compare with [ONNX decomposition](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) for ONNX. |

---

## Development workflow

| Symptom | What to do |
|---------|------------|
| Translator tests pass locally, CI fails | Pin Rust version; enable same **`--features`** as CI. |
| `cargo doc` missing symbols | Use **`cargo doc -p coelanox-core --no-deps`**; optional docs may need **`--document-private-items`** for internals. |

---

## Getting help

- **In-tree references:** `coelanox-packager/src/translators/onnx.rs` (large decomposition reference), `coelanox-core/src/scalar/mapping.rs` (what each `OpType` does on CPU).
- **Issues:** track API breaks against **CHANGELOG** and **semver** on `coelanox-core`.

← [SDK index](README.md)
