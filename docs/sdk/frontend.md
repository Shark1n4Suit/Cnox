# Frontend SDK — translators and UniversalIR

There is no separate `coelanox-translator-sdk` crate. **`coelanox-core`** is the public contract: the `CoelanoxTranslator` trait (`coelanox_core::plugins`), `UniversalIR`, `OpType`, and `ModelFormat`. Third-party translators depend only on **`coelanox-core`** and emit **UniversalIR**—the same intermediate representation the packager optimizes and serializes into `.cnox` containers.

**Symmetry:** CLF backends expose fixed **op_id** blobs; translator frontends expose a fixed **IR vocabulary** (`OpType`). See [roadmaps/FRONTEND_SDK_ROADMAP.md](../../roadmaps/FRONTEND_SDK_ROADMAP.md) for roadmap (dynamic plugins, `export_translator!`, etc.).

### Open translator, closed CLF

Kernel and CLF tooling is often **not** public the way **`coelanox-core`** is. Translator code still lands in **open** trees and is easy to fork—so be deliberate:

- **Contract only:** Your job is **bytes → valid `UniversalIR`** (`OpType`, tensors, attributes). Do not bake in unpublished **`op_id`** maps, fusion rules, or backend-only behavior; the packager and closed stacks own lowering to kernels.
- **No leakage:** Avoid customer models, proprietary formats, or NDA’d Coelanox details in fixtures, comments, error text, or commit messages. Use synthetic or licensed **minimal** fixtures for tests.
- **No “hidden API” assumptions:** If something is not documented on `UniversalIR` / `OpType` / published CLI behavior, don’t rely on it in public translator code—coordinate under partner terms instead.

**API reference (Rust):**

```bash
cd coelanox
cargo doc -p coelanox-core --no-deps --open
```

Focus on modules **`plugins`** (trait) and **`ir`** (graph, tensors, attributes).

---

## `CoelanoxTranslator` trait

Defined in `coelanox-core/src/plugins.rs`. Implementors must be `Send + Sync + 'static`.

| Method | Role |
|--------|------|
| `translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>` | **Primary entry:** bytes → IR. |
| `translate_with_context(model_data, model_type, custom_input_shapes, custom_output_shapes)` | Optional CLI/context hooks; **default** ignores extras and calls **`translate`**. Override when your format needs shape overrides from the user. |
| `translate_from_path(&Path)` | Reads file, calls `translate_from_path_with_weights(..., None)`. |
| `translate_from_path_with_weights(&Path, Option<&Path>)` | ONNX-style external weights; default reads single file or errors if external path set. **Override** if your format supports sidecar weights. |
| `extract_weights` / `extract_weights_with_external` | Float weights for manifests/logging; same external-weights pattern. |
| `supported_formats() -> Vec<ModelFormat>` | Declare what you handle (`Onnx`, `Coelanox`, or `Custom("id")`). |
| `name`, `version`, `author` | Metadata for logs and discovery. |
| `description` | Optional; default `"No description available"`. |
| `supports_format(ModelFormat) -> bool` | Default: `supported_formats().contains(&format)`. |

**Output contract:** Build a valid **`UniversalIR`**: register tensors, add nodes, wire `add_node_input` / `add_node_output`, attach **`Attribute`** values where the runtime expects them (e.g. `shape` on `Reshape`), then call **`compute_topological_order()`** before returning. Optionally call **`validate()`** on the IR before returning to catch wiring errors early. Invalid graphs fail at package time, not silently in the scalar backend.

**Thread safety:** The trait requires **`Send + Sync`**. Translators are typically **stateless** or behind **`Arc`**; avoid mutable global state.

**Errors:** Return **`CoelanoxResult<UniversalIR>`** using **`CoelanoxError::format(...)`** with **actionable** messages (what failed, what the exporter should change). Do not panic in production paths.

---

## UniversalIR and `OpType`

- **Nodes** reference `TensorId`s (intermediates and weights). Use `UniversalIR::add_intermediate_tensor`, `add_weight_tensor_with_data`, `add_node`, etc.
- **Ops** use the `OpType` enum in `coelanox-core/src/ir.rs`. Prefer a real primitive (`MatMul`, `Relu`, …) over `Custom`.
- **Unsupported source ops:** If you cannot lower an operator, **`OpType::Custom("OnnxOpName")`** is allowed during construction, but **`coelanox package` fails** with a clear error if any **`Custom`** remains—do not rely on the scalar backend to execute unknown op names.
- **Reshape:** supply target shape via node **`shape`** attribute (`Attribute::IntArray`), not only a second tensor input, so the planner can read dimensions without int64 tensors on the edge.

For ONNX specifically, the in-tree translator maps and decomposes many opset-13 operators; extend `coelanox-packager/src/translators/onnx.rs` following existing patterns.

---

## UniversalIR builder reference (production)

Source: **`coelanox-core/src/ir.rs`**. Typical sequence:

1. **`UniversalIR::new(model_name)`**
2. Register weights: **`add_weight_tensor_with_data`** (or **`add_weight_tensor`** + fill later) — prefer **float32** little-endian bytes for **`DataType::Float32`**.
3. Register intermediates: **`add_intermediate_tensor(name, shape, data_type)`** for each graph edge.
4. **`let id = add_node(op_type)`** then **`add_node_input` / `add_node_output`** for each tensor edge, or use **`get_node_mut(id)`** and **`add_attribute`** on the node.
5. Fill **`ir.metadata`**: **`add_input_shape`**, **`add_output_shape`** for package-time validation (element counts used in planning).
6. **`compute_topological_order()`** — required before returning.

| Method (subset) | Use |
|-----------------|-----|
| `add_node` | Returns **`NodeId`**; creates an empty node. |
| `add_node_input` / `add_node_output` | Wire **`TensorId`** to node. |
| `get_node_mut` | Attach **`Attribute`** (`Int`, `IntArray`, `Float`, …). |
| `validate` | Structural checks (recommended before return). |
| `metadata` | **`IRMetadata`**: input/output shapes, optional dtype and key/value strings. |

**Dtypes:** Runtime packaging targets **float32** for activations/weights unless you have a negotiated quantized path. Int64 initializers in ONNX are often **shape constants**, not tensors on the hot path.

---

## Production checklist (translator authors)

Before merging or releasing a translator change:

- [ ] **`compute_topological_order()`** succeeds; **`validate()`** passes if applicable.
- [ ] No unresolved **`OpType::Custom`** in the final IR (or packaging will fail by design).
- [ ] **Input/output metadata** populated so manifests and workspace sizing are correct.
- [ ] **`name` / `version` / `author` / `description`** set for supportability.
- [ ] **`extract_weights`** returns a sensible flat **`Vec<f32>`** when the format has learnable weights (may be empty for weightless graphs).
- [ ] CI runs **`cargo test -p coelanox-packager`** with the relevant **`--features`**.
- [ ] Documented limitation for any **non–IEEE** lowering (e.g. approximate **`Pow`**).

---

## Testing and debugging translators

- **IR validity** — Call **`compute_topological_order()`** before returning from `translate`; ordering failures surface early.
- **`OpType::Custom`** — Allowed only as a temporary during lowering; **`coelanox package`** fails if any node still uses **`Custom`** after translation (by design). Replace with a real op or a decomposition, or improve the ONNX mapper.
- **Feature flags** — Build the packager with the same translator features you need (e.g. `--features translator-onnx`) so plugins register correctly.
- **Gold tests** — Prefer small fixed inputs/outputs against the **scalar** backend in `coelanox-core` for deterministic reference behavior.
- **Rust API** — `cargo test -p coelanox-packager --features onnx-protobuf` after translator changes.

---

## Registering an in-tree translator

1. **Implement** `CoelanoxTranslator` under `coelanox-packager/src/translators/` (see `onnx.rs`, `bert.rs`, `resnet_tiny_mnist.rs`).
2. **Feature-gate** in `coelanox-packager/Cargo.toml` (e.g. `translator-myfmt = []`) and add to `default = [...]` if it should ship by default.
3. **Register** in `PluginManager::load_default_plugins()` in `coelanox-packager/src/plugins/mod.rs`:

   ```rust
   #[cfg(feature = "translator-myfmt")]
   {
       let t = Box::new(MyTranslator::new());
       self.register_plugin("my_plugin_key".to_string(), t);
   }
   ```

4. **`supported_formats`** must include the `ModelFormat` the CLI selects (e.g. `ModelFormat::Onnx` for `.onnx`).

The packager resolves plugins by format and optional **`--use-demo-translator`** (prefers specialized demos like BERT when applicable). See [CLI_REFERENCE.md](../CLI_REFERENCE.md) for `package` flags.

**Out-of-tree / dynamic plugins** are not wired in this repo yet; see the roadmap linked above.

---

## Cargo features (packager translators)

From `coelanox-packager/Cargo.toml`:

| Feature | Effect |
|---------|--------|
| `translator-onnx` | Enables ONNX (`onnx-protobuf` / `rten-onnx`). |
| `translator-bert` | BERT bundle translator. |
| `translator-resnet` | ResNet-tiny-MNIST demo translator. |

**Defaults:** all three are enabled. Minimal ONNX-only build:

```bash
cargo build -p coelanox-packager --no-default-features --features translator-onnx
```

`onnx-protobuf` is pulled in by `translator-onnx`.

---

## Quick trait skeleton

```rust
use std::path::Path;
use coelanox_core::format::ModelFormat;
use coelanox_core::ir::UniversalIR;
use coelanox_core::plugins::CoelanoxTranslator;
use coelanox_core::CoelanoxResult;

pub struct MyTranslator;

impl CoelanoxTranslator for MyTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        let mut ir = UniversalIR::new("my_model".to_string());
        // … build graph …
        ir.compute_topological_order()?;
        Ok(ir)
    }

    fn extract_weights(&self, _model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        Ok(vec![])
    }

    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Custom("myfmt".into())]
    }

    fn name(&self) -> &str { "my_translator" }
    fn version(&self) -> &str { "0.1.0" }
    fn author(&self) -> &str { "You" }
    fn description(&self) -> &str { "Short description for logs" }
}
```

---

## See also

| Doc | Topic |
|-----|--------|
| [pipeline.md](pipeline.md) | IR → plan → container → run |
| [clf.md](clf.md) | `op_id`, CLF blobs, executor |
| [glossary.md](glossary.md) | Terms |
| [troubleshooting.md](troubleshooting.md) | Common failures |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | System layout |
| [CLI_REFERENCE.md](../CLI_REFERENCE.md) | CLI flags |

← [SDK index](README.md)
