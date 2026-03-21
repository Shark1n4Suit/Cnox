# Developer Guide

Building translators and integrating the CLF crate.

---

## Building a translator

Translators lower model formats (ONNX, BERT, etc.) to UniversalIR. Implement `CoelanoxTranslator` in `coelanox-core`:

```rust
impl CoelanoxTranslator for MyTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> { todo!() }
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> { todo!() }
    fn supported_formats(&self) -> Vec<ModelFormat> { vec![ModelFormat::Custom("my_format".into())] }
    fn name(&self) -> &str { "my_translator" }
    fn version(&self) -> &str { "1.0.0" }
    fn author(&self) -> &str { "You" }
}
```

**IR builder API:** `UniversalIR::new()` → `add_intermediate_tensor`, `add_weight_tensor_with_data`, `add_node`, `add_node_input`, `add_node_output`, `add_attribute`, `compute_topological_order()`.

Use `OpType` enum; unsupported ops → `OpType::Custom("OpName".into())`. Reshape uses `"shape"` attribute (`IntArray`), not a second tensor input.

**Cargo features:** Packager gates translators. `--no-default-features --features translator-onnx` for ONNX only. See `coelanox-packager/Cargo.toml`.

---

## CLF integration

The packager uses a CLF reader for kernel blobs. Your implementation must provide:

- `ClfReader::open(path) -> Result<Self>` — Open .clf file
- `get_blob(&self, op_id: u32) -> Option<Vec<u8>>` — Return blob for op_id
- `read_blob_array(&self)` — Full payload (optional)
- `list_op_ids(&self) -> Vec<u32>` — All op_ids

**Adding your CLF repo:**
- **Submodule:** Add as submodule, point `coelanox-packager` at it in Cargo.toml
- **Path dep:** `coelanox-clf = { path = "../../your-clf-repo" }` in packager
- **Published:** `coelanox-clf = "0.1"` once on crates.io

Op_id numbering must match packager's `op_type_to_clf_id` mapping. See `coelanox-clf/src/op_registry.rs`.
