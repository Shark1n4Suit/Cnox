# Frontend SDK — How to Build a Translator

Translators are **compiler frontends**: they lower model formats (ONNX, BERT, PyTorch, etc.) to UniversalIR. This guide describes how to implement a new translator.

---

## 1. Trait Contract

Implement [`CoelanoxTranslator`] in `coelanox-core`:

```rust
use coelanox_core::{CoelanoxTranslator, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use coelanox_core::ir::UniversalIR;
use std::path::Path;

pub struct MyTranslator;

impl CoelanoxTranslator for MyTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        // Parse model_data, build IR
        todo!()
    }

    fn translate_from_path_with_weights(
        &self,
        input_path: &Path,
        external_weights_path: Option<&Path>,
    ) -> CoelanoxResult<UniversalIR> {
        let data = std::fs::read(input_path)?;
        self.translate(&data)
    }

    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        todo!()
    }

    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Custom("my_format".to_string())]
    }

    fn name(&self) -> &str { "my_translator" }
    fn version(&self) -> &str { "1.0.0" }
    fn author(&self) -> &str { "Your Name" }
}
```

**Required methods:** `translate`, `translate_from_path_with_weights`, `extract_weights`, `supported_formats`, `name`, `version`, `author`.

---

## 2. IR Output Schema

Your translator must produce a valid `UniversalIR` graph.

### Builder API

```rust
let mut ir = UniversalIR::new("model_name".to_string());
ir.metadata.add_input_shape(Shape::new(vec![1, 3, 224, 224]));
ir.metadata.add_output_shape(Shape::new(vec![1, 1000]));

let input_id = ir.add_intermediate_tensor("input", Shape::new(vec![1, 3, 224, 224]), DataType::Float32);
let weight_id = ir.add_weight_tensor_with_data("conv.weight", Shape::new(vec![64, 3, 7, 7]), DataType::Float32, weight_bytes);
let output_id = ir.add_intermediate_tensor("output", Shape::new(vec![1, 64, 112, 112]), DataType::Float32);

let node_id = ir.add_node(OpType::Convolution);
ir.add_node_input(node_id, input_id)?;
ir.add_node_input(node_id, weight_id)?;
ir.add_node_output(node_id, output_id)?;

ir.get_node_mut(node_id).unwrap().add_attribute("kernel_size".to_string(), Attribute::IntArray(vec![7, 7]));
ir.compute_topological_order()?;
```

---

## 3. OpType Vocabulary

Use the canonical `OpType` enum. For unsupported ops, use `OpType::Custom("OpName".to_string())`.

---

## 4. Shape Constants and Non-Float Initializers

### What the IR expects

- **Weight tensors** — Only float32 (and float16, converted to f32). Use `add_weight_tensor_with_data` with `DataType::Float32`.
- **Shape constants** — Reshape, Slice, and similar ops use **attributes**, not weight tensors. Do not add int64/int32 shape tensors as weights.

### Reshape

- **IR contract:** Reshape has one tensor input (data). The target shape is an **attribute** `"shape"` (`Attribute::IntArray`), e.g. `[1, -1, 224, 224]` where `-1` means infer from total elements.
- **Translator responsibility:** When the source format provides shape as a separate input (e.g. ONNX Reshape’s second input), read that constant and add it as the `"shape"` attribute. Do not wire the shape as a second tensor input.

### Non-float initializers (ONNX)

- **int64 / int32 initializers** — Often used for Reshape’s shape input. Translators must:
  1. **Skip** them when building the weight blob (they are not weights).
  2. **Store** their values (e.g. in a `HashMap<name, Vec<i64>>`).
  3. **Use** them when processing Reshape nodes: add `shape` attribute from the stored constant instead of wiring as input.

### What was fixed (ONNX translator)

- Previously: `"Only float32 and float16 initializers supported"` — models like ResNet-50 failed because they use int64 shape constants.
- Now: int64/int32 initializers are skipped as weights and used as shape attributes for Reshape. ResNet-50, MobileNet, and similar models package successfully.

---

## 5. ModelFormat

Use `ModelFormat::Custom("your_format".to_string())` for new formats.

---

## 6. Versioning

Pin to `coelanox-core` semver. Same major version for IR compatibility.

---

## 7. Packager Cargo features (optional)

The packager gates translators via Cargo features. To build a smaller binary:

```bash
# ONNX only
cargo build -p coelanox-packager --no-default-features --features translator-onnx

# BERT only
cargo build -p coelanox-packager --no-default-features --features translator-bert

# ResNet only
cargo build -p coelanox-packager --no-default-features --features translator-resnet
```

Default enables all three. See `coelanox-packager/Cargo.toml` for feature definitions.

---

## References

- [Frontend SDK Roadmap](../roadmaps/FRONTEND_SDK_ROADMAP.md)
- `coelanox-core/src/plugins.rs` — CoelanoxTranslator trait
- `coelanox-core/src/ir.rs` — UniversalIR, OpType
