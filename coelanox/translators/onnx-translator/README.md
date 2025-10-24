# COELANOX ONNX Translator

A build-time translator plugin that converts ONNX models into COELANOX's UniversalIR format.

## Features

- **Pure Rust Implementation**: Uses `prost` for protobuf parsing with no C++ dependencies
- **Build-time Only**: Zero runtime dependencies - translator runs only during packaging
- **Complete ONNX Support**: Handles all ONNX operations, tensors, and attributes
- **UniversalIR Integration**: Seamlessly integrates with COELANOX's innovative IR system
- **Operation Mapping**: Maps ONNX operations to COELANOX primitives with decomposition support
- **Weight Extraction**: Extracts and converts all model weights and parameters
- **Graph Construction**: Builds complete computation graphs with proper tensor connections

## Architecture

### Translation Pipeline

```
ONNX Model → Prost Parser → UniversalIR → COELANOX Container
```

### Operation Mapping Strategy

**Direct Mapping (Simple Operations):**
- `Add` → `OpType::Add`
- `Relu` → `OpType::Relu`
- `Conv` → `OpType::Convolution`
- `MatMul` → `OpType::MatMul`

**Decomposition (Complex Operations):**
- `Attention` → `MatMul` + `Softmax` + `MatMul`
- `LSTM` → `MatMul` + `Sigmoid` + `Tanh` + `Multiply` + `Add`

**Fallback:**
- Unknown operations → `OpType::Custom(String)`

### Data Type Mapping

| ONNX Type | COELANOX Type | Notes |
|-----------|---------------|-------|
| FLOAT (1) | Float32 | Direct mapping |
| UINT8 (2) | UInt8 | Direct mapping |
| INT32 (6) | Int32 | Direct mapping |
| INT64 (7) | Int64 | Direct mapping |
| BOOL (9) | Bool | Direct mapping |
| FLOAT16 (10) | Float32 | Converted to float32 |
| DOUBLE (11) | Float64 | Direct mapping |

## Usage

### Integration with COELANOX Packager

The translator automatically integrates with the COELANOX packager:

```rust
use onnx_translator::OnnxTranslator;
use coelanox_core::plugins::TranslatorPlugin;

let translator = OnnxTranslator::new();
let ir = translator.translate(&onnx_model_bytes)?;
```

### Plugin Registration

Register the translator in your packager:

```rust
use onnx_translator::OnnxTranslator;

let translator = Box::new(OnnxTranslator::new());
plugin_manager.register_plugin("onnx_translator".to_string(), translator)?;
```

## Implementation Details

### Tensor ID System

Uses COELANOX's revolutionary bit-packed TensorId system:
- **Weight tensors**: `TensorId::weight(index)` - bit 0 = 0
- **Intermediate tensors**: `TensorId::intermediate(index)` - bit 0 = 1
- **O(1) lookups**: Efficient tensor access without hash maps

### Graph Construction

Builds complete computation graphs with:
- **NodeId**: Unique identifiers for operations
- **TensorId**: Smart tensor references with type encoding
- **Proper connections**: Input/output tensor linking
- **Attribute mapping**: ONNX attributes → COELANOX attributes

### Memory Efficiency

- **Arena-based storage**: Cache-friendly memory layout
- **Arc weight sharing**: Zero-copy weight sharing during optimization
- **Bit-packed IDs**: Minimal memory overhead

## Building

The translator uses `prost-build` to generate Rust structs from ONNX protobuf schemas:

```bash
cargo build
```

This will:
1. Compile ONNX protobuf schemas using `prost-build`
2. Generate `onnx.rs` with Rust structs
3. Compile the translator with zero runtime dependencies

## Testing

Run the test suite:

```bash
cargo test
```

Tests cover:
- Translator creation and metadata
- Operation mapping
- Data type conversion
- Supported formats

## Dependencies

- **prost**: Pure Rust protobuf implementation
- **prost-build**: Build-time protobuf compilation
- **coelanox-core**: COELANOX IR and plugin system
- **serde**: Serialization support

## License

MIT License - see LICENSE file for details.

## Contributing

This translator is part of the COELANOX ecosystem. Contributions should follow the COELANOX development guidelines and maintain compatibility with the UniversalIR format.
