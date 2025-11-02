# COELANOX BERT Translator

A specialized build-time translator that converts BERT ONNX models into COELANOX's UniversalIR format. This translator is optimized specifically for BERT models, making it faster and simpler than the general ONNX translator.

## Features

- **BERT-Specific Optimizations**: Recognizes BERT patterns (attention, embeddings, layer norm)
- **Pattern Recognition**: Detects attention blocks, embedding layers, and feed-forward layers
- **Efficient Weight Extraction**: Optimized for BERT weight naming conventions
- **Faster Translation**: Simpler than general ONNX translator (handles only BERT-relevant ops)
- **Pure Rust**: No external dependencies beyond prost for protobuf parsing

## Architecture

### Translation Pipeline

```
BERT ONNX Model → Protobuf Parser → BERT Pattern Recognition → UniversalIR
```

### BERT-Specific Operation Mapping

**Attention Operations:**
- `MatMul` with "query"/"key"/"value" in name → `AttentionMatMul`
- Attention blocks decomposed to MatMuls + Softmax

**Embedding Operations:**
- `Gather` with "embedding" in name → `Embedding`
- Token, position, and segment embeddings recognized

**Layer Operations:**
- `LayerNormalization` → `LayerNorm`
- `Gelu` → `Gelu` activation

**Standard Operations:**
- All standard ONNX operations mapped directly

## Usage

### Integration with COELANOX Packager

The translator automatically integrates with the COELANOX packager:

```rust
use bert_translator::BertTranslator;
use coelanox_core::plugins::TranslatorPlugin;

let translator = BertTranslator::new();
let ir = translator.translate(&bert_model_bytes)?;
```

### Plugin Registration

Register the translator in your packager (done automatically in `load_default_plugins()`).

## BERT Model Support

- **bert-base-uncased**: Fully supported
- **bert-large-uncased**: Fully supported
- **Custom BERT variants**: Supported if they follow standard BERT architecture

## Limitations

- Only handles BERT-specific ONNX models
- Assumes standard BERT architecture (12 or 24 layers)
- Custom BERT variants with unusual operations may not be fully supported
- Use general ONNX translator for non-BERT models

