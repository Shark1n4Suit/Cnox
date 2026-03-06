# COELANOX Container Internals

This document describes the binary layout and contents of the `.cnox` container format. The structure is illustrated with a ResNet-tiny-MNIST example; the layout applies to all COELANOX containers.

## File layout (byte order)

| Section | Offset | Size | Description |
|--------|--------|------|-------------|
| **Header** | 0 | 76 | Fixed header (magic, version, sizes, security hash) |
| **Manifest** | 76 | 4 + L | Length (u32 LE) + JSON manifest |
| **Optimized code** | 76 + 4 + L | `manifest.model_size` | Target machine code (e.g. scalar backend) |
| **Compressed weights** | next | 4 + W | Length (u32 LE) + weight bytes |
| **IR data** | next | 4 + I | Length (u32 LE) + bincode-serialized OptimizedIR |
| **NVMe backing** | next | 8 + N | Length (u64 LE) + pre-allocated memory layout |
| **CLFMM** (optional) | next | 8 + size | Memory-movement blobs |
| **CLFMP** (optional) | next | 8 + size | Memory-protection blobs |

## Header (76 bytes)

- **Magic** (4): `COEL`
- **Version** (4): `0x00010000` (1.0.0)
- **Flags** (4): reserved
- **Container size** (8): total file size
- **Manifest offset** (8): always 76
- **Code offset** (8): where machine code starts
- **Security hash** (32): SHA-256 of entire container (written after build)
- **Reserved** (8)

## Manifest (JSON)

- **name**: Model name (e.g. `resnet_tiny_mnist`)
- **format**: Source format (currently stored as `Onnx`; ResNet uses Custom in pipeline but manifest is not yet format-aware)
- **input_shapes** / **output_shapes**: e.g. `[[1,1,28,28]]` → `[[1,10]]`
- **model_size**: Size of optimized code in bytes (0 when code is empty, e.g. scalar path)
- **created_at** / **modified_at**: Unix timestamps
- **nvme_backing_offset** / **nvme_backing_size**: Offset and size of NVMe backing region in the file
- **backend_metadata**, **clfmm_size**, **clfmp_size**: Optional backend and HAL data

## ResNet container example (`resnet_tiny_mnist.cnox`)

| Section | Size (bytes) |
|---------|----------------|
| Header | 76 |
| Manifest | 327 (4 + 323 JSON) |
| Optimized code | 0 |
| Compressed weights | 733,996 |
| IR data | 750,318 |
| NVMe backing | 2,393,280 |
| **Total** | **3,877,997** |

- **Optimized code = 0**: Scalar/CPU path may emit no separate “machine code” blob; execution is driven by the IR.
- **Compressed weights**: Flat weight buffer (optionally compressed); runtime reconstructs tensors from IR metadata + this buffer.
- **IR data**: Serialized `OptimizedIR` (bincode): nodes, tensor metadata, topology—no raw weights, only shapes and wiring.
- **NVMe backing**: Precomputed memory layout (offsets/sizes) for loading the model into the target memory tier.

## How to inspect

- **CLI**: `coelanox info -f resnet_tiny_mnist.cnox` — manifest summary, sizes, I/O shapes.
- **Validate**: `coelanox validate -f resnet_tiny_mnist.cnox` — structure and hash.
- **Debug**: `coelanox debug -f resnet_tiny_mnist.cnox` — more detailed container and IR view.

To parse the manifest from the file: read 76 bytes (skip header), then 4 bytes (u32 LE) = manifest JSON length, then that many bytes as UTF-8 JSON.
