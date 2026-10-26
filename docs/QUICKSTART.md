# COELANOX Quickstart

Get a model packaged, verified, and running in under five minutes.

---

## Install

**Option A: From release** (no Rust) — Download [Releases](https://github.com/coelanox-ai/coelanox/releases) (`coelanox-linux-x86_64.tar.gz` or `coelanox-windows-x86_64.zip`), extract, add to PATH. Run `./install.sh` on Linux/macOS or add the folder to PATH.

**Option B: From source** — Rust stable. `cd coelanox && cargo build --release -p coelanox-cli`. Or: `cargo install --path coelanox/coelanox-cli`.

---

## 1. Package a model

```bash
coelanox package -i model.onnx -o model.cnox --target cpu
```

**Example output:**
```
✅ Plugin manager initialized
✅ Packaged model written to model.cnox
```

For a demo model (ResNet-tiny-MNIST bundle):

```bash
coelanox package -i resnet_tiny_mnist.bin -o resnet.cnox -m cv --use-demo-translator --target cpu
```

---

## 2. Verify integrity

```bash
coelanox verify -f model.cnox
```

**Example output (success):**
```
✅ Container integrity verified
```

**Example output (tampered):**
```
❌ Security validation failed: hash mismatch
```

---

## 3. Run inference

```bash
# With input file (JSON array of f32)
coelanox run -f model.cnox -i input.json -o output.json

# With dummy input (for testing)
coelanox run -f model.cnox -o output.json
```

**Example output:**
```
✅ Container integrity verified
✅ Inference completed
Output written to output.json
```

`output.json` is a **JSON array of `f32` floats** (flattened outputs). For **serve** (binary IPC), **benchmark** text vs JSON, and how to turn floats into **human-readable text** in your app, see **[INTEGRATION_OUTPUT.md](INTEGRATION_OUTPUT.md)**.

---

## 4. Inspect the container

```bash
coelanox info -f model.cnox
```

Shows: manifest, input/output shapes, sizes, workspace requirements.

---

## Optional: signing

```bash
# Generate keypair
coelanox keygen -o ./keys

# Package with signing
coelanox package -i model.onnx -o model.cnox --signing-key keys/secret.bin

# Verify with trusted key
coelanox verify -f model.cnox --trusted-key keys/public.bin

# Run with trusted key
coelanox run -f model.cnox -o out.json --trusted-key keys/public.bin
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Package` fails with "No plugin" or format error | Ensure ONNX feature: build with `--features onnx-protobuf` or use a demo model with `--use-demo-translator` |
| `Verify` reports hash mismatch | Container is corrupted or tampered; re-download or re-package from a trusted source |
| `Run` is slow | Scalar fallback is in use; place `.clfc` files in `~/.coelanox/clf/clfc/` for SIMD, or expect scalar for `--fallback-only` packages |
| `Run` fails with "workspace" or "scratch" | Model needs more memory; use `--max-workspace-mb` at package or run time |

---

## Demo benchmark (ResNet-tiny MNIST)

With CLF in `~/.coelanox/clf/clfc/`, full CLF path matches or slightly beats scalar (~56–61 ms cold). Scalar fallback: ~57–66 ms. See [Operations](OPERATIONS.md) for CLF path setup.

---

## Next steps

- [CLI Reference](CLI_REFERENCE.md) — Full command options
- [Architecture](ARCHITECTURE.md) — System design, codebase map
- [Operations](OPERATIONS.md) — Config, security, runbooks
