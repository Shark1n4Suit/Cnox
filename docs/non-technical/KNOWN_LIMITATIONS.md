# COELANOX — Known limitations (honest list)

**Audience:** Customers and security teams—written to surface limits **before** they surface as incidents.

This document is **not** a roadmap commitment; features may move. For **implementation status**, see also [Reference](../REFERENCE.md).

---

## 1. Quantization

| Topic | Limitation |
|-------|------------|
| **INT8 / QAT / ONNX quant** | **Not** a first-class, documented production path for arbitrary quantized graphs. **QuantizeLinear** / **DequantizeLinear** and similar ops are **not** treated as guaranteed end-to-end paths in the customer ONNX story. |
| **What works today** | **Float32** activations/weights are the **primary** packaging target for ONNX import unless you have a **custom** translator negotiated with your vendor. |

**If you need quantization:** Assume **export to float**, **validate numerics**, or invest in **translator/runtime work**—do not assume parity with ONNX Runtime’s quant paths.

---

## 2. Dynamic shapes

| Topic | Limitation |
|-------|------------|
| **Fully dynamic** graphs | Packaging and planning assume **manifest-fixed** input/output shapes for the **container** you ship. Highly dynamic ONNX (runtime-dependent ranks) may **not** translate or may require **static** export (fixed batch, fixed sequence length). |
| **What to do** | Export with **fixed** representative shapes; validate with **`coelanox info`** and test runs. |

---

## 3. ONNX coverage

| Topic | Limitation |
|-------|------------|
| **Not full ONNX** | Only what the **opset 13** translator lowers is supported. Unsupported ops become **`Custom`** and **`package` fails**. |
| **Control flow** | **If** / **Loop** are **rejected**—move logic to the application. |
| **Reference** | [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md) and the [full decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md). |

---

## 4. Framework import

| Topic | Limitation |
|-------|------------|
| **PyTorch / TensorFlow direct** | **No** generic “save → .cnox” for arbitrary models. **ONNX export** or **supported bundles** (BERT demo, ResNet-tiny demo) are the in-tree paths. |

---

## 5. Scalar (fallback) performance

| Topic | Limitation |
|-------|------------|
| **Speed** | **Scalar** execution is a **correctness / portability** path. Large models (e.g. big transformers) can be **orders of magnitude slower** than optimized runtimes. |
| **When it runs** | **`--fallback-only`** packages; missing CLF; wrong backend discovery; unsupported native path. |
| **What to do** | Package **with** native/SIMD path when available; install **`.clfc`** artifacts in the documented discovery path; set expectations for **batch** vs **real-time**. |

---

## 6. Hardware and platforms

| Topic | Limitation |
|-------|------------|
| **SIMD / CLF** | Optimized paths target **x86_64** in typical deployments; other architectures may be **scalar-only** unless your vendor ships otherwise. |
| **GPU / NPU** | **Not** described here as generally available first-class backends; roadmap territory—confirm per release. |

---

## 7. Long-running service features

| Topic | Limitation |
|-------|------------|
| **HTTP / gRPC server** | **Not** built into the open-source CLI. **`serve`** is **stdio IPC**. You provide the outer service, health checks, and metrics. |
| **Observability** | **Tracing/logs** yes; **Prometheus / OpenTelemetry** not built-in—wire stdout to your stack. |

See [Operations](../OPERATIONS.md) “Production readiness”.

---

## 8. Security and compliance

| Topic | Limitation |
|-------|------------|
| **Encryption at rest** | **`.cnox`** is **not** encrypted by COELANOX; use **disk encryption** or outer packaging. |
| **Compliance regimes** | COELANOX helps with **integrity** and **evidence**; it does **not** by itself satisfy EU AI Act or similar **organizational** obligations. |

---

## 9. Numerical equivalence

| Topic | Limitation |
|-------|------------|
| **Bit-exact** | **Not guaranteed** between **scalar** and **SIMD** (or across OS/libm). Expect **close** floats for well-behaved models. |
| **Determinism** | See [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md). |

---

## Related documents

- [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) — Non-goals  
- [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md) — Behaviour  
- [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md) — What compiles  

← [Non-technical hub](README.md)
