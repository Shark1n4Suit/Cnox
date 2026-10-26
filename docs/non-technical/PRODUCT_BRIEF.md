# COELANOX — Product brief (technical / security audience)

**Audience:** Security architects, procurement technical reviewers, platform engineers evaluating deployable inference artifacts.  
**Length:** This document is intentionally short; it states what the product **is**, what it **guarantees**, what it **does not** promise, and the design stance that **errors must be visible**—not smoothed over.

---

## 1. What COELANOX is

COELANOX is an **inference packaging and runtime system** that turns supported source models into a single **`.cnox` binary container**: serialized graph (**Universal IR**), **weights** (optionally compressed), optional **ahead-of-time code** (CLF blobs / execution plans), and a **SHA-256 integrity hash** over the artifact. A single **`coelanox`** CLI supports **package**, **verify** (integrity without executing), and **run** (verify-then-execute by default).

Execution is **not** defined by ONNX at runtime. **Universal IR** plus a versioned **IR semantics** contract is the behavioral contract; ONNX is one **import** path. See [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md) and [IR semantics](../IR_SEMANTICS.md).

---

## 2. What it guarantees (technical)

| Guarantee | Meaning |
|-----------|---------|
| **Artifact integrity** | A **SHA-256** hash covers the container content (with the hash field zeroed for calculation). **`coelanox verify`** and **`coelanox run`** (unless `--no-verify`) detect tampering or corruption. |
| **Semantics versioning** | Containers carry **`ir_semantics_version`**. Loaders **reject** unknown or incompatible versions with an explicit error—no silent execution across incompatible toolchains. |
| **Explicit failure on unknown graph ops at package time** | If translation leaves **`Custom("OnnxOpName")`** nodes, **`coelanox package`** **fails** until the graph is expressible in IR or the translator is extended. The product does not “best effort” ship undefined ops. |
| **Configurable bounds** | Runtime configuration can cap container size, input size, workspace, execution time, and path policy (see [Operations](../OPERATIONS.md)). |
| **Optional provenance signing** | **Ed25519** signing at package time; verification with a **trusted public key**; optional **`--require-signature`** on consume paths. |

These are **software guarantees** around integrity, versioning, and policy—not a certification of model **correctness** or **fairness**.

---

## 3. What it explicitly does **not** do

| Non-goal | Detail |
|----------|--------|
| **Training** | COELANOX is **inference-only**. It does not train or fine-tune models. |
| **Full ONNX parity** | Only what the **in-tree translator** lowers to Universal IR is supported. Unsupported ONNX ops become **`Custom`** and **block packaging** until addressed. |
| **Bit-identical outputs across all backends** | **Scalar**, **SIMD/CLF plan**, and future paths may differ within **floating-point** tolerance. See [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md). |
| **Generic quantized production path** | Quantized ONNX paths are **not** a first-class, documented guarantee today. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). |
| **Compliance program** | COELANOX strengthens **artifact integrity** and **technical evidence** (audit logs, fingerprints); it is **not** a substitute for organizational compliance (e.g. EU AI Act) or formal certification. |
| **Managed cloud / HTTP API** | The shipped **`coelanox serve`** mode is **stdin/stdout framed IPC** for one loaded container—not a full HTTP/gRPC product. See [Operations](../OPERATIONS.md). |

---

## 4. Supported model types (import paths today)

| Source | Notes |
|--------|--------|
| **ONNX** | **Opset 13** translator in-tree. Coverage is **not** all of ONNX; see [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md) and the [full decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md). |
| **BERT bundle** | Custom bundle format (e.g. `bert_base_uncased` style) via translator plugin. |
| **ResNet-tiny-MNIST demo** | Demo bundle for CV path; not a general PyTorch/TensorFlow importer. |

**Roadmap** (not commitments): broader framework import, more ops, vendor backends—see [Reference](../REFERENCE.md) and project roadmaps.

---

## 5. Supported hardware and platforms (today)

| Aspect | Today |
|--------|--------|
| **Primary execution** | **CPU**. Scalar fallback is **always available** and runs wherever the runtime binary runs. |
| **SIMD / CLF** | **x86_64** CLF blobs and SIMD execution paths are the **expected** fast path when packaged and discovered; discovery uses `~/.coelanox/clf/` layout by default (see [Operations](../OPERATIONS.md)). |
| **Portability** | **`--fallback-only`** packages contain **no embedded native code**; execution uses the **scalar** path for maximum portability at the cost of performance. |
| **Other ISAs** | Not a guaranteed parity target in this brief; treat as **project-specific** until documented per release. |

---

## 6. “It tells you when you’re wrong” (design philosophy)

COELANOX is intentionally **strict** at boundaries:

1. **Wrong container** → hash or signature verification **fails** with an error; execution does not proceed silently.  
2. **Wrong toolchain mix** → IR semantics version **mismatch** → **fail fast** with a clear message.  
3. **Unsupported graph** → **`Custom`** ops → **package fails** until fixed or translated.  
4. **Policy violation** → path, size, or timeout limits → **controlled errors** (see runtime spec).  
5. **No silent downgrade of meaning** → the manifest records **how** the container was built (**packaging fingerprint**); it does not change numerics, but it **surfaces** drift between build and runtime expectations.

The goal is **observable failure** and **auditability**, not hiding errors behind ambiguous behaviour.

---

## Related documents

- [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md) — Behaviour, determinism, failure modes  
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — Honest limits  
- [DATA_FLOW.md](DATA_FLOW.md) — Where data moves and what persists  
- [Architecture (detail)](../ARCHITECTURE.md) — System design  

← [Non-technical hub](README.md)
