# COELANOX — Runtime behaviour specification

**Audience:** Security and platform teams approving deployment next to existing infrastructure.  
**Scope:** Behaviour of **`coelanox`** when loading and executing a **`.cnox`** container. Packaging (`package`) has separate failure modes; only critical interactions are noted here.

---

## 1. Contract summary

| Topic | Specification |
|-------|----------------|
| **Source of truth** | Execution semantics are defined by **Universal IR** + **IR semantics version** in the manifest, not by ONNX at runtime. |
| **Same `.cnox`, same input → same output?** | **Not guaranteed bit-identical** across different **backends** (e.g. scalar vs SIMD/CLF) or across **library/OS** versions. **Scalar reference** behaviour is test-backed; SIMD should match within **float tolerance** where applicable. |
| **Determinism on same hardware** | **Scalar** path: **largely deterministic** for a fixed IR and inputs (single-threaded reference semantics for ops under test). **SIMD / native** paths may use **parallel** or **reordered** reductions; **do not assume** bitwise reproducibility for floats. |
| **Determinism across runs** | Same binary, same container, same input, same backend, same config: expect **stable** results for scalar; for SIMD, expect **close** floats unless documented otherwise. |

For IR versioning rules, see [IR semantics](../IR_SEMANTICS.md).

---

## 2. Load and verify phases

### 2.1 Container open and parse

**Failure modes:**

| Condition | Typical behaviour |
|-----------|---------------------|
| File missing / unreadable | Error returned; **no** partial load. |
| Truncated or invalid binary | Parse error; **no** execution. |
| **Container size** exceeds `COELANOX_MAX_CONTAINER_SIZE_BYTES` (or equivalent config) | Load **rejected** with policy error. |
| **`ir_semantics_version`** not accepted by this runtime | Load **rejected** with explicit version error (no silent run). |

### 2.2 Integrity verification (`verify` and default `run`)

| Condition | Behaviour |
|-----------|-----------|
| SHA-256 mismatch | **Failure**; container treated as **untrusted**. |
| `--no-verify` on `run` | Skips hash check (**development-only**; discouraged in production). |

### 2.3 Optional Ed25519 signature

When signing is used:

| Condition | Behaviour |
|-----------|-----------|
| Signature required (`--require-signature`) and key/signature invalid | **Failure**; no run. |
| Trusted key provided | Verification succeeds or fails **explicitly**. |

---

## 3. Execution phase

### 3.1 Input validation

When enabled (`COELANOX_ENABLE_INPUT_VALIDATION`, default **true**):

| Check | On failure |
|-------|------------|
| Input tensor size vs manifest limits | **Error**; no inference. |
| NaN / Inf policy (when enforced) | **Error** per configuration. |

### 3.2 Backend selection (high level)

- **Scalar:** Interpreted execution over IR; always available if the graph is supported.  
- **SIMD / CLF / plan:** Used when packaged and compatible with runtime discovery; otherwise **fallback** to scalar may occur (see logs; not silent in the sense of “wrong answer”—**path** may change).  
- User **must not** assume two backends return **identical** floats.

### 3.3 Execution timeout

`COELANOX_EXECUTION_TIMEOUT_MS` (default **30000** ms) can abort long runs. The runtime checks the deadline **between** execution steps (plan steps or scalar waves). On expiry:

| Result | Behaviour |
|--------|-----------|
| Timeout | Inference **aborts**; **error** returned to caller; **no** partial success guarantee. |

### 3.4 Resource limits

| Limit | Effect when exceeded |
|-------|----------------------|
| `COELANOX_MAX_MEMORY_BYTES` | Runtime may reject or fail allocation paths. |
| Workspace cap (`--max-workspace-mb` / env) | **Error** if the model cannot run within cap. |

---

## 4. Failure modes (operator-facing)

| Symptom | Cause class | Typical response |
|---------|-------------|------------------|
| Hash / integrity failed | Tampering, corruption | **Stop**; do not run; re-source artifact. |
| Version mismatch | Old/new toolchain mismatch | **Stop**; rebuild or align runtime version. |
| Package-time `Custom` op | Unsupported ONNX/op | **Fix graph** or extend translator (not a runtime patch). |
| Timeout | Slow or stuck work | **Abort**; tune timeout or workload. |
| Path rejected | `COELANOX_ALLOW_ABSOLUTE_PATHS=false` | **Error**; use allowed paths. |
| Backend missing | No CLF / wrong path | May **fall back** to scalar; **performance** drop, not silent correctness. |

Extended runbooks: [Operations](../OPERATIONS.md).

---

## 5. Audit and logging

- **`coelanox run --audit`** (and related options) can emit **per-op** evidence to configured outputs.  
- **`serve`** mode flushes audit state during long-lived sessions (see IPC / serve documentation).  
- Logging uses **tracing**; integrate with your log pipeline (`RUST_LOG`, `COELANOX_LOG_LEVEL`).

This specification does **not** enumerate every error string; it describes **classes** of behaviour security teams should expect.

---

## Related documents

- [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) — Guarantees and non-goals  
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) — Float, quantization, ops  
- [Operations](../OPERATIONS.md) — Env vars, runbooks, production checklist  

← [Non-technical hub](README.md)
