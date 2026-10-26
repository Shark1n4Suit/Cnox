# COELANOX — Non-technical reader hub

This folder collects **procurement-ready** and **security-review** documents in one place. For engineering depth, see the parent [documentation index](../README.md).

---

## Core documents (start here)

| Document | Purpose |
|----------|---------|
| [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) | What COELANOX is, **guarantees**, **non-goals**, supported model types, hardware today, and the **“tells you when you’re wrong”** stance (~2–3 pages). |
| [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md) | Runtime **contract**: determinism, float equivalence, **failure modes**, timeouts, verify/load/run behaviour. |
| [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md) | Customer-facing **ONNX opset 13** summary: what compiles, what is rejected, what to do if yours does not (full detail links to the decomposition tree). |
| [DATA_FLOW.md](DATA_FLOW.md) | **Data flow**: build → verify → run/serve; what touches data; what **persists**; **airgapped** notes. |
| [ONBOARDING.md](ONBOARDING.md) | Step-by-step **model → `.cnox` → run** without vendor SSH. |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | **Honest limits**: quantization, dynamic shapes, ONNX gaps, scalar performance, service features. |

---

## Short overviews (still useful)

| Document | Who it’s for |
|----------|----------------|
| [COELANOX overview (one page)](../COELANOX_ONE_PAGER.md) | Executive / first skim |
| [Quickstart](../QUICKSTART.md) | Fastest path to a working command sequence |
| [Reference](../REFERENCE.md) | What a `.cnox` contains + high-level implementation status |

---

## Technical references (linked from the docs above)

| Topic | Location |
|-------|----------|
| Full ONNX → IR mapping | [ONNX decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) |
| IR versioning contract | [IR_SEMANTICS.md](../IR_SEMANTICS.md) |
| Operations, env vars, runbooks | [OPERATIONS.md](../OPERATIONS.md) |
| System architecture | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| CLI | [CLI_REFERENCE.md](../CLI_REFERENCE.md) |
| SDK / translators | [sdk/README.md](../sdk/README.md) |

---

## If you need an engineer

| Need | Go to |
|------|--------|
| Full command reference | [CLI_REFERENCE.md](../CLI_REFERENCE.md) |
| Developers | [DEVELOPER.md](../DEVELOPER.md) |
| API / Rust crates | `cd coelanox && cargo doc --no-deps --open` |

---

← [Documentation home](../README.md)
