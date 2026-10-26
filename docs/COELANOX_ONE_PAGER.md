# COELANOX — Secure AI Container

**One artifact. Verified integrity. Deploy anywhere.**

---

## Executive summary

COELANOX packages AI models into a single **`.cnox`** binary container: **graph (Universal IR)**, **compressed weights**, optional **ahead-of-time machine code** (CLF), and a **SHA-256** hash over the whole file. Operators use one **`coelanox`** CLI to **package**, **verify integrity** independently of execution, and **run** inference on the target—without shipping Python or a full training framework in production. Optional **per-op audit logs** and **Ed25519 signing** support security reviews and supply-chain posture.

**Built in Rust** (`coelanox-core`, `coelanox-packager`, `coelanox-orchestrator`, `coelanox-cli`) with explicit validation at load and execution boundaries.

---

## The problem

| Pain | What goes wrong |
|------|------------------|
| **Runtime sprawl** | ONNX Runtime, vendor SDKs, and glue code multiply; upgrades and CVEs hit different layers. |
| **“What did we ship?”** | Models and configs drift; reproducing a production inference stack is hard. |
| **Integrity** | Files on disk or in CI can change; teams need a **single checkable artifact** and a **pre-deploy verify** step. |
| **Evidence** | Regulated and safety-adjacent programs (automotive, industrial, aerospace supply chains) need **traceability**—not only accuracy, but **what ran** and **that it matched what was approved**. |

COELANOX targets teams that want **one governed deployable unit**: package once, verify anywhere, run with a defined execution path—including a **deterministic scalar fallback** when native code is not embedded or not usable.

---

## What COELANOX is

**COELANOX** is a **universal binary container system** for inference—not a training platform. It turns supported source models into **`.cnox`** files through **translator plugins** (e.g. ONNX, BERT bundle, ResNet demo bundle), a **Universal IR**, optimization, compression, and hashing.

**End-to-end pipeline:**

`Source model → Translator → Universal IR → Optimize → Weights + optional code → .cnox → Runtime (verify → load → execute)`

The runtime can execute via **embedded plan / CLF code** when present, or fall back to **interpreted scalar execution** over the IR—useful for portability, correctness baselines, and environments where native blobs are not shipped.

---

## How operators use it (core loop)

| Phase | Command | Purpose |
|-------|---------|---------|
| **Build** | `coelanox package` | Produce `.cnox` from a model; embed or omit backend code (`--fallback-only` for scalar-only containers). |
| **Assure** | `coelanox verify` | Standalone SHA-256 integrity check—use in CI/CD or before copying to an air-gapped host. |
| **Execute** | `coelanox run` | Load container, **verify hash again** (unless explicitly overridden for dev), run inference. |
| **Inspect** | `coelanox info` | Manifest, shapes, workspace needs, sizes. |
| **Measure** | `coelanox benchmark` | Latency and throughput on a chosen backend. |
| **Serve** | `coelanox serve` | Long-lived process: framed **binary IPC** on stdin/stdout for repeated inference (see `docs/CLI_REFERENCE.md` §7). |

Additional commands include **`validate`**, **`keygen`** (Ed25519 keypair for signing), **`env`** (capabilities), **`uninstall`** (removal instructions and optional local audit cleanup), **`debug`**, and **`extract`** (copy the full CLI or minimal **`coelanox-run`**—including **`--serve`** IPC—next to `.cnox` files for deployment).

**Typical workflow:** `package` → `verify` → ship `.cnox` + `coelanox` (or extracted runtime) → `run` on target.

---

## What’s inside a `.cnox`

| Section | Role |
|---------|------|
| **Header** | Magic, version, flags, **SHA-256** over the container |
| **Manifest** | JSON: model name, input/output shapes, backend metadata |
| **Machine code** | Optional CLF / execution-plan blobs; **empty** for fallback-only builds |
| **Weights** | Zstd/LZ4-compressed or uncompressed |
| **IR** | Serialized optimized graph for scalar / interpreted execution |

This structure keeps **integrity**, **metadata**, and **execution artifacts** in one file that can be versioned and signed.

---

## Security & governance (summary)

| Layer | Behavior |
|-------|----------|
| **Integrity** | Full-container SHA-256 at package time; **`verify`** for pre-deployment; **`run`** verifies again by default (defense in depth). |
| **Signing** | Optional **Ed25519**: generate keys with `keygen`, sign at package time, verify at run with a **trusted public key**; `--require-signature` can enforce provenance. |
| **Audit** | `run --audit` with optional `--audit-output` writes per-op evidence; containers can be built with **`--audit-required`** so audit is always on. |
| **Limits** | Configurable max container size, input size, workspace, execution timeout, path policies—via env vars or JSON config for production hardening. |

COELANOX is **not** a substitute for a full compliance program (e.g. EU AI Act); it strengthens **artifact integrity**, **repeatable execution**, and **technical evidence** for what was deployed and how it ran.

---

## Fit today & portability

- **Model import:** ONNX (opset 13), BERT bundle (`bert_base_uncased`), ResNet-tiny-MNIST demo bundles—via translator plugins. Broader framework import is on the roadmap.
- **Execution:** **Scalar IR** (always available) plus **CLF / plan** path when packaged; **`--fallback-only`** produces containers with **no embedded native code**—ideal when the same `.cnox` must run across OS/arch with only the scalar path (verify + IR execution).
- **Distribution:** Prebuilt CLI/runtime archives for common platforms; developers can also build from the Rust workspace.

---

## Roadmap (high level)

- More translators (generic PyTorch/TF paths where feasible), graph fusion, broader op coverage  
- Deeper hardware integration (vendor SDKs, reference backends)  
- Cross-arch **`extract`** and richer deployment packaging  
- Optional encryption and stronger code-signing workflows  





## One-line positioning

*Coelanox the abstraction between math and physics*

---

*CLI overview: `coelanox --help` · Documentation index: `docs/README.md`*
