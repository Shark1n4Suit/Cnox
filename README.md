# COELANOX — Secure AI Container

<img src="assets/coelanox-coelanox_logo.png" alt="COELANOX" width="200"/>

**One artifact. Verified integrity. Deploy anywhere.**

COELANOX is the universal binary container and toolchain for AI inference. Package supported models (e.g. ONNX) into a single **`.cnox`** file: graph, optimized code, compressed weights, and a SHA-256 integrity hash. Verify before deploy; run the same file on target systems with no framework runtime or format lock-in.

---

## Why COELANOX

| Capability | Benefit |
|------------|---------|
| **Single artifact** | One `.cnox` file—no Python, no framework runtime, no dependency sprawl in production. |
| **Integrity by design** | SHA-256 over the full container. Standalone `verify` and mandatory verification before `run`. |
| **Portable execution** | Same container runs with optimized (SIMD/CLF) or scalar fallback; one build, multiple deployment scenarios. |
| **Auditability** | Deterministic packaging, hash verification, and optional audit logs for compliance and traceability. |

---

## Overview

- **Package** — Build a `.cnox` container from a source model (translator plugins; ONNX supported via packager).
- **Verify** — Standalone integrity check (SHA-256) before deployment.
- **Run** — Load and execute: verify, then infer. Optional benchmarking and diagnostics.

CLI commands: `package`, `validate`, `verify`, `run`, `info`, `benchmark`, `env`, `debug`. Full reference: [docs/README.md](docs/README.md) (architecture, operations, security, CLI).

---

## Architecture (summary)

```
┌─────────────────────────────────────────────────────────────┐
│  Container (.cnox)                                           │
│  Header (magic, version, SHA-256) │ Manifest │ IR, code, weights
├─────────────────────────────────────────────────────────────┤
│  Package:  Translate → Universal IR → Optimize → Compress → Hash
│  Runtime:  Verify → Load → Execute (SIMD/CLF or scalar fallback)
└─────────────────────────────────────────────────────────────┘
```

**Components:** `coelanox-core` (format, IR, security, scalar ops), `coelanox-packager` (translate, optimize, compress), `coelanox-runtime` (load, execute), `coelanox-cli` (commands).

---

## Security

- **Integrity** — SHA-256 over full container at package time; verified by `verify` and again before `run`. See [docs/SECURITY.md](docs/SECURITY.md).
- **Validation** — Path and format checks; configurable limits (container size, memory, timeouts). See [docs/OPERATIONS.md](docs/OPERATIONS.md).
- **Implementation** — Rust; memory-safe core with explicit validation at load and execution boundaries.

---

## Getting started

**Build (from repository root):**

```bash
cargo build --release
```

**CLI (run from workspace root or install):**

```bash
cargo run --release -- <command> ...
# or
cargo install --path coelanox-cli
coelanox <command> ...
```

**Typical workflow:** `coelanox package` (create `.cnox`) → `coelanox verify` (integrity check) → deploy → `coelanox run` (verify then execute on target).

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index (architecture, operations, security, CLI, formats) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, three-command model, backend and container layout |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Configuration, environment variables, runbooks, pre-deployment checklist |
| [docs/SECURITY.md](docs/SECURITY.md) | Security architecture, hash verification, best practices |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Full CLI command tree and options |
| [COELANOX_REFERENCE.md](COELANOX_REFERENCE.md) | Product and format reference |

---

## Platform and format

- **Platforms:** Linux, Windows, macOS (Rust toolchain).
- **Model format:** ONNX (opset 13) supported via packager translator; other formats via translator plugins. See [ONNX decomposition tree](coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) for ONNX → IR mapping.

---

## License

COELANOX is proprietary software. See [LICENSE](LICENSE) for terms.
