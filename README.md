# COELANOX — Secure AI Container

<img src="assets/coelanox-coelanox_logo.png" alt="COELANOX" width="200"/>

**One artifact. Verified integrity. Deploy anywhere.**

COELANOX is the universal binary container and toolchain for AI inference. Package supported models (e.g. ONNX) into a single **`.cnox`** file: graph, optimized code, compressed weights, and a SHA-256 integrity hash. Verify before deployment; run the same file on target systems with no framework runtime or format lock-in.

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

CLI commands: `package`, `validate`, `keygen`, `verify`, `run`, `info`, `benchmark`, `env`, `debug`, `extract`. See the [documentation index](docs/README.md) for architecture, operations, security, and full CLI reference.

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

**Components:** `coelanox-core` (format, IR, scalar ops), `coelanox-packager` (translate, optimize), `coelanox-orchestrator` (load, execute), `coelanox-cli` (commands). See [docs/DEVELOPER](docs/DEVELOPER.md) to build a translator.

---

## Security

- **Integrity** — SHA-256 over full container at package time; verified by `verify` and again before `run`. See [docs/OPERATIONS](docs/OPERATIONS.md).
- **Validation** — Path and format checks; configurable limits (container size, memory, timeouts). See [docs/OPERATIONS.md](docs/OPERATIONS.md).
- **Implementation** — Rust; memory-safe core with explicit validation at load and execution boundaries.

---

## Getting started

**Prerequisites:** Rust toolchain (stable). Build from the `coelanox/` workspace.

**Build:**

```bash
cd coelanox
cargo build --release -p coelanox-cli
```

**Run the CLI** (from `coelanox/` or install globally):

```bash
cargo run --release -p coelanox-cli -- <command> ...
# or
cargo install --path coelanox/coelanox-cli
coelanox <command> ...
```

**Typical workflow:** `coelanox package` (create `.cnox`) → `coelanox verify` (integrity check) → deploy → `coelanox run` (verify then execute on target).

### Install from release (no Rust required)

1. Open [Releases](https://github.com/coelanox-ai/coelanox/releases) and download the archive for your platform (`coelanox-linux-x86_64.tar.gz` or `coelanox-windows-x86_64.zip`).
2. Extract the archive. **Linux/macOS:** From the extracted folder run `./install.sh` to install into `/usr/local/bin`, or add that folder to your `PATH`. **Windows:** Add the folder containing `coelanox.exe` to your `PATH`.
3. Run `coelanox --help`. See [docs/QUICKSTART](docs/QUICKSTART.md) for details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | **Start here** — Install, package → verify → run |
| [docs/REFERENCE.md](docs/REFERENCE.md) | Product summary, `.cnox` format, status |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, codebase map, binary layout |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Config, security, runbooks |
| [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | CLI commands and options |
| [docs/DEVELOPER.md](docs/DEVELOPER.md) | Translators, CLF integration |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Platform and format

- **Platforms:** Linux (x86_64), Windows (x86_64). Build from source for macOS.
- **Model format:** ONNX (opset 13) supported via packager translator; other formats via translator plugins. See [ONNX decomposition tree](coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) for ONNX → IR mapping.

---

## License

Proprietary. See [LICENSE](LICENSE) for terms.
