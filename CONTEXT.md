# CONTEXT.md

## Last Updated
2026-04-19

## What This Project Is
**COELANOX** (Coelanox) is a **universal binary container system for AI inference**—not a training stack. It packages supported source models into a single **`.cnox`** file: **Universal IR** (graph), **compressed weights**, optional **ahead-of-time / CLF execution-plan** blobs, and a **SHA-256** hash over the container (see `coelanox-core` format). Operators use the **`coelanox`** Rust CLI to **package**, **verify** integrity out of band, and **run** on the target without shipping Python or a full training framework in production. Optional **Ed25519 signing** and **per-op audit** logs support supply-chain and review workflows.

*Positioning (product):* the abstraction between math and physics.

## Stack
- **Language:** Rust (workspace under [`coelanox/Cargo.toml`](coelanox/Cargo.toml)).
- **Crates:** `coelanox-core`, `coelanox-clf`, `coelanox-executor`, `coelanox-packager`, `coelanox-orchestrator`, `coelanox-ipc`, `coelanox-cli`, `coelanox-standalone`; integration tests in `coelanox/tests`.
- **Representative deps:** `clap`, `serde`/`serde_json`, `sha2`, `zstd`/`lz4`, `tokio` (where async is used), `ed25519-dalek` (signing).

## Coelanox directory (`coelanox/`) — crates and folders

Summaries from crate roots and layout (see [`coelanox/Cargo.toml`](coelanox/Cargo.toml) for workspace members).

### Workspace root (`coelanox/`)
Rust **workspace manifest**: crate members, shared `[workspace.dependencies]`, release profile (e.g. LTO) for shipping binaries.

### `coelanox-core/`
**Shared foundation:** `.cnox` **container format** (header, manifest, SHA-256, optional Ed25519 flags), **Universal IR** and serialization, **scalar** execution path helpers, **optimization** hooks, **translator plugin API** (`CoelanoxTranslator` in `plugins`), memory/protection **HAL** types, and shared errors. Treated as the **frontend SDK** for model translators.

### `coelanox-clf/`
**CLF reader** (in-tree): parse `.clfc`/`.clf` headers and manifests, resolve **kernel blobs by `op_id`** per public CLF spec; op registry lines up with IR for packaging. **Read-only** here—CLF production/tooling lives outside this repo.

### `coelanox-executor/`
**CLF mini-executor:** consumes a **binary execution plan**, walks nodes, dispatches either **embedded machine code** (CLFE 6-argument ABI) or **linked SIMD symbols** (`clf/kernels/simd` on x86_64). Bridges IR/plan to runnable kernels.

### `coelanox-packager/`
**Compiler / packager:** **translators** (e.g. ONNX, BERT, ResNet) → Universal IR → optimize → compressed weights + optional **plan/CLF code section** → write **`.cnox`**. Orchestrates `coelanox-executor` plan build, `coelanox-clf` for blobs, and backend selection.

### `coelanox-orchestrator/`
**Load-time and run-time:** open `.cnox`, **verify** hash/signature, memory-map payload, route execution to **executor** (plan + blobs or symbols) or **scalar IR fallback**, optional **per-op audit**, container-embedded CLFMM/CLFMP or HAL backends, execution **deadlines**. Exposes `CoelanoxRuntime` and related config.

### `coelanox-ipc/`
**stdio IPC** for **`coelanox serve`:** framed binary protocol over stdin/stdout (`run_serve_stdio`, payload limits, version constant)—wraps orchestrator without changing the artifact format.

### `coelanox-cli/`
**`coelanox` CLI:** subcommands for **package**, **verify**, **run**, **info**, **benchmark**, **serve**, **keygen**, **extract**, env/config loading, audit and signing flags—user-facing entry to packager + orchestrator + IPC.

### `coelanox-standalone/`
**Thin embed API** and **`coelanox-run`** binary: load one container and infer (optional **`--serve`** via `coelanox-ipc`), with trust/verify options aligned to the main CLI—intended for minimal edge deployments.

### `tests/` (`coelanox-tests`)
**Integration test crate:** hashes, backends, AOT selection, integration and scalar paths, NVMe allocation, signing, translator SDK, hardening, runtime semantics regressions, etc. (see `tests/Cargo.toml` for `[[test]]` targets).

### Other directories under `coelanox/`
- **`docs/`** — empty in this tree; primary documentation lives at repo [`docs/`](docs/) (e.g. [`COELANOX_ONE_PAGER.md`](docs/COELANOX_ONE_PAGER.md)).
- **`models/`** — local **outputs** (`output/`) and **test inputs** (`test_models/`).
- **`scripts/`** — shell demos (e.g. BERT scalar vs SIMD, verify/run flows).
- **`tools/`** — empty placeholder.
- **`target/`** — Cargo **build artifacts** (local, not source).

## Current State
- **Container:** `.cnox` format, SHA-256, `verify`, `run`, manifest and validation at load/execute boundaries.
- **Import / translators:** ONNX (opset 13), BERT bundle, ResNet demo bundles—broader PyTorch/TF-style import is roadmap.
- **Execution:** **scalar / interpreted IR** path always available; **CLF / plan** path when packaged; `--fallback-only` builds omit embedded native code for portability.
- **Security / ops:** hash verification, optional Ed25519 (`keygen` / sign / verify), configurable limits (size, workspace, timeout, paths), optional `--audit` / `--audit-required` on containers.
- **Distribution:** CLI commands including `extract` for shipping a minimal runtime next to `.cnox` files; cross-arch `extract` called out in docs as planned.

## Active WIP
- Uncommitted work (see **Recent changes**): runtime/config/signing hardening, standalone parity with `coelanox run`, ONNX/LSTM robustness—ready to commit when you choose.

## Architecture Decisions
- **Single governed artifact:** one `.cnox` versions what shipped; CI/target can re-verify without executing.
- **Defense in depth:** package-time hash; `verify` for pre-deploy; `run` re-verifies by default unless explicitly overridden for development.
- **Scalar IR fallback:** portable correctness baseline and cross-environment runs when CLF/native blobs are absent or unusable (`--fallback-only`).
- **Evidence:** optional audit trails and Ed25519 provenance—complement to, not a substitute for, organizational compliance programs.

## Known Issues
- None tracked here (roadmap items such as cross-arch `extract` and broader translators are **gaps / roadmap**, not necessarily defects).

## Refactor Candidates
None yet

## Recent Changes
- 2026-04-19: **BERT + `clf/clfc/bert.clfc`:** Package with `COELANOX_CLF_PATH=/home/shark/cnox/clf` (parent of `clfc/`), `--use-demo-translator`, `--target cpu`. Runtime CLF backend name is the file stem: **`--backend bert`**. Example output: `models/output/bert_base_uncased_bert_clfc.cnox`.
- 2026-04-19: **Version 0.7.5:** workspace `Cargo.toml` crates; CLI `--version` and `validate`/`info` version lines driven by **`coelanox-cli` `CARGO_PKG_VERSION`** (0.7.5); tests, `CHANGELOG`, and doc examples (CLI_REFERENCE, CLI_MOCK_SPEC, issue templates, release workflow hint) aligned to **0.7.5**.
- 2026-04-13: **Version 0.7.4:** workspace `Cargo.toml` crates, CLI `#[command(version)]`, `env`/`info` version strings, tests, `CHANGELOG`, and doc examples (CLI_REFERENCE, OPERATIONS, CLI_MOCK_SPEC, issue templates, release workflow hint) aligned to **0.7.4**.
- 2026-04-13: **Docs:** added [`docs/INTEGRATION_OUTPUT.md`](docs/INTEGRATION_OUTPUT.md) (run JSON I/O, serve binary IPC, `coelanox-run` stdout, benchmark/debug outputs, post-processing floats to text). Linked from [`docs/README.md`](docs/README.md) and [`docs/non-technical/DATA_FLOW.md`](docs/non-technical/DATA_FLOW.md).
- 2026-04-13: **Review follow-up:** centralized `RuntimeConfig::from_env_with_optional_config_file` in `coelanox-orchestrator` (replaces duplicated CLI/standalone loaders); added `CoelanoxError::ContainerIntegrity` + `is_container_integrity_failure()` for stable CLI hints; `coelanox debug` gained `--trusted-key`, `--require-signature`, `--no-verify` (parity with `run`, including GDB subprocess args). Regression: unit test on integrity error kind. `cargo test` workspace passed.
- 2026-04-13: **Code review** (pre-follow-up): noted mutex/signing hardening plus items later fixed in **Review follow-up** (DRY config load, typed integrity errors, `debug` trust flags).
- 2026-04-12: Added per-crate and per-folder summaries for everything under [`coelanox/`](coelanox/) (from `lib.rs` / `main.rs` and workspace layout).
- 2026-04-12: Populated this file with product definition from `docs/COELANOX_ONE_PAGER.md`, `docs/REFERENCE.md`, and workspace layout; spot-checked `coelanox-core` header/hash/signing against docs (aligned).
