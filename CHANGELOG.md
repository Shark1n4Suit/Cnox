# Changelog

All notable changes to COELANOX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project does not yet use semantic versioning for all artifacts; version numbers in the CLI and packages may differ.

---

## [Unreleased]

### Added

### Changed

- **`coelanox extract`:** Default output name is **`coelanox`** (full CLI copy, includes `serve`/IPC). New **`--minimal`** copies sibling **`coelanox-run`** for smaller deployments. Release CI builds and ships both binaries.
- **`coelanox-run`:** **`--serve`** mode — same framed stdin/stdout IPC v1 as **`coelanox serve`**.

### Fixed

---

## [0.7.5] — 2026-04-19

### Changed

- Workspace crates and CLI **`coelanox -V` / `--version`** aligned to **0.7.5**. `validate` / `info` text lines that report CLI/format version use the same crate version; **`coelanox env`** prints that CLI version first. Documentation examples, issue templates, mock CLI spec, and release workflow hints use **v0.7.5** / **0.7.5** where a concrete version is shown.

---

## [0.7.4] — 2026-04-13

### Changed

- Workspace crates and CLI **`coelanox -V` / `--version`** aligned to **0.7.4**. Documentation examples, issue templates, mock CLI spec, and release workflow hints now use **v0.7.4** / **0.7.4** where a concrete version is shown.

---

## [0.7.0] — 2026-03-28

### Added

- **`coelanox serve`**: long-lived process with framed binary IPC on stdin/stdout (`CREQ` / `CROk` / `CREr` / `CNXT`), implemented in the **`coelanox-ipc`** crate. Container format unchanged; IPC wraps the runtime only.
- **`coelanox uninstall`**: prints removal steps for the installed CLI and optional `--purge-audit` to delete `./coelanox_audit` in the current working directory.

### Changed

- Workspace crates aligned to **0.7.0**; CLI reports **`0.7.0`**. Logging for `serve` uses **stderr only** so stdout stays binary-clean for IPC frames.

### Fixed

- CPU vendor detection in `coelanox env` (stable `__cpuid` on x86_64); CPU model/RAM via **sysinfo** on supported platforms.

---

## [0.6.0-beta] and earlier

- Initial beta: container format (`.cnox`), package/verify/run CLI, scalar and CLF execution paths, backend discovery, ONNX translator, security hash verification. See [docs/REFERENCE](docs/REFERENCE.md) and [roadmaps/PRODUCTION_MVP_ROADMAP](roadmaps/PRODUCTION_MVP_ROADMAP.md) for details.
