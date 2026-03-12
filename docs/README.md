# COELANOX Documentation

This index is the entry point for all project documentation. Documents are grouped by audience and topic.

---

## Getting started

| Document | Audience | Description |
|----------|----------|-------------|
| [README](../README.md) | All users | Product overview, CLI summary, architecture diagram, quick start |
| [INSTALL.md](INSTALL.md) | End users | Install from a release (download, extract, PATH); no Rust required |
| [COELANOX_REFERENCE](../COELANOX_REFERENCE.md) | Product / strategy | High-level reference: COELANOX value proposition, `.cnox` format, strategy |

---

## Architecture and design

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Three-command model, security hash, backend manager, AOT selection, container layout, runtime flow |
| [CNOX_INTERNALS.md](CNOX_INTERNALS.md) | Binary layout of `.cnox` (header, manifest, code, weights, IR, NVMe backing) |
| [HAL_MAPPING.md](HAL_MAPPING.md) | OpType ↔ CLF ↔ Memory HAL ↔ Protection HAL mapping |

---

## Operations and deployment

| Document | Description |
|----------|-------------|
| [OPERATIONS.md](OPERATIONS.md) | Configuration, environment variables, runbooks, pre-deployment checklist |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | CLI command tree, options, and example invocations |
| [CLI_MOCK_SPEC.md](CLI_MOCK_SPEC.md) | Mock CLI specification and sample outputs for integration and tooling |
| [SECURITY.md](SECURITY.md) | Security architecture, hash verification, container security, best practices, limitations |

---

## Features and capabilities

| Document | Description |
|----------|-------------|
| [FEATURES.md](FEATURES.md) | Implemented and planned features: container, CLI, execution backends, IR, security, config, observability |

---

## Integrations and development

| Document | Description |
|----------|-------------|
| [CLF_INTEGRATION.md](CLF_INTEGRATION.md) | Integrating the CLF crate (submodule, path dependency, or published); required API surface |

---

## Model formats and translators

| Document | Location | Description |
|----------|----------|-------------|
| [ONNX decomposition tree](../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) | `coelanox/coelanox-packager/docs/` | ONNX opset 13 → IR: 1:1 mapping, control-flow handling, decomposed ops, Custom |

---

## Version history

| Document | Description |
|----------|-------------|
| [CHANGELOG](../CHANGELOG.md) | Version history and notable changes |

---

## Roadmaps

| Document | Description |
|----------|-------------|
| [PRODUCTION_MVP_ROADMAP](../roadmaps/PRODUCTION_MVP_ROADMAP.md) | Production MVP roadmap |
| [coelanox-constellation-roadmap](../roadmaps/coelanox-constellation-roadmap.md) | Constellation roadmap |

---

## Deployment targets

| Document | Description |
|----------|-------------|
| [SERVER_EDGE_EMBEDDED_ANALYSIS.md](SERVER_EDGE_EMBEDDED_ANALYSIS.md) | Server (Arc cache, eviction), edge, and embedded deployment; warm cache, weight sharing |
| [DEMO_RESULTS.md](DEMO_RESULTS.md) | Demo run results and reproduction steps |
