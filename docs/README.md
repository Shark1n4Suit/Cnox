# COELANOX Documentation

This index is the entry point for all project documentation. Documents are grouped by audience and topic.

---

## Getting started

| Document | Audience | Description |
|----------|----------|-------------|
| [README](../README.md) | All users | Product overview, CLI summary, architecture diagram, quick start |
| [INSTALL.md](INSTALL.md) | End users | Install from a release (download, extract, PATH); no Rust required |
| [COELANOX_REFERENCE](../COELANOX_REFERENCE.md) | Product / strategy | High-level reference: COELANOX value proposition, `.cnox` format, strategy |
| [FOUNDER_NOTE](../FOUNDER_NOTE.md) | Community / investors | Origin story and vision (optional reading) |

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
| [OPERATIONS.md](OPERATIONS.md) | Configuration, environment variables, deployment targets, runbooks, pre-deployment checklist |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | CLI command tree, options, and example invocations |
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

| Document | Description |
|----------|-------------|
| [FRONTEND_SDK.md](FRONTEND_SDK.md) | **Frontend SDK** — How to build a translator. `coelanox-core` is the SDK: implement `CoelanoxTranslator`, produce UniversalIR. Trait contract, IR schema, OpType vocabulary, builder API. |
| [ONNX decomposition tree](../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) | ONNX opset 13 → IR: 1:1 mapping, control-flow handling, decomposed ops, Custom |

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
| [FRONTEND_SDK_ROADMAP](../roadmaps/FRONTEND_SDK_ROADMAP.md) | Frontend SDK: translator contract, Cargo features, dynamic plugins (Phase 4) |
| [coelanox-constellation-roadmap](../roadmaps/coelanox-constellation-roadmap.md) | Constellation roadmap |

---

## Demo and developer tools

| Document | Description |
|----------|-------------|
| [DEMO_RESULTS.md](DEMO_RESULTS.md) | Demo run results and reproduction steps |
| [dev/CLI_MOCK_SPEC.md](dev/CLI_MOCK_SPEC.md) | Mock CLI spec for website/frontend terminal (parsing rules, sample outputs) |
