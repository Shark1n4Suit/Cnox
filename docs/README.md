# COELANOX documentation index

Use this as the entry point for all project documentation.

---

## Start here

| Doc | Audience | Description |
|-----|----------|-------------|
| [../README.md](../README.md) | Everyone | What COELANOX is, CLI summary, architecture diagram, quick start |
| [COELANOX_REFERENCE.md](../COELANOX_REFERENCE.md) | Product / high-level | Reference: what COELANOX is, what a `.cnox` file contains, strategy |

---

## Architecture & design

| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Three-command model, security hash, backend manager, AOT selection, container layout, runtime flow |
| [CNOX_INTERNALS.md](CNOX_INTERNALS.md) | Binary layout of `.cnox` (header, manifest, code, weights, IR, NVMe backing) |
| [HAL_MAPPING.md](HAL_MAPPING.md) | OpType ↔ CLF ↔ Memory HAL ↔ Protection HAL mapping |

---

## Operations & deployment

| Doc | Description |
|-----|-------------|
| [OPERATIONS.md](OPERATIONS.md) | Production runbooks: config loading, env vars, verification failed, OOM, timeout, pre-deployment checklist |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Full CLI command tree, options, and example invocations |
| [CLI_MOCK_SPEC.md](CLI_MOCK_SPEC.md) | Mock CLI spec and sample outputs (e.g. for website) |
| [SECURITY.md](SECURITY.md) | Security architecture, hash verification, container security, best practices, limitations |

---

## Features & capabilities

| Doc | Description |
|-----|-------------|
| [FEATURES.md](FEATURES.md) | Implemented vs planned: container, CLI, execution backends, IR, security, config, observability, codegen, storage, platform |

---

## Integrations & development

| Doc | Description |
|-----|-------------|
| [CLF_INTEGRATION.md](CLF_INTEGRATION.md) | How to add the CLF crate (submodule, path dep, published), API your implementation must provide |
| [CODE_REVIEW_PHASE1.md](CODE_REVIEW_PHASE1.md) | Phase 1 code review: security hash, backend manager, AOT selection, CLI |

---

## Model formats & translators

| Doc | Location | Description |
|-----|----------|-------------|
| [ONNX decomposition tree](../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) | `coelanox/coelanox-packager/docs/` | ONNX opset 13 → IR: 1:1 map, control-flow error, decomposed ops, Custom |

*(Other format docs — e.g. BERT/ResNet demo translators — can be added here.)*

---

## Roadmaps & business

| Doc | Description |
|-----|-------------|
| [PRODUCTION_MVP_ROADMAP.md](../roadmaps/PRODUCTION_MVP_ROADMAP.md) | Production MVP roadmap |
| [coelanox-constellation-roadmap.md](../roadmaps/coelanox-constellation-roadmap.md) | Constellation roadmap |
| [INVESTOR_READINESS.md](INVESTOR_READINESS.md) | Proof points, risk analysis, one-line pitch |

---

## Doc health & gaps

See [DOC_ASSESSMENT.md](DOC_ASSESSMENT.md) for a short assessment of doc structure, content, and suggested improvements.
