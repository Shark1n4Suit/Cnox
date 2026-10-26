# COELANOX Documentation

| Document | Description |
|----------|-------------|
| **[non-technical/](non-technical/README.md)** | **Procurement & security pack** — product brief, runtime spec, ONNX customer ops, data flow, onboarding, known limitations |
| [QUICKSTART](QUICKSTART.md) | **Start here** — Install, package → verify → run in 5 minutes |
| [INTEGRATION_OUTPUT](INTEGRATION_OUTPUT.md) | **Integration** — How run/serve/standalone expose outputs (JSON, binary IPC, stdout) and how apps turn floats into text |
| [REFERENCE](REFERENCE.md) | What COELANOX is, `.cnox` format, implementation status |
| [ARCHITECTURE](ARCHITECTURE.md) | Three-command model, security hash, backend manager, codebase map, binary layout |
| [OPERATIONS](OPERATIONS.md) | Config, security, env vars, deployment, runbooks, FAQ |
| [CLI_REFERENCE](CLI_REFERENCE.md) | Full command tree and options |
| [DEVELOPER](DEVELOPER.md) | Hub → **[sdk/](sdk/README.md)** (translators, pipeline, CLF, glossary, troubleshooting) |
| [IR_SEMANTICS](IR_SEMANTICS.md) | UniversalIR v1 contract, semantics versioning, packaging fingerprint |
| [sdk/](sdk/README.md) | **SDK** — [frontend](sdk/frontend.md), [pipeline](sdk/pipeline.md), [clf](sdk/clf.md), [glossary](sdk/glossary.md), [troubleshooting](sdk/troubleshooting.md) |
| [dev/CLI_MOCK_SPEC](dev/CLI_MOCK_SPEC.md) | Mock CLI spec for website/terminal |

**API docs:** `cd coelanox && cargo doc --no-deps --open`  
**Roadmaps:** [roadmaps/](../roadmaps/)  
**Changelog:** [CHANGELOG.md](../CHANGELOG.md)
