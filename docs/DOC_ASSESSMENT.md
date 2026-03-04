# COELANOX documentation assessment

Short assessment of current doc structure and content, and what would make it “set” or better.

---

## What’s in good shape

- **README.md** — Clear “what it is,” CLI summary, architecture diagram, component list. Good entry point.
- **ARCHITECTURE.md** — Detailed three-command model, security hash, backend manager, AOT selection, container layout, runtime flow. Strong for implementers.
- **OPERATIONS.md** — Production-focused: config loading, env vars, runbooks (verify failed, OOM, timeout), pre-deployment checklist.
- **CLI_REFERENCE.md** — Full command tree and options; useful for integration and scripting.
- **SECURITY.md** — Hash verification, container security, best practices, limitations. Good for security review.
- **FEATURES.md** — Implemented vs planned in one place; helps set expectations.
- **CNOX_INTERNALS.md** — Concise binary layout of `.cnox`; good for debugging and tooling.
- **ONNX_DECOMPOSITION_TREE.md** — Clear mapping of ONNX → IR (1:1, decomposed, control-flow error, Custom).
- **CLF_INTEGRATION.md** — How to add CLF (submodule/path/pub), API contract. Useful for contributors.

---

## Structure gaps

| Gap | Recommendation |
|-----|----------------|
| **No single doc index** | Add **docs/README.md** as the index (done). Link to it from the root README under “Documentation.” |
| **Packager/translator docs scattered** | Only ONNX has a dedicated doc under `coelanox-packager/docs/`. Either (a) add a short “Translators” section in FEATURES or ARCHITECTURE that links to ONNX_DECOMPOSITION_TREE and future format docs, or (b) add **docs/TRANSLATORS.md** that lists supported formats and points to format-specific pages. |
| **No “Contributing” path** | Add **CONTRIBUTING.md** at repo root (or **docs/CONTRIBUTING.md**) with: build/test commands, where to find ARCHITECTURE + FEATURES + CODE_REVIEW, how to add a translator or backend, link to CLF_INTEGRATION. |
| **Overlap between README, COELANOX_REFERENCE, ARCHITECTURE** | Keep all three; in README add one line: “For a quick reference see [COELANOX_REFERENCE](COELANOX_REFERENCE.md); for architecture see [docs/ARCHITECTURE](docs/ARCHITECTURE.md).” So “start here” vs “reference” vs “deep dive” is clear. |

---

## Content gaps

| Gap | Recommendation |
|-----|----------------|
| **FEATURES out of date** | FEATURES says “No default ONNX translator” and “ONNX translator is not registered in-tree.” Update to: “ONNX translator (opset 13) is available in coelanox-packager (feature `onnx-protobuf`); register it via plugins for packaging ONNX models.” |
| **No IR/OpType reference** | Optional: add **docs/IR_REFERENCE.md** (or a section in ARCHITECTURE) listing Universal IR OpTypes and how they map to CLF/backends. HAL_MAPPING and ONNX_DECOMPOSITION_TREE already give partial coverage. |
| **No “Development setup”** | CONTRIBUTING (above) can include: Rust version, `cargo build`, `cargo test`, optional CLF path, how to run CLI from workspace. |
| **Demo translators (BERT, ResNet)** | Optional: one-page **docs/TRANSLATORS.md** (or packager docs) describing BERT (BRT1 bundle) and ResNet-tiny-MNIST as demo translators and pointing to ONNX for the main production translator. |

---

## Summary: are we set?

- **For users/operators:** Yes, if they use **README** → **OPERATIONS** / **CLI_REFERENCE** / **SECURITY**. Adding **docs/README.md** and a “Documentation” link in the root README improves findability.
- **For developers/contributors:** Mostly. Add **CONTRIBUTING.md** and keep ARCHITECTURE + FEATURES + CLF_INTEGRATION + CODE_REVIEW as the main technical set. Optional: IR reference, Translators overview.
- **For product/investors:** COELANOX_REFERENCE + INVESTOR_READINESS + FEATURES are enough; no structural change needed.

**Minimal next steps to be “set”:**

1. Add **docs/README.md** (index) — done.  
2. In root **README.md**, add a “Documentation” subsection linking to `docs/README.md`.  
3. Update **FEATURES.md** to reflect ONNX translator (opset 13, plugin, feature flag).  
4. Add **CONTRIBUTING.md** (build, test, where to read, how to add translator/backend).

After that, optional improvements are IR reference, Translators overview, and keeping FEATURES/ARCHITECTURE in sync with new backends or formats.
