# UniversalIR semantics (v1 contract)

COELANOX execution is defined by **UniversalIR** (`coelanox_core::ir`), not by ONNX or any single frontend. Frontends lower source formats into this IR; all backends (scalar reference, SIMD/CLF, future paths) must implement the **same** behavior for each `OpType` + attributes.

## Versioning

- **`ir_semantics_version`** in the container manifest matches **`COELANOX_IR_SEMANTICS_VERSION`** in `coelanox_core::format`.
- Runtime loaders call `validate_ir_semantics_version()`; mismatches fail with a clear error so old/new toolchains cannot silently mix.

## Packaging fingerprint

`manifest.packaging_fingerprint` records **how** the container was built (translator name/version, demo mode, `fallback_only`, target string, optimization level, packager crate version). It does not change math; it is for support and audit.

## Per-op reference

For a concrete, test-backed list of scalar behavior, see **`coelanox/tests/test_scalar_ops_correctness.rs`** (one test per primitive op). SIMD backends should match scalar outputs within documented tolerances where applicable.

## Design rule

- **IR is the contract.** ONNX is an import path only.
- If a frontend emits IR that scalar cannot execute, the bug is in the **frontend lowering** or the IR spec, not in “ONNX parity” as an independent goal.

← [Developer hub](DEVELOPER.md)
