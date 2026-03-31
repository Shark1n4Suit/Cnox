# Frontend SDK Roadmap

**Status:** MVP Complete (Phases 1–3); Phase 4 deferred  
**Target:** Structured frontend contract for model translators (mirrors CLF backend contract)  
**Owner:** TBD

---

## Vision

The backend has a clear contract: CLF format + op_id registry. Vendors produce `.clfc` files; no SDK crate needed.

The frontend needs the same: a **structured contract** for translators. Output is always **UniversalIR** — no ad-hoc blobs. The Frontend SDK formalizes the trait, IR schema, and OpType vocabulary so third-party translators can build against a stable contract.

**Target architecture (Phase 4):** SDK lives in `coelanox-core`. Plugins are dynamic libraries (`.so`/`.dll`) that depend only on core, implement the trait, and use `export_translator!` to generate the C ABI. PluginManager discovers and loads them from `~/.coelanox/translators/`. No separate SDK crate.

---

## Current State

| Aspect | Status |
|--------|--------|
| **CoelanoxTranslator trait** | Renamed from TranslatorPlugin; slimmed (Phase 1 done) |
| **Translators** | Internal to packager (BERT, ONNX, ResNet); no third-party path |
| **Discovery** | None – all compiled in via `load_default_plugins()` |
| **SDK crate** | None – no `coelanox-translator-sdk` or formal frontend SDK |
| **Documentation** | `docs/sdk/frontend.md` — "How to build a translator"; [docs/sdk/README.md](../docs/sdk/README.md) index |
| **IR schema** | Documented under `docs/sdk/` as SDK output |

---

## Goals

1. **Stable contract** - Translators implement a minimal trait; output is always UniversalIR
2. **Structured output** - IR schema is the canonical output format (nodes, tensors, attributes)
3. **OpType vocabulary** - Canonical set of ops translators may emit; documented in SDK
4. **Pluggability** - Cargo features first; dynamic plugins (Option C) as target: SDK in core, `export_translator!` macro, plugins attach via one dependency
5. **Symmetry with backend** - Frontend SDK mirrors CLF's role: format + contract

---

## Phase 1: Contract Cleanup (1–2 weeks)

### 1.1 Slim the CoelanoxTranslator trait ✅ DONE

**Location:** `coelanox-core/src/plugins.rs`

**Trait renamed:** `TranslatorPlugin` → `CoelanoxTranslator`

**Required methods (keep):**
- `translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>`
- `translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR>`
- `translate_from_path_with_weights(&self, input_path: &Path, external_weights_path: Option<&Path>) -> CoelanoxResult<UniversalIR>`
- `extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>`
- `supported_formats(&self) -> Vec<ModelFormat>`
- `name(&self) -> &str`

**Remove or make optional (unused):**
- `configure`, `validate_config`, `PluginConfig`, `PluginMetadata`, `PluginCapabilities`
- `get_plugin_metadata`, `get_plugin_capabilities`, `configure_plugin`, `validate_all_plugins` (PluginManager)

**Tasks:**
- [x] Audit all usages of `configure`, `validate_config`, `PluginMetadata`, `PluginCapabilities`
- [x] Remove or deprecate unused trait methods
- [x] Update all in-tree translators (BERT, ONNX, ResNet)
- [x] Update PluginManager to match

### 1.2 Treat coelanox-core as the Frontend SDK ✅ DONE

**Decision:** No separate `coelanox-translator-sdk` crate. `coelanox-core` is the public contract.

**Rationale:**
- Translators already depend on core for UniversalIR, OpType, Shape, DataType, etc.
- Single source of truth; semver on core = translator API version
- Re-export layer adds nothing unless core grows significantly

**Tasks:**
- [x] Document `coelanox-core` as the Frontend SDK in README / docs
- [x] Add `#[doc = "Stable API for translator plugins"]` to `plugins` and `ir` modules

---

## Phase 2: Documentation (1 week)

### 2.1 Create "How to Build a Translator" guide ✅ DONE

**Location:** `docs/sdk/frontend.md` (see [sdk/README.md](../docs/sdk/README.md))

**Contents:**
- [x] Trait contract (CoelanoxTranslator)
- [x] IR output schema (UniversalIR structure)
- [x] OpType vocabulary (canonical ops; Custom(String) for extensions)
- [x] Builder API: `add_node`, `add_node_input`, `add_node_output`, `add_weight_tensor_with_data`, etc.
- [x] ModelFormat: `ModelFormat::Custom(String)` for new formats
- [x] Example: minimal translator skeleton
- [x] Versioning: pin to `coelanox-core` semver

### 2.2 IR schema documentation ✅ DONE

**Location:** `docs/sdk/` (see [sdk/README.md](../docs/sdk/README.md))

**Contents:**
- [x] UniversalIR fields
- [x] GraphNode, OpType, Attribute
- [x] TensorId, NodeId
- [x] Topological order requirement
- [x] Valid attribute keys per OpType (where known)

---

## Phase 3: Cargo Features (1 week)

### 3.1 Optional translators per feature ✅ DONE

**Goal:** Smaller packager binary; users opt-in to translators they need.

**Location:** `coelanox-packager/Cargo.toml`

**Tasks:**
- [x] Add features: `translator-onnx`, `translator-bert`, `translator-resnet`
- [x] Default: `translator-onnx`, `translator-bert`, `translator-resnet` (all enabled)
- [x] Gate each translator in `load_default_plugins()` via `#[cfg(feature = "...")]`
- [x] Conditional imports in `plugins/mod.rs` to avoid unused-import warnings
- [x] Document: `cargo build --no-default-features --features translator-onnx`

---

## Phase 4: Dynamic Plugin Architecture (Option C)

### 4.1 Architecture: SDK in core, plugins attach via trait + macro

**Goal:** Dynamic discovery from `~/.coelanox/translators/` or `COELANOX_TRANSLATOR_PATH`. Plugins are `.so`/`.dll`; one dependency (`coelanox-core`).

```
coelanox-core                    ← SDK (trait + IR types + export_translator! macro)
    ↑
    │ depends on
    │
translator .so / .dll            ← impl trait, export_translator!(MyTranslator)
    ↑
    │ loaded by (libloading)
    │
PluginManager (in packager)       ← scans translators/, dlopen, calls ABI
```

**No separate SDK crate.** Everything lives in `coelanox-core`:
- `plugins.rs` — trait, `TranslatorError`, `export_translator!` macro
- `ir.rs` — UniversalIR, OpType, etc.

**Third-party translator:**
```toml
[dependencies]
coelanox-core = "1.0"   # One dependency
```

```rust
impl CoelanoxTranslator for MyTranslator { ... }
export_translator!(MyTranslator);  // Macro generates all ABI exports
```

### 4.2 export_translator! macro (attachment point)

The macro bridges Rust trait ↔ C ABI. Plugin author implements trait; macro generates C-callable wrappers.

**Generated exports:**
- `translator_create()` → `*mut c_void` (handle)
- `translator_free(handle)`
- `translator_translate(handle, data_ptr, data_len, out_ptr, out_len)` → status
- `translator_name(handle)` → `*const c_char`
- `translator_supported_formats(handle)` → format list
- `translator_extract_weights(handle, ...)` (if needed)

### 4.3 Serialization boundary

No shared struct layout across ABI. Data crosses as bytes.

**Flow:**
```
Host: model file → bytes
Plugin: bytes → parse → UniversalIR → serialize (bincode/serde) → bytes
Host: bytes → deserialize → UniversalIR
```

- **Input:** Host passes model bytes.
- **Output:** Plugin returns bytes (serialized `UniversalIR`).
- Both use `coelanox-core` serde; same version ⇒ compatible format.

### 4.4 TranslatorError (ABI-safe errors)

`CoelanoxError` cannot cross ABI. Use `#[repr(C)]` struct:

```rust
#[repr(C)]
pub struct TranslatorError {
    pub code: u32,       // 0=ok, 1=parse, 2=unsupported, etc.
    pub message_len: u32,
    // message as UTF-8 bytes; host copies and maps to CoelanoxError
}
```

Or `(code, *const c_char, len)`. Host maps to `CoelanoxError::Translation(...)`.

### 4.5 Version contract

Plugin and host must agree on `coelanox-core` (same major version). Optional handshake at load: `translator_api_version() -> u32` to reject mismatched plugins.

### 4.6 Tasks

- [ ] Add `TranslatorError` to `coelanox-core`
- [ ] Implement `export_translator!` macro in `coelanox-core`
- [ ] Add `libloading` to packager
- [ ] Implement discovery in PluginManager (scan `~/.coelanox/translators/`, `COELANOX_TRANSLATOR_PATH`)
- [ ] Build ONNX (or BERT) as separate `cdylib` crate as proof of concept
- [ ] Document plugin build: `crate-type = ["cdylib"]`, `export_translator!` usage

**Defer:** Not required for MVP; Cargo features (Phase 3) suffice initially.

---

## Phase 5: Ecosystem & Validation (Ongoing)

### 5.1 Third-party translator support

**Tasks:**
- [ ] Publish `coelanox-core` to crates.io (if not already)
- [x] Ensure translator authors can depend on `coelanox-core` without packager (tested in `test_translator_sdk.rs`)
- [ ] Add `#[doc(hidden)]` or `pub(crate)` for internal types translators should not use (audit when needed)

### 5.2 Validation

**Tasks:**
- [x] IR validation: `validate()` before use (exists in `UniversalIR::validate()`)
- [ ] Optional: schema version in serialized IR for forward compatibility
- [x] Test: external crate implements CoelanoxTranslator (`test_translator_sdk.rs`)

---

## Contract Summary

| Contract Element | Backend (CLF) | Frontend (SDK) |
|------------------|---------------|----------------|
| **Format** | .clfc (manifest + blobs) | UniversalIR (graph) |
| **Canonical set** | op_id table | OpType enum |
| **Contract crate** | coelanox-clf | coelanox-core |
| **Output** | Blobs per op_id | IR graph (nodes, tensors) |
| **Discovery** | Path-based (.clfc) | Cargo features; path-based (.so/.dll) in Phase 4 |
| **Attachment** | N/A (format only) | `export_translator!` macro in core |

---

## Success Criteria

- [x] CoelanoxTranslator trait is minimal and stable
- [x] "How to build a translator" doc exists
- [x] IR schema documented as SDK output
- [x] Cargo features allow smaller packager binary
- [x] Third-party translator can depend on `coelanox-core` and implement the trait
- [x] No breaking changes to in-tree translators
- [ ] (Phase 4) Dynamic plugins: `export_translator!` macro, serialization boundary, TranslatorError, discovery from path

---

## Timeline (Estimate)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Contract | 1–2 weeks | None |
| Phase 2: Docs | 1 week | Phase 1 |
| Phase 3: Features | 1 week | Phase 1 |
| Phase 4: Discovery | 2–4 weeks | Phase 1–2 |
| Phase 5: Ecosystem | Ongoing | Phase 1–2 |

**MVP:** Phases 1–3 (4–5 weeks)

---

## References

- `coelanox-core/src/plugins.rs` — CoelanoxTranslator trait
- `coelanox-core/src/ir.rs` — UniversalIR, OpType
- `coelanox-packager/src/plugins/mod.rs` — PluginManager
- `coelanox-packager/src/translators/` — In-tree translators
- `docs/REFERENCE.md` — Model input / plugin system
- `roadmaps/PRODUCTION_MVP_ROADMAP.md` — Frontend SDK overview (Section 1797+)
