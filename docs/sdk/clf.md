# CLF — compute backends and the `coelanox-clf` crate

**CLF** (Coelanox Library File) is the on-disk / embedded format for **kernel blobs** keyed by **`op_id`**. The **`coelanox-clf`** crate (path: `coelanox/coelanox-clf`) is an **in-tree reader** for the public CLF spec: it parses headers and manifests and exposes **`get_blob(op_id)`**. It does **not** implement the producer/packer that *writes* CLF files—that lives beside the SIMD build (e.g. `clf/kernels/simd` and pack scripts in this repo).

---

## File kinds and layout

| Extension / use | Role |
|-----------------|------|
| **`.clfc`** | **Compute** CLF (`ClfKind::Compute`). These are the blobs the mini-executor loads for SIMD kernels. Prefer these under `clfc/` in discovery paths. |
| **`.clf`** | Legacy / other kinds; discovery still accepts them outside strict compute-only dirs. |
| **Embedded slices** | **`ClfReaderFromBytes`** parses the same layout from a byte slice (e.g. CLFMM/CLFMP payloads inside a container). Use **`open_with_expected_kind(..., Some(ClfKind::MemoryMovement))`** (etc.) when the kind must match. |

Binary layout (simplified): magic **`CLF1`**, version byte, vendor string, target string, alignment, **kind** (v2+), manifest of **`(op_id, offset, size)`** entries, then a **blob store**. Optional **SIG** block at EOF. See `coelanox-clf/src/format.rs` for constants (`CLF_MAGIC`, `CLF_VERSION`).

---

## `ClfReader` API (file-backed)

| Method | Purpose |
|--------|---------|
| `ClfReader::open(path)` | Open a file; any `ClfKind` accepted. |
| `open_with_expected_kind(path, Some(kind))` | Fail with **`ClfError::KindMismatch`** if the file’s kind does not match (e.g. only compute blobs in a compute pipeline). |
| `get_blob(op_id)` | **`&mut self`** — read **one** blob from the blob store; returns **`Ok(None)`** if `op_id` is missing from the manifest. |
| `op_ids()` | Sorted list of all **`op_id`**s present. |
| `build_code_section(op_ids, MissingOpIdPolicy)` | Concatenate blobs **in the order of `op_ids`**. **`Fail`** → error if any id missing; **`Skip`** → omit missing (use with care). Used when assembling the **machine code** section for a container. |

**Errors** — See **`ClfError`** in `coelanox-clf/src/reader.rs`: invalid magic, unsupported version, missing op with **`Fail`**, IO, kind mismatch.

---

## Op registry (`op_id` ↔ operator)

Canonical numeric ids are defined in **`coelanox-clf/src/op_registry.rs`**:

- **`coelanox_clf::OpType`** — enum aligned with the CLF spec (subset of operators).
- **`op_type_to_clf_id(clf_op: ClfOpType) -> u32`** — single source of truth for **numbers** (e.g. `Add` → **1**, `Abs` → **19**, `Sqrt` → **20**, …).

The **IR** uses a **different** enum: **`coelanox_core::ir::OpType`**. The packager bridges them with **`ir_op_type_to_clf_op_type`** in **`coelanox-packager/src/lib.rs`** (maps each IR variant to **`coelanox_clf::OpType`**). If you add a new **IR** op and a **kernel**, you must:

1. Add **`OpType`** in **`coelanox-core/src/ir.rs`**.
2. Add **`coelanox_clf::OpType`** + **`op_type_to_clf_id`** arm in **`coelanox-clf`** (choose the next free id; see existing table / `clf/kernels/simd` header comments).
3. Extend **`ir_op_type_to_clf_op_type`** in the packager.
4. Implement the **kernel** (e.g. `clf/kernels/simd`) and **executor** dispatch (`coelanox-executor` **`match step.op_id`** for symbol path; blob path uses the same **`op_id`**).
5. Rebuild / repack **`.clfc`** so **`get_blob(op_id)`** returns the new blob.

Never renumber existing **`op_id`**s in shipped CLF files without updating **executor**, **registry**, and **packager** in the same release.

---

## Packager and orchestrator integration

- **`BackendManager`** (`coelanox-orchestrator`) **discovers** `.clfc` / `.clf` files under configurable search paths. Default layout includes **`~/.coelanox/clf/`** with subdirs **`clfc/`**, **`clfmm/`**, **`clfmp/`**, **`clfe/`** (compute vs memory-movement vs memory-protection vs executor-specific payloads). Only **`.clfc`** files are picked up when the scan directory is the **`clfc/`** compute folder.
- Each discovered file is opened with **`ClfReader::open`**, header inspected (vendor / target / kind), and registered as a **backend** candidate for packaging and metadata.
- During **`package`**, if kernel offsets are resolved, **`build_code_section`**-style concatenation can feed the **machine code** region; the **execution plan** still lists per-step **`op_id`** and offsets for **`run_with_plan`**.

---

## Runtime: blobs vs symbols

| Path | When | Behavior |
|------|------|----------|
| **`run_with_plan`** | Container embeds raw blob bytes + offsets | **`ClfeKernelFn`**: 6-arg uniform ABI (**input**, **output**, **weights**, **in_len**, **out_len**, **w_len**). |
| **`run_with_plan_symbols`** | No blobs; **x86_64** + linked **`clf_kernels_simd`** | Dispatch **`match op_id`** to **`clf_simd_*`** functions (same numeric **`op_id`** as registry). |

If a plan step references an **`op_id`** with no blob and no symbol, execution fails—**scalar fallback** in the orchestrator is a separate path using **`coelanox_core::scalar::execute_scalar_op`**, not arbitrary **`Custom`** IR ops.

---

## In-repo SIMD and ABI

- **Kernels:** `clf/kernels/simd` (Rust **AVX2** helpers, **`clf_simd_*`** + optional **`clf_abi6_*`** sections for CLFC extraction).
- **ABI doc:** [clf/kernels/simd/ABI.md](../../clf/kernels/simd/ABI.md) — C signatures, **CLFE 6-arg** contract, packaging notes.

---

## Linking an external CLF / producer tree

- **Submodule** or **path dependency** on **`coelanox-clf`** is already how the workspace reads files.
- Vendors ship **`.clfc`** files that conform to the same manifest + blob layout; **`op_id`** table must match **`op_registry.rs`**.
- **`crates.io`** — publish **`coelanox-clf`** if you need a standalone reader crate version pin.

---

## Production checklist (CLF / kernel vendors)

Ship compute CLF that COELANOX can load and execute in CI and customer environments:

- [ ] **Magic / version** — **`CLF1`**, version ≤ **`CLF_VERSION`** in `coelanox-clf/src/format.rs`.
- [ ] **Kind** — Compute files use **`ClfKind::Compute`**; readers may **`open_with_expected_kind`**.
- [ ] **Manifest** — Every **`op_id`** required by shipped models has a blob, or document **`MissingOpIdPolicy`** explicitly (plans expect kernels for each step).
- [ ] **ABI** — Blobs for **`run_with_plan`** implement the **CLFE 6-arg** contract ([ABI.md](../../clf/kernels/simd/ABI.md)); **`op_id`** matches **`coelanox_clf::op_registry`**.
- [ ] **Target string** — Header **`target`** aligns with backend discovery / manifest conventions.
- [ ] **Signature block** — If **`SIG0`** footer is present, align with your signing pipeline.
- [ ] **Regression tests** — **`ClfReader::open`**, **`get_blob`** per **op_id**, minimal executor smoke test in CI.

Coordinate **op_id** additions with **`coelanox-core`** / **`coelanox-clf`** releases; do not fork the numeric table silently.

---

## See also

| Doc | Topic |
|-----|--------|
| [pipeline.md](pipeline.md) | Where IR meets `op_id` in the container |
| [frontend.md](frontend.md) | Translator output (UniversalIR) |
| [troubleshooting.md](troubleshooting.md) | CLF load errors |
| [glossary.md](glossary.md) | Terms |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Backend Manager discovery |

← [SDK index](README.md)
