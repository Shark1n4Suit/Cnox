# Packaging pipeline (frontend → container → runtime)

How **`UniversalIR`** becomes a **`.cnox`** file and how inference runs.

**Contracts:** **Frontend** = IR + `OpType`; **backend** = **`op_id`** + kernel ABI. They meet in **`coelanox-packager`** (`ir_op_type_to_clf_op_type` + `op_type_to_clf_id`) and in the **binary execution plan**.

---

## Stages

1. **Translate** — `CoelanoxTranslator::translate` (or path helpers) returns **`UniversalIR`** (graph, tensor metadata, weights).
2. **Optimize** — Packager runs IR optimizers for the selected **hardware target**; output **`OptimizedIR`** (shared weights, optimized node list).
3. **Plan** — Each optimized node’s **`coelanox_core::ir::OpType`** → **`op_id`** (`coelanox-packager/src/lib.rs` + `coelanox_clf::op_type_to_clf_id`). **`build_plan_from_ir`** (`coelanox-executor`) emits: workspace offsets, per-step **`op_id`**, **`in_len` / `out_len` / `w_len`**, optional **kernel byte offsets** into the concatenated code section.
4. **Package** — **`CoelanoxContainer`**: manifest JSON, **compressed weights**, serialized IR (often **`OptimizedIR`**), optional **machine code** blob (concatenated CLF extracts), optional **execution plan**, NVMe backing slice, security flags.
5. **Run** — **`verify`** checks hash; runtime loads container, allocates workspace, either **`run_with_plan`** (embedded blobs + offsets) or **`run_with_plan_symbols`** (linked **`clf_kernels_simd`**, same **`op_id`** dispatch). **Scalar fallback** uses **`execute_scalar_op`** when configured—not for arbitrary **`Custom`** ops.

---

## Container artifacts (conceptual)

| Piece | Role |
|-------|------|
| Manifest | JSON: model name, shapes, backend hints, workspace size, offsets into blob |
| Compressed weights | Float (or agreed) payload |
| IR bytes | Serialized graph for inspection / re-planning |
| Machine code | Optional; CLF blob concatenation per backend build |
| Execution plan | Binary steps for the mini-executor |
| NVMe backing | Optional arena metadata for large workspaces |

Exact layout: [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Failure modes (production)

| Stage | Failure | User-visible behavior |
|-------|---------|------------------------|
| Translate | Parse / unsupported op | **`CoelanoxError`** from translator; fix model or extend translator. |
| Package | **`Custom`** op left in IR | Explicit **not implemented** message; must lower or reject in translator. |
| Package | Topological sort | Cycle or dangling tensor; fix graph. |
| Package | Workspace / `--max-workspace` | Plan exceeds limit; reduce model or raise cap. |
| Run | Missing kernel for **op_id** | Runtime error; ensure CLF contains all ids in plan or use symbol path. |

See [troubleshooting.md](troubleshooting.md).

---

## See also

| Doc | Topic |
|-----|--------|
| [frontend.md](frontend.md) | Building IR |
| [clf.md](clf.md) | **`op_id`**, blobs |
| [glossary.md](glossary.md) | Terms |

← [SDK index](README.md)
