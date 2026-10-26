# Glossary — SDK and runtime

| Term | Meaning |
|------|--------|
| **UniversalIR** | Graph IR in `coelanox-core::ir`: nodes (`OpType`), tensors (`TensorId`), weights, metadata. **Translator output** before packager optimization. |
| **OptimizedIR** | IR after packager passes (`OptimizedIR` in `ir.rs`): fused/reordered nodes for a target; serialized into `.cnox`. |
| **OpType** | Enum of primitive ops (`MatMul`, `Relu`, …). Prefer these over **`Custom(String)`** — packaging **rejects** remaining `Custom` nodes. |
| **TensorId** | Opaque id for an intermediate or weight tensor; wires nodes together. |
| **Attribute** | Key/value metadata on a node (`axes`, `shape`, conv params, …). Consumed by optimizer and runtime. |
| **op_id** | **u32** kernel opcode in CLF and execution plan. Mapped from IR via **`coelanox_clf::op_type_to_clf_id`**. |
| **CLF** | Coelanox Library File: header + manifest + blob store for kernels. |
| **.clfc** | Compute CLF (`ClfKind::Compute`); SIMD / native blobs keyed by **op_id**. |
| **Execution plan** | Binary layout: per-step **op_id**, buffer offsets, lengths; optional kernel offsets into embedded code. Built by **`build_plan_from_ir`** (`coelanox-executor`). |
| **CLFE 6-arg ABI** | Uniform C ABI for packed blobs: `(input_ptr, output_ptr, weights_ptr, in_len, out_len, w_len)`. See [ABI.md](../../clf/kernels/simd/ABI.md). |
| **BackendManager** | Orchestrator component that **discovers** `.clfc` files at **package time** for backend metadata and optional codegen. |
| **Scalar fallback** | Reference execution via `coelanox_core::scalar::execute_scalar_op` — **not** a substitute for lowering arbitrary **`Custom`** IR ops. |

← [SDK index](README.md)
