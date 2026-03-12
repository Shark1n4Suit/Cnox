# COELANOX Features

This document lists implemented and planned features for the current release. Items marked **Planned** are on the roadmap but not yet implemented.

---

## 1. Container & format

### Implemented
- **Universal container (`.cnox`)** — Single file: 76-byte header (magic COEL, version, SHA-256), JSON manifest, IR, machine code (or CLF blobs), compressed weights.
- **SHA-256 integrity** — Hash over full container at package time; verified by `coelanox verify` and before load in `coelanox run`.
- **Three-command flow** — `package` (create container), `verify` (standalone integrity check), `run` (verify then load then execute).
- **Format-agnostic payload** — Container can hold any model representation; translation is done by packager translators.

### Model input (packaging)
- **In-tree translators:** BERT (BRT1 bundle; `--model-type nlp`, `--use-demo-translator`), ResNet-tiny-MNIST (config + weights; CV demo), **ONNX (opset 13)** in `coelanox-packager` with feature `onnx-protobuf`; register via plugins to package ONNX models.
- **Plugin system** — Translator plugins (trait-based); new formats can be added via plugins. See [ONNX decomposition tree](../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) for ONNX → IR mapping.

### Not implemented / planned
- Generic ONNX/PyTorch/TensorFlow translators are **not** shipped in-tree (planned or community).
- Single container “runs everywhere” depends on backend: scalar runs anywhere; SIMD/CLF requires matching CLF artifacts.

---

## 2. CLI commands

| Command    | Status | Description |
|-----------|--------|-------------|
| **package** | ✅ | Build `.cnox` from model (BERT, ResNet-tiny-MNIST; target/backend flags); optional `--signing-key` for Ed25519 signing. |
| **validate** | ✅ | Check container structure and manifest. |
| **keygen**    | ✅ | Generate Ed25519 keypair for signing (secret.bin, public.bin). |
| **verify**   | ✅ | Standalone SHA-256 integrity check; optional `--trusted-key` for signature verification; `--require-signature` to fail if signed but no key. |
| **run**      | ✅ | Verify → load → run inference; `--input`, `--output`, `--backend`, `--audit`, `--max-workspace-mb`, `--no-verify`, `--trusted-key`, `--require-signature`. |
| **info**     | ✅ | Print manifest, sizes, workspace; optional `--dump-plan`. |
| **benchmark**| ✅ | Repeated inference, latency/throughput; `--backend`, `--iterations`, `--trusted-key`, `--require-signature`. |
| **env**      | ✅ | System environment, CPU, SIMD, recommended target. |
| **debug**    | ✅ | Run with diagnostics; optional GDB. |
| **extract**  | ✅ | Same-arch: copies current executable to --output for deployment (binary + .cnox only). Cross-arch (--arm/--target) planned. |

---

## 3. Execution & backends

### Implemented
- **Dual execution path** — Same `.cnox` can run via **scalar** (pure Rust, all platforms) or **plan/SIMD** (CLF mini-executor when compute backends are available).
- **Scalar fallback** — Single implementation in `coelanox-core`; reference for correctness; no vendor runtime required.
- **CLF path** — Execution plan (workspace, node steps, op_ids, kernel offsets); mini-executor dispatches to CLF kernel blobs; proven with BERT and ResNet-tiny-MNIST.
- **Backend discovery** — Default path `~/.coelanox/clf/clfc/` (no env required); override via `COELANOX_CLF_PATH`; additional paths via `COELANOX_BACKEND_PATH`. BackendManager discovers CLF and vendor backends at package and run time.
- **AOT-style selection at package time** — `--target` (slim), `--fat`, `--auto`, `--backend-add`; container carries selected backend metadata.
- **Run-time backend choice** — `--backend auto|scalar|cpu|simd` for `run` and `benchmark`.

### Not in-tree
- No OpenVINO, TensorRT, ROCm, or CUDA runtime in the repo; vendor backends are via discovery (e.g. CLF) or future SDK.
- AMD reference backend and formal TensorRT comparison are **planned** (roadmap).

---

## 4. Universal IR & operations

### Implemented
- **Universal IR** — Graph-based; nodes (OpType), weight/intermediate tensors, topological order.
- **Primitive OpType set** — Arithmetic (Add, Subtract, Multiply, Divide); activations (Relu, Sigmoid, Tanh, Softmax, LogSoftmax, Gelu, Swish); math (Sqrt, Pow, Cos, Sin, Exp, Log); conv/pool/norm (Convolution, MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool, BatchNorm, LayerNorm, Dropout); tensor ops (Reshape, Transpose, Permute, Concatenate, Split, Slice, Gather, Scatter); MatMul, Gemm; reductions (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd); Broadcast, Expand; comparisons (Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual); logical (And, Or, Not); Custom(String).
- **CLF op_id mapping** — Executor uses canonical op_ids (e.g. Add=1, MatMul=50, Gemm=51, LayerNorm=36, Gelu=15, Softmax=13, etc.) for plan dispatch.
- **Optimization** — Graph optimization, primitive-level optimizations (attributes, layout hints). Operator fusion and constant folding are planned.

### Planned
- Broader decomposition (e.g. full RNN/Transformer as primitives) and additional ops as needed.

---

## 5. Security

### Implemented
- **Integrity** — SHA-256 over full container; verify before load; tamper detection.
- **Container signing** — Ed25519 signing of containers; `keygen` (keypair), `package --signing-key`, `verify`/`run`/`benchmark --trusted-key`; optional `--require-signature` to enforce provenance.
- **Path validation** — Path traversal protection (`..`, `~`); optional rejection of absolute paths (`COELANOX_ALLOW_ABSOLUTE_PATHS=false`).
- **Size limits** — Configurable max container size and max input size (`COELANOX_MAX_CONTAINER_SIZE_BYTES`, `COELANOX_MAX_INPUT_SIZE_BYTES`).
- **Security event logging** — Verification failure, path rejection, and size-limit errors logged (tracing).
- **Memory safety** — Rust; bounds checking; controlled use of `unsafe` for executable memory.
- **Input validation** — Optional NaN/Inf checks; input size vs expected shape.

### Planned
- Optional weight encryption; code signing for backend libraries (roadmap).

---

## 6. Configuration & operations

### Implemented
- **Runtime config from environment** — `COELANOX_MAX_CONTAINERS`, `COELANOX_MAX_MEMORY_BYTES`, `COELANOX_EXECUTION_TIMEOUT_MS`, `COELANOX_MAX_INPUT_SIZE_BYTES`, `COELANOX_MAX_CONTAINER_SIZE_BYTES`, `COELANOX_ALLOW_ABSOLUTE_PATHS`, `COELANOX_LOG_LEVEL`, `COELANOX_DEFAULT_BACKEND`, and related flags.
- **Config file** — Optional JSON config via `COELANOX_CONFIG_FILE`; partial config supported (missing fields use defaults).
- **Workspace cap** — `COELANOX_MAX_WORKSPACE_MB` or `--max-workspace-mb` for plan execution.
- **Container cache** — In-process cache with configurable max size and optional TTL.
- **Runbooks** — OPERATIONS.md: verification failed, size limit, timeout, backend fallback, absolute path rejected; pre-deployment checklist.

---

## 7. Observability & diagnostics

### Implemented
- **Structured logging** — Tracing (e.g. `RUST_LOG`); security and failure events logged.
- **Audit mode** — `--audit` with optional `--audit-output` for op-level execution trace (op_ids, shapes, samples).
- **Benchmark metrics** — Latency (avg, min, max, p50/p95/p99), throughput, true cold, warm avg; optional long-run stability (first/last 10%).
- **Runtime stats** — Execution time, container cache, memory (e.g. RSS) available via API/stats.

### Not in-tree
- No built-in Prometheus/OpenTelemetry export or HTTP health endpoint; integrate via your own service and log pipeline.

---

## 8. Code generation & executor

### Implemented
- **CPU codegen** — x86-64; SIMD loop generation; stack-based buffer layout; operation codegen for supported ops.
- **CLF mini-executor** — Parses binary execution plan; walks graph; dispatches to kernel blobs by op_id; single ABI (e.g. input/output/weights pointers + lengths).
- **Executable memory** — Runtime allocates executable region (mmap/VirtualAlloc or Memory HAL); copies code; runs plan.

### HALs (implemented, integration varies)
- **Memory HAL** — Trait and registry; packager can use for backing (e.g. NVMe); runtime can use for executable allocation.
- **Protection HAL** — Trait and registry for MMU/MPU-style protection; not yet wired into runtime load/lock/launch (planned).

---

## 9. Storage & compression

### Implemented
- **Weight compression** — Zstd and LZ4 at package time.
- **Unified container layout** — Header, manifest, code, weights (and optional backing) in one file format.

### Planned
- Chunk streaming, double ring buffer, NVMe/RAM tiering with async I/O (roadmap). Max-shape pre-allocation and NVMe-backed backing exist in part; full tiering is planned.

---

## 10. Platform & hardware

### Implemented
- **Platforms** — Linux, Windows (Rust target); macOS noted in README.
- **CPU** — x86-64 with scalar path everywhere; SIMD/CLF path when CLF kernels and `COELANOX_CLF_PATH` are provided.
- **Cross-arch** — Same container can be run with scalar on any arch; CLF path is x86_64 in current executor (other arches fall back to scalar).

### Not in-tree
- No OpenVINO (Intel), TensorRT (NVIDIA), or ROCm (AMD) runtime in repo; GPU/NPU support is via future vendor backends or CLF.

---

## 11. Production readiness (beta)

### Implemented
- **Config-driven limits** — Memory, container size, input size, timeouts, path policy.
- **Lock hardening** — Runtime uses `expect()` instead of `unwrap()` on locks for clearer failure behavior.
- **Verify-then-run** — Default flow verifies integrity before load; `--no-verify` for dev only.
- **Documentation** — OPERATIONS.md (env vars, runbooks), SECURITY.md, ARCHITECTURE.md.
 - **Runtime extract** — `coelanox extract` produces a **same-arch** standalone runtime binary that matches the CLI version; deploy with `.cnox` files only. Cross-arch (`--arm` / `--target`) is **not yet implemented** and currently prints a warning/error.

### Planned
- Formal TensorRT/vendor benchmarks and vendor SDK documentation (roadmap).

---

## 12. Summary table

| Area              | In beta / implemented | Planned / not in-tree |
|-------------------|----------------------|------------------------|
| Container format  | ✅ .cnox, SHA-256, verify, run | — |
| Model import      | ✅ BERT, ResNet-tiny-MNIST (demo translators) | Generic ONNX/PyTorch/TF translators |
| Execution         | ✅ Scalar + CLF/plan, backend discovery, AOT selection | Vendor SDK doc, AMD reference |
| IR & ops          | ✅ Universal IR, primitive OpType set, CLF op_ids, fusion | — |
| Security          | ✅ Hash, path/size limits, logging, Rust safety | Encryption, code signing |
| Config & ops      | ✅ Env + file config, runbooks, workspace cap | — |
| Observability     | ✅ Logging, audit, benchmark metrics | Prometheus, health endpoint |
| Codegen / executor| ✅ CPU codegen, CLF mini-executor, Memory HAL | Protection HAL wired, chunk streaming |
| Storage           | ✅ Compression, unified layout | Full tiering, ring buffer |
| Platform          | ✅ Linux, Windows; x86_64 scalar + CLF | macOS tested; GPU/NPU via vendor |
| Deployment        | ✅ Full CLI; extract (same-arch runtime binary) | extract --arm/--target (cross-arch) |

---

This sheet reflects the current codebase and docs; remove or adjust bullets as you ship or deprecate features.
