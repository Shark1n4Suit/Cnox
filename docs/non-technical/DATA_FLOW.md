# COELANOX — Data flow (procurement / airgapped context)

**Audience:** Security, compliance, and procurement teams who need **where data goes**, **what touches it**, and **what persists**—including **airgapped** deployment.

---

## 1. Actors and trust boundaries

| Actor | Role |
|-------|------|
| **Build machine** | Runs **`coelanox package`**; sees **training export** (e.g. ONNX) and emits **`.cnox`**. |
| **Artifact store / USB** | Stores **`.cnox`**; integrity is protected by **SHA-256** (and optional **Ed25519**). |
| **Runtime host** | Runs **`coelanox verify`**, **`coelanox run`**, or **`coelanox serve`**; sees **`.cnox`**, **inputs**, **outputs**, **config**. |

**Trust assumption:** The **`.cnox`** file is the **unit of integrity**. Anything that modifies bytes **after** packaging should fail **verify**.

---

## 2. Packaging flow (build machine)

```
[Trained model export]     e.g. ONNX, BERT bundle, demo bundle
        │
        ▼
[coelanox package]  ── reads model files from disk
        │
        ├── Translator loads weights/activations into memory
        ├── IR + optimization + optional CLF codegen
        ├── Compression (optional)
        └── Writes .cnox to disk
```

**What touches model data:** Packager process, translator plugins, optimizer, compressor.  
**What persists:** **`.cnox`** file (and sidecar keys if you manage them separately). **No** cloud requirement.

**Secrets:** **Signing private key** (if used) must be protected **outside** this document’s scope; only the **public** key is needed on verify/run for signature checks.

---

## 3. Verify flow (no inference)

```
[.cnox file]
      │
      ▼
[coelanox verify]  ── reads file, recomputes hash, compares header
      │
      └── Exit success / failure (no tensor I/O)
```

**What persists:** None beyond logs you configure. **No** input/output tensors.

---

## 4. Run flow (single invocation)

```
[.cnox] + [input tensor file or synthetic input]
        │
        ▼
[coelanox run]  ── optional verify → load container → decompress weights in memory
        │
        ├── Reads input (e.g. JSON f32 array) from disk
        ├── Executes (scalar or SIMD/CLF path)
        └── Writes output (e.g. JSON) to disk
```

**What touches inference data:**

| Data | Typical location | Persistence |
|------|------------------|-------------|
| Container | Disk read | Cached in process memory (LRU cache configurable) |
| Weights | Inside `.cnox`; decompressed in RAM | Not written back to `.cnox` by default |
| Input / output tensors | Paths you pass to CLI | **Your** output files; **your** responsibility to protect |
| Logs / traces | stderr, `RUST_LOG` | Your log pipeline |
| **Audit** (`--audit`, `--audit-output`) | Optional file path | **Persistent** JSONL (or configured format) on disk |

---

## 5. Serve flow (long-lived process)

```
[.cnox] loaded once
[stdin binary IPC]  ──► framed requests (CREQ / CNXT)
[stdout binary IPC]  ──► framed responses (CROk / CREr)
[stderr]  ──► logs, COELANOX_SERVE_READY line
```

**What persists:** Same as run for **audit** if enabled; **no** automatic database. Process holds **container + decompressed weights** in memory according to runtime cache settings.

---

## 6. Airgapped deployment (typical)

1. **Build** `.cnox` on a connected machine **or** on a dedicated build island.  
2. **Verify** on the same machine or in CI: `coelanox verify -f model.cnox`.  
3. **Transfer** the artifact via **approved media** (USB, cross-domain transfer).  
4. **Re-verify** on the airgapped host before production use.  
5. **Run** with local input files only; **no** network required for core inference.

**Network:** COELANOX CLI does **not** require outbound network for **verify** or **run**. (Release downloads are a separate step.)

---

## 7. What COELANOX does not phone home

The **open-source CLI/runtime** described in this repository does **not** implement telemetry or license callbacks. Any “phone home” behaviour would be **your** wrapper or distribution.

---

## 8. Summary table

| Stage | Input | Output / persistence |
|-------|--------|----------------------|
| Package | Source model files | `.cnox` on disk |
| Verify | `.cnox` | Exit code + console |
| Run | `.cnox`, input file | Output file + optional audit file + logs |
| Serve | `.cnox`, stdin IPC | stdout IPC + stderr logs + optional audit |

---

## Related documents

- [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) — Guarantees  
- [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md) — Failure modes  
- [Operations](../OPERATIONS.md) — Env limits, signing, paths  

← [Non-technical hub](README.md)
