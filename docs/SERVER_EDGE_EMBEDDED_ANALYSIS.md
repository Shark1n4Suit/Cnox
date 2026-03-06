# COELANOX: Server, Edge, and Embedded Deployment

This document describes how the codebase supports server, edge, and embedded deployment with a single artifact (`.cnox`): verify once, deploy per environment. It covers container and weight sharing (e.g. `Arc<CoelanoxContainer>`, warm cache) and references the relevant implementation locations.

---

## 1. Summary Table

| Environment   | Host / runtime role                    | Container cache              | Weights / warm data                    | Arc usage (server) |
|---------------|----------------------------------------|------------------------------|----------------------------------------|--------------------|
| **Server**    | Orchestrator: load, schedule, I/O, evict | `Arc<CoelanoxContainer>` per entry | Warm cache: decompressed + IR; *recommend* `Arc<Vec<f32>>` for weights | Container: **yes**; weights: **recommended** |
| **Edge**      | Same runtime, one/few containers        | Same type, lighter usage     | Same warm path; optional `--static-ram` | Not required       |
| **Embedded**  | Minimal shim: load `.cnox`, run plan    | No runtime cache             | In-container; no warm cache             | N/A                |

---

## 2. Server: Arc&lt;CoelanoxContainer&gt; and Arc&lt;weights&gt;

### 2.1 Container cache: Arc&lt;CoelanoxContainer&gt; (implemented)

The runtime stores one **Arc** per loaded container so many requests share the same in-memory container instead of copying it (e.g. avoiding 70GB clone per request).

**Code (orchestrator):**

- **Cache type:**  
  `containers: Arc<RwLock<HashMap<String, (Arc<CoelanoxContainer>, Instant)>>>`  
  — key = path/name, value = `(Arc<CoelanoxContainer>, load time)` for TTL eviction.

- **On load:** after validation, the container is wrapped once in `Arc` and inserted:  
  `let container = Arc::new(container);`  
  `containers.insert(path.to_string(), (container, Instant::now()));`  
  (See e.g. `load_container`, `load_container_no_verify`, `load_container_from_data` in `coelanox-orchestrator/src/runtime.rs`.)

- **On run:** execution gets an **Arc clone** (pointer copy), not a full container clone:  
  `Some((container, _)) => Arc::clone(container)`  
  (See `run_backend_execution` / execute paths that look up by `container_name`.)

**Roadmap:** *“Use Arc&lt;CoelanoxContainer&gt; in server container cache … Each request gets an Arc clone (8 bytes) not a full copy.”* — **Done** in code.

**Eviction:** Implemented so the server does not hold every model forever:

- `max_containers`: when at limit, one entry (e.g. oldest key) is removed before insert.
- `container_cache_ttl_secs`: `evict_expired_containers_inner` removes entries older than TTL; warm_cache entry for that container is also removed.

So for **server**, **Arc&lt;CoelanoxContainer&gt;** is in place and eviction (TTL + max size) is implemented.

---

### 2.2 Weights: current vs Arc&lt;weights&gt; for server

**Where weights live:**

- **In container:** Compressed in the container blob; decompressed on first use (or when building warm data).
- **In IR:** `OptimizedIR.shared_weights: Arc<Vec<WeightTensor>>` — **metadata** (layout, shape) is shared via Arc when IR is cloned (see `coelanox-core/src/ir.rs`).
- **At runtime (warm cache):** One decompressed copy per container name in `warm_cache: Arc<RwLock<HashMap<String, ContainerWarmData>>>`, with  
  `ContainerWarmData { decompressed_weights: Vec<f32>, optimized_ir: OptimizedIR }`  
  (see `coelanox-orchestrator/src/execution_backend.rs` and runtime `warm_cache` in `runtime.rs`).

**Current behavior:**

- **resolve_warm_data** does:
  - Cache hit: `return Ok(Some(warm.clone()))` — **clones** the whole `ContainerWarmData`, including the full `Vec<f32>` of decompressed weights.
  - Cache miss: decompress + deserialize, build `ContainerWarmData`, insert, return.

So on every inference that uses warm data, the runtime **clones** the decompressed weight vector. For large models on server this can mean hundreds of MB copied per request.

**Arc&lt;weights&gt; for server (recommended):**

- Store decompressed weights in a **shared** form so that multiple requests do not copy them. Concretely:
  - Change `ContainerWarmData` so that decompressed weights are **Arc&lt;Vec&lt;f32&gt;&gt;** (or similar), and/or
  - Store `Arc<ContainerWarmData>` in the warm cache and return `Arc::clone` on hit instead of `warm.clone()`.
- Then “Arc&lt;weights&gt;” for server means: **one decompressed weight buffer per container, shared via Arc across concurrent inferences** (no per-request clone of the weight buffer).

**In execution (scalar fallback):** Per-node weight **slices** are passed as `Arc<Vec<f32>>` (slices copied into Arcs for parallel workers) in `coelanox-orchestrator/src/fallback/mod.rs`. The main gain for server is sharing the **full** decompressed buffer via Arc in the warm cache, not cloning it in `resolve_warm_data`.

**Summary:**

- **Arc&lt;CoelanoxContainer&gt;** for server: **implemented** (container cache).
- **Arc&lt;weights&gt;** for server: **partially** — IR has `Arc<Vec<WeightTensor>>` for metadata; fallback uses Arc for weight slices during execution; **warm cache currently clones** `ContainerWarmData` (including weights) on every resolve. **Recommendation:** add Arc-backed sharing for decompressed weights (e.g. `Arc<Vec<f32>>` in `ContainerWarmData` or `Arc<ContainerWarmData>` in cache) so server does not clone large weight buffers per request.

---

## 3. Edge

- **Same artifact and runtime** as server; typically one (or few) containers, loaded once.
- **Arc not required:** Roadmap: *“Edge doesn’t need Arc”* — no request routing or shared cache across many models; a single loaded container is fine without Arc for the cache (though the same type `Arc<CoelanoxContainer>` is still used in the implementation).
- **Packaging:** `--target edge` (or specific edge target) and optional `--static-ram` for fixed memory budget (see roadmap and CLI reference).
- **Deploy:** Same verify-then-run; optional `coelanox extract` for a runtime-only binary (e.g. ARM) to ship with `.cnox`.

---

## 4. Embedded (e.g. ESP32, bare metal, RTOS)

- **No full Coelanox runtime.** User builds a **minimal shim**: load `.cnox`, parse plan, set up buffers, run plan (single entry or step-by-step).
- **Container is self-contained:** IR + weights + code (CLFC) + optional CLFMM/CLFMP; no OS required for memory or protection (container can bring its own HAL blobs).
- **Extract:** `coelanox extract` is for server/desktop runtimes (x86, ARM with OS). For constrained embedded, the stack is “minimal shim + .cnox” only.

---

## 5. Code References (checklist)

| Topic | Location |
|-------|----------|
| Container cache type | `coelanox-orchestrator/src/runtime.rs`: `containers: Arc<RwLock<HashMap<String, (Arc<CoelanoxContainer>, Instant)>>>` |
| Load → Arc::new + insert | `runtime.rs`: `load_container`, `load_container_no_verify`, `load_container_from_data` |
| Run → Arc::clone(container) | `runtime.rs`: execute paths that get container from cache (e.g. around 1495–1499, 1591, 2437–2440) |
| Eviction (TTL + max) | `runtime.rs`: `evict_expired_containers_inner`, `evict_container`, and load path when `containers.len() >= config.max_containers` |
| Warm cache type | `runtime.rs`: `warm_cache: Arc<RwLock<HashMap<String, ContainerWarmData>>>` |
| ContainerWarmData | `coelanox-orchestrator/src/execution_backend.rs`: `decompressed_weights: Vec<f32>`, `optimized_ir: OptimizedIR` |
| resolve_warm_data (clone on hit) | `runtime.rs`: `resolve_warm_data` — on hit returns `Ok(Some(warm.clone()))` |
| IR shared_weights | `coelanox-core/src/ir.rs`: `shared_weights: Arc<Vec<WeightTensor>>` |
| Fallback weight slices as Arc | `coelanox-orchestrator/src/fallback/mod.rs`: `Arc::new(weights_ref[start..end].to_vec())` for node inputs |
| Roadmap Arc + edge | `roadmaps/PRODUCTION_MVP_ROADMAP.md`: “Use Arc<CoelanoxContainer>…”, “Edge doesn’t need Arc”, deployment targets, extract |

---

## 6. Redo Summary

- **Server:**  
  - **Arc&lt;CoelanoxContainer&gt;** is implemented: container cache stores and returns `Arc<CoelanoxContainer>`, with TTL and max_containers eviction.  
  - **Arc&lt;weights&gt;:** IR already uses `Arc<Vec<WeightTensor>>` for metadata; warm cache currently **clones** decompressed weights on every resolve — for true server-scale weight sharing, add Arc for the decompressed weight buffer (e.g. `Arc<Vec<f32>>` in `ContainerWarmData` or `Arc<ContainerWarmData>` in the cache).

- **Edge:** Same container and runtime; Arc not required for the use case; optional `--static-ram` and edge target.

- **Embedded:** Same `.cnox`; no full runtime; minimal shim loads container and runs plan; container is self-contained (optional CLFMM/CLFMP).

All of the above is consistent with the current codebase and the production MVP roadmap.
