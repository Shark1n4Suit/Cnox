# HAL mapping: OpType ↔ CLF ↔ Memory HAL ↔ Protection HAL

The pipeline maps **OpType (IR)** to **CLF** (code), then **Memory HAL** (where regions live), then **Protection HAL** (permissions on those regions).

---

## 1. OpType → CLF (code)

| Layer | Role |
|-------|------|
| **IR OpType** | Graph nodes (Add, Relu, MatMul, …). |
| **CLF** | Canonical op_id and kernel blobs. Packager maps `ir::OpType` → `clf::OpType` → `op_type_to_clf_id()` → op_id; then `ClfReader::build_code_section(op_ids)` yields the machine code array. |

So: **OpType** (IR) → **CLF op_id** → kernel blob; concatenated blobs = container’s **code** region.

---

## 2. Region kinds → Memory HAL

Three region kinds correspond to the container and AOT lock model:

| Region kind | Content | Memory HAL role |
|-------------|--------|------------------|
| **Code** | Machine code (from CLF or translator). | Allocate **executable** region (e.g. `default_for_executable()`, `RegionOptions { executable: true }`). Packager does not allocate code; runtime allocates and copies `container.optimized_code`. |
| **Weights** | Decompressed weights (static). | **Backing** at package time: `create_backing(metadata)` (e.g. NVMe layout). Runtime loads backing into a region (future: allocate via Memory HAL or map backing). |
| **Intermediates** | Activations, temp buffers. | Same backing as weights (pre-allocated layout from `MaxShapeMetadata`). Runtime uses that layout for activation buffers. |

So: **Code** = Memory HAL executable allocation. **Weights + intermediates** = Memory HAL backing / `create_backing` (and later allocation if desired).

---

## 3. Region kinds → Protection HAL

After Memory HAL has provided regions, Protection HAL applies permissions (AOT lock: code=X, weights=R, intermediates=RW).

| Region kind | Protection HAL permission |
|-------------|---------------------------|
| **Code** | `RegionPermissions::code()` → read + execute, no write. |
| **Weights** | `RegionPermissions::weights()` → read-only, no write, no execute. |
| **Intermediates** | `RegionPermissions::intermediates()` → read + write, no execute. |

Flow: **Memory HAL** allocates/places region → **Protection HAL** `configure_region(region, permissions)` → `enable_protection()` for lock.

---

## 4. End-to-end map

```
OpType (IR)  ──►  CLF op_id  ──►  code blobs  ──►  Code region   ──►  Memory HAL (allocate executable)
                                                                   ──►  Protection HAL (code = X)

Weights + intermediates (from IR/metadata)  ──►  create_backing()  ──►  Backing / regions
                                                                   ──►  Memory HAL (backing, then allocate if used)
                                                                   ──►  Protection HAL (weights = R, intermediates = RW)
```

So **OpType** maps to **CLF** (code); **region kinds** (code, weights, intermediates) map to **Memory HAL** (allocation/backing) and **Protection HAL** (permissions).
