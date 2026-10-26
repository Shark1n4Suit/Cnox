# COELANOX SDK documentation

This section is the **authoritative contract** for compiler frontends (translators) and compute backends (CLF) that integrate with COELANOX. It is written for **engineers shipping production pipelines**: CI integration, vendor plugins, and internal feature work.

There is no separate `coelanox-translator-sdk` crate. The public API lives in **`coelanox-core`** (`CoelanoxTranslator`, `UniversalIR`, `OpType`, `ModelFormat`). CLF I/O is in **`coelanox-clf`**.

**Open vs closed:** Translator integrations are typically **open** (or public-facing); **CLF / kernel SDKs** are often **closed** or partner-only. Translator authors should treat **every commit** as publishable: stick to the **`coelanox-core` IR contract**, avoid embedding closed-stack assumptions or NDA’d details in code, tests, or messages. See [frontend.md](frontend.md#open-translator-closed-clf).

---

## Audience and scope

| Reader | Start here |
|--------|------------|
| **Translator author** | [frontend.md](frontend.md) → [pipeline.md](pipeline.md) |
| **Kernel / CLF vendor** | [clf.md](clf.md) → [clf/kernels/simd/ABI.md](../../clf/kernels/simd/ABI.md) |
| **Operator / SRE** | [pipeline.md](pipeline.md), [OPERATIONS.md](../OPERATIONS.md), [troubleshooting.md](troubleshooting.md) |

Out of scope here: full CLI tutorial ([QUICKSTART.md](../QUICKSTART.md)), product marketing ([REFERENCE.md](../REFERENCE.md)).

---

## Document map

| Doc | Contents |
|-----|----------|
| [frontend.md](frontend.md) | `CoelanoxTranslator`, UniversalIR construction, `OpType`, registration, features, **production checklist** |
| [pipeline.md](pipeline.md) | Translate → optimize → plan → `.cnox` → runtime; artifacts and failure modes |
| [clf.md](clf.md) | `coelanox-clf`, `.clfc`, `op_id` registry, executor, **vendor checklist** |
| [glossary.md](glossary.md) | Terms (IR, CLF, plan, op_id, …) |
| [troubleshooting.md](troubleshooting.md) | Common failures (package, verify, Custom ops, CLF) |

---

## Versioning and compatibility

- **Semver** applies to **`coelanox-core`** (and published crates). Bumping **minor** may add `OpType` variants or trait methods with defaults; **major** breaks IR serialization or trait shape.
- **Translators** should pin a **workspace or crate version** in CI and run **`cargo test`** on translator changes before release.
- **`op_id`** values in **`coelanox-clf`** are **stable per release line**; do not renumber without a **major** CLF / runtime coordinated release (see [clf.md](clf.md)).

**Rust API (generated):**

```bash
cd coelanox
cargo doc -p coelanox-core -p coelanox-clf --no-deps --open
```

Modules to read first: **`coelanox_core::plugins`**, **`coelanox_core::ir`**, **`coelanox_clf::reader`**, **`coelanox_clf::op_registry`**.

---

## Related

| Resource | Topic |
|----------|--------|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | System design, container layout |
| [CLI_REFERENCE.md](../CLI_REFERENCE.md) | `package`, `verify`, `run` |
| [OPERATIONS.md](../OPERATIONS.md) | Security, env, deployment |
| [roadmaps/FRONTEND_SDK_ROADMAP.md](../../roadmaps/FRONTEND_SDK_ROADMAP.md) | Dynamic plugins, future work |
| [CHANGELOG.md](../../CHANGELOG.md) | Release notes |

← [Developer hub](../DEVELOPER.md)
