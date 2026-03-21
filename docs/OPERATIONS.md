# COELANOX Operations Guide

Configuration, security, deployment, and runbooks for the COELANOX runtime and CLI.

---

## Security

**Integrity** — SHA-256 hash over full container; verified by `verify` and before `run`. Any modification invalidates the hash.

**Signing** — Ed25519 optional: `keygen` (keypair), `package --signing-key`, `verify`/`run`/`benchmark --trusted-key`. Use `--require-signature` to enforce provenance.

**Path & size limits** — Path traversal (`..`) blocked; `COELANOX_ALLOW_ABSOLUTE_PATHS=false` in production; `COELANOX_MAX_CONTAINER_SIZE_BYTES`, `COELANOX_MAX_INPUT_SIZE_BYTES` enforced. See [ARCHITECTURE](ARCHITECTURE.md) for hash flow.

---

## How the CLI uses configuration

Commands that **load and run** a container (`run`, `benchmark`, `debug`) use production config every time:

1. **Config loading**  
   The CLI calls `load_runtime_config()`:
   - If `COELANOX_CONFIG_FILE` is set → load JSON from that path (missing keys use defaults).
   - Else → build config from `RuntimeConfig::from_env()` (each `COELANOX_*` env var overrides the default).

2. **Runtime creation**  
   `CoelanoxRuntime::with_config(config)` is used (not `CoelanoxRuntime::new()`). So:
   - **SecurityValidator** gets: `max_container_size_bytes`, `allow_absolute_paths`, `max_input_size_bytes`, and security flags from config.
   - **Cache and limits** use: `max_containers`, `container_cache_ttl_secs`, `max_memory_bytes`, `execution_timeout_ms`, etc.

3. **Per-command overrides**  
   - `--max-workspace-mb` (or `COELANOX_MAX_WORKSPACE_MB`) still overrides workspace cap after the runtime is created.
   - `--no-verify` skips integrity verification (dev only; avoid in production).

4. **Commands that do not use runtime config**  
   `validate`, `verify`, `info`, and `package` do not create a runtime; they work on files or the packager. So env-based runtime limits do not apply there—only when a container is actually loaded and executed.

**Summary:** For `coelanox run`, `coelanox benchmark`, and `coelanox debug`, the same config path runs: env (or config file) → `RuntimeConfig` → `SecurityValidator` and runtime limits. Set `COELANOX_*` (or `COELANOX_CONFIG_FILE`) in your deployment environment so production uses your limits and policies.

---

## Environment Variables

### Runtime configuration

Loaded in order: `COELANOX_CONFIG_FILE` (if set, JSON file overrides); otherwise individual `COELANOX_*` env vars apply.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `COELANOX_CONFIG_FILE` | path | (none) | Path to JSON config file. If set, config is loaded from file; unset fields fall back to defaults. |
| `COELANOX_MAX_CONTAINERS` | usize | 100 | Max containers to cache; LRU eviction when exceeded. |
| `COELANOX_CONTAINER_CACHE_TTL_SECS` | u64 | (none) | TTL in seconds for cached containers; unset = no TTL. |
| `COELANOX_MAX_MEMORY_BYTES` | usize | 1073741824 (1GB) | Max runtime memory usage in bytes. |
| `COELANOX_EXECUTION_TIMEOUT_MS` | u64 | 30000 | Execution timeout in milliseconds. |
| `COELANOX_MAX_INPUT_SIZE_BYTES` | usize | 104857600 (100MB) | Max input tensor size in bytes. |
| `COELANOX_MAX_CONTAINER_SIZE_BYTES` | usize | 1073741824 (1GB) | Max container file size in bytes (enforced on load). |
| `COELANOX_LOG_LEVEL` | string | "info" | Log level for runtime (e.g. info, debug, warn). |
| `COELANOX_DEFAULT_BACKEND` | string | "cpu" | Default backend (cpu, scalar, auto, etc.). |
| `COELANOX_ALLOW_ABSOLUTE_PATHS` | bool | true | If false, reject container paths starting with `/` or `\` (production hardening). |
| `COELANOX_ENABLE_SECURITY_VALIDATION` | bool | true | Enable path and size validation. |
| `COELANOX_ENABLE_INPUT_VALIDATION` | bool | true | Enable input size and NaN/Inf checks. |

### CLI / execution

| Variable | Description |
|----------|-------------|
| `COELANOX_CLF_PATH` | Override for CLF backend root. If unset, default is `~/.coelanox/clf/` (see Backend discovery below). |
| `COELANOX_BACKEND_PATH` | Colon-separated list of additional paths for backend discovery. Appended after the CLF paths. |
| `COELANOX_MAX_WORKSPACE_MB` | Max workspace size in MB for plan execution (0 = no cap). Overridable by `--max-workspace-mb`. |
| `COELANOX_DOUBLE_RING` | Set to `1` or `true` to force double ring buffer for testing. |
| `RUST_LOG` | Control tracing/log level (e.g. `coelanox=info`, `coelanox_orchestrator=debug`). |

### Config file format (JSON)

When `COELANOX_CONFIG_FILE` is set, the file must be JSON with optional keys (unset = default):

```json
{
  "max_containers": 50,
  "container_cache_ttl_secs": 3600,
  "max_memory_bytes": 2147483648,
  "execution_timeout_ms": 60000,
  "max_input_size_bytes": 209715200,
  "max_container_size_bytes": 2147483648,
  "allow_absolute_paths": false,
  "enable_security_validation": true,
  "enable_input_validation": true,
  "log_level": "info",
  "default_backend": "cpu"
}
```

---

## Backend discovery and CLF path

Backends (e.g. `.clfc` compute libraries) are discovered from a fixed layout so that no environment variable is required for normal use.

**Default (no override):** The CLI uses `~/.coelanox/clf/` as the root. Under it, four subdirectories are used:

| Subdirectory | Purpose |
|--------------|---------|
| `clfc` | Compute backends (e.g. `simd.clfc`). Place `.clfc` and compatible libraries here. |
| `clfmm` | Memory movement (CLFMM) artifacts. Reserved for future use. |
| `clfmp` | Memory protection (CLFMP) artifacts. Reserved for future use. |
| `clfe` | Extensions (CLFE). Reserved for future use. |

The root and these subdirectories are created automatically when the default path is used. Place compute backends in `~/.coelanox/clf/clfc/` so they are discovered without setting any environment variable.

**Override:** Set `COELANOX_CLF_PATH` to a directory (the new root for the four subdirectories) or to the path of a single `.clfc` file (its parent directory is used for discovery). Use this for development or custom installs.

**Additional paths:** Set `COELANOX_BACKEND_PATH` to a colon-separated list of directories to search in addition to the default or overridden CLF path.

---

## FAQ

**Q: Where does COELANOX look for CLF backends?**  
A: `~/.coelanox/clf/clfc/` by default. Override with `COELANOX_CLF_PATH`. See Backend discovery above.

**Q: Why is execution slow?**  
A: If no CLF/SIMD code was packaged (or CLF path is wrong), the runtime uses scalar fallback. Run `coelanox env` to check; ensure `.clfc` files are in the discovery path if you expect SIMD.

**Q: How do I get the version?**  
A: `coelanox --version` for CLI; crate versions are in `coelanox/*/Cargo.toml`.

---

## Runbooks

### Verification failed

**Symptom:** `coelanox verify` or `coelanox run` reports container integrity verification failed (hash mismatch).

**Steps:**

1. Do **not** load or run the container; treat it as untrusted.
2. Re-download or re-copy the container from a trusted source.
3. Run `coelanox verify --file <path>` again.
4. If it still fails, the file is corrupted or tampered; discard and obtain a new copy.
5. Log the event (path, error); alert if this occurs in production.

**Prevention:** Use integrity checks in CI/CD; verify after download and before deploy.

---

### Container load OOM / size limit exceeded

**Symptom:** Load fails with "Container size exceeds maximum allowed" or process OOM when loading.

**Steps:**

1. Check `COELANOX_MAX_CONTAINER_SIZE_BYTES` and `COELANOX_MAX_MEMORY_BYTES`. Increase only if the container is trusted and the host has capacity.
2. For very large models, ensure the host has enough RAM and that no other process is exhausting memory.
3. Consider using a smaller or quantized model, or split workloads.

---

### Execution timeout

**Symptom:** Inference fails with "Execution timeout exceeded (COELANOX_EXECUTION_TIMEOUT_MS)" or does not complete within the configured timeout.

The runtime enforces the deadline between execution steps (SIMD plan steps or scalar waves); if the deadline is exceeded, inference is aborted and an error is returned.

**Steps:**

1. Increase `COELANOX_EXECUTION_TIMEOUT_MS` (or equivalent in config file) if the workload legitimately needs more time.
2. Check for resource contention (CPU, memory, disk).
3. Try scalar backend to rule out backend-specific hangs: `coelanox run ... --backend scalar`.
4. Inspect logs for errors or warnings before the timeout.

---

### Backend fallback to scalar

**Symptom:** Logs show "falling back to scalar" or execution is slower than expected.

**Steps:**

1. Confirm CLF/SIMD path is available: set `COELANOX_CLF_PATH` to the directory (or file) containing the `.clfc` or CLF artifacts used at package time.
2. Run `coelanox env` and check recommended target and backend discovery.
3. If you intended scalar-only (e.g. portability), no action needed.
4. If vendor backend failed, check backend discovery paths and file permissions; re-package with the correct target if needed.

---

### Absolute path rejected (production)

**Symptom:** Load fails with "Absolute path not allowed".

**Steps:**

1. This is intentional when `COELANOX_ALLOW_ABSOLUTE_PATHS=false` (production hardening).
2. Use relative paths or a path under a known root (e.g. a mounted volume with a fixed prefix).
3. If you must allow absolute paths (e.g. dev only), set `COELANOX_ALLOW_ABSOLUTE_PATHS=true`; do not use in production unless required.

---

## Deployment targets

Same `.cnox` artifact works across environments; verify once, deploy per target.

| Environment | Container cache | Notes |
|-------------|-----------------|-------|
| **Server** | `Arc<CoelanoxContainer>` per entry; TTL and max_containers eviction | Warm cache holds decompressed weights; `COELANOX_MAX_CONTAINERS` and `COELANOX_CONTAINER_CACHE_TTL_SECS` control eviction |
| **Edge** | Same runtime, typically one/few containers | Same verify-then-run flow; optional `coelanox extract` for runtime-only binary |
| **Embedded** | No runtime cache | Minimal shim: load `.cnox`, parse plan, run; container is self-contained (IR + weights + code) |

Server uses Arc sharing for loaded containers so requests get pointer clones, not full copies. See `coelanox-orchestrator/src/runtime.rs` for implementation.

---

## Pre-deployment checklist

- [ ] Run `coelanox verify --file <container>` on every container before deploy.
- [ ] Set production limits: `COELANOX_MAX_CONTAINER_SIZE_BYTES`, `COELANOX_MAX_MEMORY_BYTES`, `COELANOX_EXECUTION_TIMEOUT_MS`, `COELANOX_MAX_CONTAINERS`.
- [ ] In production, set `COELANOX_ALLOW_ABSOLUTE_PATHS=false` unless you have a specific need.
- [ ] Use `COELANOX_CONFIG_FILE` or env vars so the runtime never runs with implicit defaults in production.
- [ ] Ensure `RUST_LOG` (or `COELANOX_LOG_LEVEL`) is set so security and failure events are visible (e.g. `info` or `warn`).

---

## Production readiness (short assessment)

| Aspect | Status | Notes |
|--------|--------|--------|
| **CLI for batch / single-run** | ✅ Ready | `run`, `verify`, `benchmark`, `debug` with config from env/file; integrity check, limits, and security logging in place. |
| **Config and limits** | ✅ Ready | `RuntimeConfig::from_env()` / `from_file()`; SecurityValidator enforces path and size; runbooks in this doc. |
| **Long-running server** | ⚠️ Partial | Runtime supports cache, TTL, timeouts; no HTTP/gRPC server, health endpoint, or metrics export in-tree. Use CLI in scripts or wrap in your own service. |
| **Observability** | ⚠️ Partial | Structured logging (tracing) and security events; no built-in Prometheus/OpenTelemetry. Rely on stdout + RUST_LOG and your log pipeline. |
| **Hardening** | ✅ Done | Lock expect() instead of unwrap(), security event logging, optional strict path policy, container size cap. |
| **Vendor / multi-backend** | ⚠️ Roadmap | Backend manager and CLF path work; AMD reference and full vendor SDK still on roadmap. |

### Summary

COELANOX is **ready for production in a “batch or scripted” sense**: verify then run (or benchmark) with explicit config and limits, in CI or on a server where you control the environment. For a **long-lived inference service** with health checks and metrics, you would run the CLI from your own process (or use the orchestrator as a library) and add your API and observability layer.

---

## References

- Security covered in this document (hash, signing, limits)
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design
- [Production MVP Roadmap](../roadmaps/PRODUCTION_MVP_ROADMAP.md) — Roadmap and phases
