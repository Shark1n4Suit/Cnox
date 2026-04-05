# COELANOX CLI Reference

Complete reference for all commands and options.

- **CLI version:** Run `coelanox --version` for current version
- **Invocation:** `coelanox [COMMAND] [OPTIONS]`

---

## Top-level options (global)

| Option | Short | Description |
|--------|--------|--------------|
| `--help` | `-h` | Print help (this message or the given subcommand) |
| `--version` | `-V` | Print version |

---

## 1. `package` — Package a model into a COELANOX container

**Usage:** `coelanox package [OPTIONS] --input <INPUT> --output <OUTPUT>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--input` | `-i` | `<INPUT>` | Input model file (path) |
| `--output` | `-o` | `<OUTPUT>` | Output COELANOX file (path) |

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--model-type` | `-m` | `<MODEL_TYPE>` | `cv` | Model type: `cv`, `nlp`, `audio`, `tabular`, `custom` |
| `--input-shapes` | — | `<INPUT_SHAPES>` | — | Input shapes for custom models, e.g. `"[1,3,224,224],[1,1,28,28]"` |
| `--output-shapes` | — | `<OUTPUT_SHAPES>` | — | Output shapes for custom models, e.g. `"[1,1000],[1,10]"` |
| `--target` | `-t` | `<TARGET>` | `auto` | Target hardware: `cpu`, `gpu`, `edge`, `auto`, or e.g. `h100`, `a100`, `amd-gpu`, `intel-npu` |
| `--optimization` | — | `<OPTIMIZATION>` | `advanced` | Optimization level: `none`, `basic`, `advanced`, `maximum` |
| `--use-demo-translator` | — | flag | false | Use demo/custom translator (e.g. BERT translator for NLP) |
| `--fat` | — | flag | false | Include all available backends (fat container). Mutually exclusive with `--target` and `--auto` |
| `--auto` | — | flag | false | Auto-detect hardware and select backends. Mutually exclusive with `--target` and `--fat` |
| `--backend-add` | — | `<BACKEND_ADD>` | — | Add additional backends, comma-separated: e.g. `amd,intel,qualcomm` |
| `--fallback-only` | — | flag | false | Package without backend code (scalar fallback only at runtime; for benchmarking) |
| `--audit-required` | — | flag | false | Set a **container header flag** so every `run` / `serve` **must** emit audit logs (compliance). If the user does not pass `--audit-output`, the runtime writes under `./coelanox_audit/` (default path pattern); audit visibility options still apply when audit is active |
| `--max-workspace` | — | `<MAX_WORKSPACE>` | — | Max dynamic memory (workspace + scratch) in MB; build fails if model needs more (e.g. 64, 1024) |
| `--signing-key` | — | `<PATH>` | — | Path to Ed25519 secret key (32 raw bytes) to sign container. Env: `COELANOX_SIGNING_KEY` |
| `--help` | `-h` | — | — | Print help |

**Mutual exclusivity:** Only one of `--fat`, `--auto`, or explicit `--target` may be used.

---

## 2. `validate` — Validate a COELANOX container

**Usage:** `coelanox validate --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to validate |

**Optional:**

| Option | Short | Description |
|--------|--------|-------------|
| `--help` | `-h` | Print help |

---

## 3. `keygen` — Generate Ed25519 keypair for container signing

Produces `secret.bin` (32 bytes, keep private) and `public.bin` (32 bytes, share with verifiers). Use with `package --signing-key` and `verify --trusted-key`.

**Usage:** `coelanox keygen [OPTIONS]`

**Optional:** `--output-dir` / `-o` (output directory; default: current dir), `--help`

---

## 4. `verify` — Verify container integrity (pre-deployment check)

**Usage:** `coelanox verify [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to verify |

**Optional:**

| Option | Short | Type | Description |
|--------|--------|------|-------------|
| `--verbose` | `-v` | flag | Enable verbose output |
| `--trusted-key` | — | `<PATH>` | Path to Ed25519 public key for signature verification. Env: `COELANOX_TRUSTED_PUBLIC_KEY` |
| `--require-signature` | — | flag | Fail if container is signed but no trusted key provided |
| `--help` | `-h` | — | Print help |

---

## 5. `info` — Show information about a COELANOX container

**Usage:** `coelanox info [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to inspect |

**Optional:**

| Option | Short | Type | Description |
|--------|--------|------|-------------|
| `--dump-plan` | — | flag | Dump execution plan conv steps (step_ix, in_len, out_len, w_len, scratch_raw) to find bad steps |
| `--list-ops` | — | flag | List IR operation counts; highlights Custom ops (not supported on scalar backend) |
| `--help` | `-h` | — | Print help |

---

## 6. `run` — Run inference on a COELANOX container

**Usage:** `coelanox run [OPTIONS] --file <FILE> --output <OUTPUT>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to run |
| `--output` | `-o` | `<OUTPUT>` | Output file (e.g. JSON) |

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--input` | `-i` | `<INPUT>` | — | Input data file (JSON). If omitted, synthetic input is used |
| `--backend` | `-b` | `<BACKEND>` | `auto` | Backend: `auto`, `cpu`, `gpu`, `npu`, `tpu`. Use `auto` to match benchmark behaviour |
| `--audit` | — | flag | false | Audit mode: after each op, log to stderr and optionally to file (structured batch; see below) |
| `--audit-output` | — | `<AUDIT_OUTPUT>` | — | When using `--audit`, write audit batches to this file |
| `--audit-strategy` | — | `<forget\|last\|long>` | `forget` | **`forget`** (default): clear RAM after each flush; **append** each batch to disk (paths ending in `.json` use **JSONL**: one **JSON object per line**, schema version **2** — `run`, optional `input`, and `ops[]`). **`last`**: overwrite file with the last batch only. **`long`**: session-cumulative in RAM; each flush rewrites the full snapshot (legacy; unbounded memory on long `serve`) |
| `--audit-sample` | — | `<N>` | `4` | Number of **leading** floats to record per op output (combined with `--audit-tail`, capped internally) |
| `--audit-tail` | — | `<M>` | `0` | Number of **trailing** floats per op output (non-overlapping with the head when the tensor is long enough) |
| `--audit-stats` | — | flag | false | Include min / max / mean and NaN / Inf counts per tensor (full scan only up to an internal element cap per tensor) |
| `--audit-hash` | — | flag | false | Include SHA-256 (hex) over raw little-endian **`f32`** bytes for the **input snapshot** and each op output (same cap as stats; large tensors may omit hash with an elision note) |
| `--audit-final-only` | — | flag | false | Only log ops whose outputs are **graph sinks** in the IR (intermediate tensors not consumed as inputs elsewhere). If the IR yields no sinks, op lines may be empty while run/input metadata still appear |
| `--max-workspace-mb` | — | `<MAX_WORKSPACE_MB>` | — | Max workspace in MB (overrides `COELANOX_MAX_WORKSPACE_MB`). 0 = no cap |
| `--no-verify` | — | flag | false | Skip integrity verification (dev only; avoid in production) |
| `--trusted-key` | — | `<PATH>` | — | Path to Ed25519 public key for signature verification. Env: `COELANOX_TRUSTED_PUBLIC_KEY` |
| `--require-signature` | — | flag | false | Fail if container is signed but no trusted key provided |
| `--help` | `-h` | — | — | Print help |

**Audit required by container:** If the `.cnox` was packaged with **`package --audit-required`**, **`run`** and **`serve`** turn on audit logging **even when `--audit` is omitted** (header flag). Output path: **`--audit-output`** if set, otherwise the default under **`./coelanox_audit/`** (see runtime messages at startup).

---

## 7. `serve` — Long-lived inference (binary IPC on stdin/stdout)

Loads **one** `.cnox` (with the same verify/signature behaviour as `run`), then stays alive and exchanges **framed** requests on **stdin** and responses on **stdout`. **stderr** is used for logs and a startup line `COELANOX_SERVE_READY …` (machine-readable). Tracing is written to stderr only so stdout stays binary-clean.

**Usage:** `coelanox serve [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | Path to the `.cnox` file |

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--backend` | `-b` | `<BACKEND>` | `auto` | Backend used for each inference (e.g. `auto`, `scalar`) |
| `--audit` | — | flag | false | Same as `run`: structured audit batches (stderr + optional file) |
| `--audit-output` | — | `<PATH>` | — | Audit log file when using `--audit` |
| `--audit-strategy` | — | `<forget\|last\|long>` | `forget` | Same as `run` (JSONL schema v2 for `.json` paths) |
| `--audit-sample` | — | `<N>` | `4` | Same as `run` |
| `--audit-tail` | — | `<M>` | `0` | Same as `run` |
| `--audit-stats` | — | flag | false | Same as `run` |
| `--audit-hash` | — | flag | false | Same as `run` |
| `--audit-final-only` | — | flag | false | Same as `run` |
| `--max-workspace-mb` | — | `<N>` | — | Workspace cap (overrides `COELANOX_MAX_WORKSPACE_MB`; `0` = unlimited) |
| `--no-verify` | — | flag | false | Skip hash verification on load (development only) |
| `--trusted-key` | — | `<PATH>` | — | Ed25519 public key for signature verification |
| `--require-signature` | — | flag | false | Error if container is signed but no trusted key is supplied |
| `--help` | `-h` | — | — | Print help |

Containers built with **`package --audit-required`** enforce audit the same way as **`run`** (see note under §6 **`run`**). With **`--audit`** and **`--audit-output`**, the server **flushes audit to disk after each inference request** (success or failure) so long-lived processes still produce on-disk evidence; parent directories are created if needed. Write failures are logged to stderr. Default **`--audit-strategy forget`** bounds RAM per request and **appends** each batch (use **`long`** only if you need legacy session-cumulative behavior). **`serve` accepts the same audit visibility flags as `run`** (`--audit-sample`, `--audit-tail`, `--audit-stats`, `--audit-hash`, `--audit-final-only`): session metadata (container path, backend, container file hash, run id) is set at process start, while the **input** block in each JSONL line reflects the **payload of that `CREQ`** (recorded immediately before executing that request).

**IPC protocol (v1):** Clients send **`CREQ`** (4 bytes) + **`u32` payload length (LE)** + **raw input tensor as `f32` LE** (total element count must match the manifest). Server replies with **`CROk`** + length + output `f32` bytes, or **`CREr`** + error code + message. Send **`CNXT`** (4 bytes) to shut down. Payloads are capped (see `coelanox-ipc` / `serve.rs`). Full details: source file `coelanox/coelanox-ipc/src/serve.rs`.

---

## 8. `benchmark` — Benchmark a COELANOX container

**Usage:** `coelanox benchmark [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to benchmark |

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--iterations` | `-i` | `<ITERATIONS>` | 100 | Number of iterations |
| `--loop-iterations` | — | `<LOOP_ITERATIONS>` | — | Loop for N iterations (long-run stability); overrides `--iterations` when set |
| `--backend` | — | `<BACKEND>` | `auto` | Backend to benchmark (e.g. `auto`, `scalar`) |
| `--output` | — | `<OUTPUT>` | `text` | Output format: `text` or `json` |
| `--max-workspace-mb` | — | `<MAX_WORKSPACE_MB>` | — | Max workspace in MB (overrides `COELANOX_MAX_WORKSPACE_MB`). 0 = no cap |
| `--no-verify` | — | flag | false | Skip integrity verification (dev only) |
| `--trusted-key` | — | `<PATH>` | — | Path to Ed25519 public key for signature verification |
| `--require-signature` | — | flag | false | Fail if container is signed but no trusted key provided |
| `--help` | `-h` | — | — | Print help |

---

## 9. `env` — Show system environment information

**Usage:** `coelanox env`

**Optional:**

| Option | Short | Description |
|--------|--------|-------------|
| `--help` | `-h` | Print help |

No required arguments. Prints OS, architecture, CPU, memory, GPU (if any), SIMD capabilities, optional toolchain hints, and heuristic packaging suggestions.

---

## 10. `uninstall` — Remove CLI / print uninstall instructions

Does **not** delete the running binary automatically. Prints the resolved executable path, typical removal steps (`cargo uninstall coelanox-cli`, PATH hints), and optional cleanup.

**Usage:** `coelanox uninstall [OPTIONS]`

**Optional:**

| Option | Short | Description |
|--------|--------|-------------|
| `--purge-audit` | — | Delete `./coelanox_audit` in the **current working directory** if it exists (local audit logs) |
| `--help` | `-h` | Print help |

---

## 11. `debug` — Debug a COELANOX container with detailed diagnostics

**Usage:** `coelanox debug [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to debug |

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--input` | `-i` | `<INPUT>` | — | Input data file (JSON) |
| `--output` | `-o` | `<OUTPUT>` | `debug_output.json` | Output file for debug logs |
| `--gdb` | — | flag | false | Enable GDB-style debugging (requires GDB) |
| `--dump-registers` | — | flag | false | Dump register state on crash |
| `--trace-limit` | `-t` | `<TRACE_LIMIT>` | 10 | Maximum operations to trace before crash |
| `--help` | `-h` | — | — | Print help |

---

## 12. `extract` — Copy runnable binaries for deployment

**Same-arch only.** Cross-arch (`--arm` / `--target`) is **not yet implemented** (clear error).

There are two modes:

1. **Default (full CLI):** Copies this **`coelanox` executable** to `-o` (default filename **`coelanox`**). The copy is the full toolchain, including **`coelanox serve`** (IPC v1 on stdin/stdout) and all other subcommands—useful when you want one binary that matches your build.

2. **`--minimal`:** Copies **`coelanox-run`** from the **same directory** as this `coelanox` binary (e.g. both in `target/release/` after `cargo build --release -p coelanox-cli -p coelanox-standalone`, or both in a release tarball). That minimal runner loads one `.cnox` per invocation and supports **`coelanox-run model.cnox --serve`** with the **same IPC framing** as `coelanox serve`. If `coelanox-run` is missing beside `coelanox`, the command fails with a build hint.

**Usage:** `coelanox extract [OPTIONS]`

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--output` | `-o` | `<OUTPUT>` | `coelanox` | Destination path for the copied file |
| `--minimal` | — | flag | false | Copy sibling **`coelanox-run`** instead of the full CLI (e.g. `-o coelanox-run`) |
| `--arm` | — | flag | false | **Planned:** ARM64 runtime; currently errors (use CI-built artifacts) |
| `--target` | — | `<TARGET>` | — | **Planned:** cross-arch triple; currently errors |
| `--help` | `-h` | — | — | Print help |

---

## 13. `help` — Print help

**Usage:** `coelanox help [SUBCOMMAND]`

- `coelanox help` — top-level help
- `coelanox help package` — help for `package`, etc.

---

## Command tree (for navigation / sitemap)

```
coelanox
├── -h, --help
├── -V, --version
├── package     (-i, -o required; -m, -t, --use-demo-translator, --fat, --auto, --backend-add, --fallback-only, --audit-required, --max-workspace, …)
├── validate    (-f required)
├── keygen      (-o output-dir)
├── verify      (-f required; -v, --trusted-key, --require-signature)
├── info        (-f required; --dump-plan, --list-ops)
├── run         (-f, -o required; -i, -b, --audit, --audit-output, --audit-strategy, --audit-sample, --audit-tail, --audit-stats, --audit-hash, --audit-final-only, --max-workspace-mb)
├── serve       (-f required; -b, --audit, --audit-output, --audit-strategy, --audit-sample, --audit-tail, --audit-stats, --audit-hash, --audit-final-only, --max-workspace-mb, --no-verify, --trusted-key, --require-signature)
├── benchmark   (-f required; -i, --loop-iterations, --backend, --output, --max-workspace-mb)
├── env         (no args)
├── uninstall   (--purge-audit)
├── debug       (-f required; -i, -o, --gdb, --dump-registers, -t)
├── extract     (-o, --minimal, --arm, --target)
└── help        [SUBCOMMAND]
```

---

## Environment variables

- **`COELANOX_MAX_WORKSPACE_MB`** — Max workspace in MB (used by `run`, `serve`, and `benchmark` unless overridden by `--max-workspace-mb`).
- **`COELANOX_CLF_PATH`** — Path(s) for CLF backend discovery (packager and runtime).

---

## Example invocations (copy-paste for your site)

```bash
# Keygen (Ed25519 keypair for signing)
coelanox keygen -o ./keys

# Package (e.g. BERT from bundle, scalar-only)
coelanox package -i model.bundle -o model.cnox -m nlp --use-demo-translator --fallback-only

# Validate
coelanox validate -f model.cnox

# Verify integrity
coelanox verify -f model.cnox -v

# Info (with execution plan dump)
coelanox info -f model.cnox --dump-plan

# Run inference
coelanox run -f model.cnox -o out.json
coelanox run -f model.cnox -i input.json -o out.json -b auto

# Serve (long-lived IPC; integrate via framed stdin/stdout — see §7)
coelanox serve -f model.cnox -b auto

# Benchmark (text or JSON)
coelanox benchmark -f model.cnox -i 50
coelanox benchmark -f model.cnox --output json

# Environment
coelanox env

# Uninstall instructions (optional: remove local audit folder)
coelanox uninstall
coelanox uninstall --purge-audit

# Debug
coelanox debug -f model.cnox -o debug.json --trace-limit 5

# Extract full CLI copy (default output name: coelanox)
coelanox extract -o ./deploy/coelanox

# Extract minimal runner (requires coelanox-run built next to this binary)
coelanox extract --minimal -o ./deploy/coelanox-run
```

Use this doc to drive dropdowns, “command builder” UIs, and deep links to each command and option on your website.
