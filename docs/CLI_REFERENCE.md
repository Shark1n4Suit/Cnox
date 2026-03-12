# COELANOX CLI Reference

Complete reference for all commands and options.

- **CLI version:** 0.6.0-beta  
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
| `--max-workspace` | — | `<MAX_WORKSPACE>` | — | Max dynamic memory (workspace + scratch) in MB; build fails if model needs more (e.g. 64, 1024) |
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

## 3. `verify` — Verify container integrity (pre-deployment check)

**Usage:** `coelanox verify [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to verify |

**Optional:**

| Option | Short | Type | Description |
|--------|--------|------|-------------|
| `--verbose` | `-v` | flag | Enable verbose output |
| `--help` | `-h` | — | Print help |

---

## 4. `info` — Show information about a COELANOX container

**Usage:** `coelanox info [OPTIONS] --file <FILE>`

**Required:**

| Argument | Short | Type | Description |
|----------|--------|------|--------------|
| `--file` | `-f` | `<FILE>` | COELANOX file to inspect |

**Optional:**

| Option | Short | Type | Description |
|--------|--------|------|-------------|
| `--dump-plan` | — | flag | Dump execution plan conv steps (step_ix, in_len, out_len, w_len, scratch_raw) to find bad steps |
| `--help` | `-h` | — | Print help |

---

## 5. `run` — Run inference on a COELANOX container

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
| `--audit` | — | flag | false | Audit mode: log after each op (op index, type, output size, sample to stderr and optionally to file) |
| `--audit-output` | — | `<AUDIT_OUTPUT>` | — | When using `--audit`, append audit log lines to this file |
| `--max-workspace-mb` | — | `<MAX_WORKSPACE_MB>` | — | Max workspace in MB (overrides `COELANOX_MAX_WORKSPACE_MB`). 0 = no cap |
| `--help` | `-h` | — | — | Print help |

---

## 6. `benchmark` — Benchmark a COELANOX container

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
| `--output` | — | `<OUTPUT>` | `text` | Output format: `text` or `json` |
| `--max-workspace-mb` | — | `<MAX_WORKSPACE_MB>` | — | Max workspace in MB (overrides `COELANOX_MAX_WORKSPACE_MB`). 0 = no cap |
| `--help` | `-h` | — | — | Print help |

---

## 7. `env` — Show system environment information

**Usage:** `coelanox env`

**Optional:**

| Option | Short | Description |
|--------|--------|-------------|
| `--help` | `-h` | Print help |

No required arguments. Prints OS, architecture, CPU, memory, GPU (if any), SIMD capabilities.

---

## 8. `debug` — Debug a COELANOX container with detailed diagnostics

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

## 9. `extract` — Extract runtime-only binary for deployment

Produces a **standalone runtime** that matches this CLI version; deploy it with `.cnox` files only. **Same-arch extract is implemented today.** Cross-arch (`--arm` / `--target`) is **not yet implemented** and will currently print a warning and exit.

**Usage:** `coelanox extract [OPTIONS]`

**Optional:**

| Option | Short | Type | Default | Description |
|--------|--------|------|---------|-------------|
| `--output` | `-o` | `<OUTPUT>` | `coelanox-run` | Output path for the runtime binary (e.g. `coelanox-run` or `coelanox-run-arm`) |
| `--arm` | — | flag | false | **Planned:** request ARM64 (aarch64) runtime; currently emits a "not yet implemented" message |
| `--target` | — | `<TARGET>` | — | **Planned:** target triple for cross-arch (e.g. `aarch64-unknown-linux-gnu`); overrides `--arm` if set; currently emits a "not yet implemented" message |
| `--help` | `-h` | — | — | Print help |

---

## 10. `help` — Print help

**Usage:** `coelanox help [SUBCOMMAND]`

- `coelanox help` — top-level help
- `coelanox help package` — help for `package`, etc.

---

## Command tree (for navigation / sitemap)

```
coelanox
├── -h, --help
├── -V, --version
├── package     (-i, -o required; -m, -t, --use-demo-translator, --fat, --auto, --backend-add, --fallback-only, --max-workspace, …)
├── validate    (-f required)
├── verify      (-f required; -v)
├── info        (-f required; --dump-plan)
├── run         (-f, -o required; -i, -b, --audit, --audit-output, --max-workspace-mb)
├── benchmark   (-f required; -i, --loop-iterations, --output, --max-workspace-mb)
├── env         (no args)
├── debug       (-f required; -i, -o, --gdb, --dump-registers, -t)
├── extract     (-o, --arm, --target)
└── help        [SUBCOMMAND]
```

---

## Environment variables

- **`COELANOX_MAX_WORKSPACE_MB`** — Max workspace in MB (used by `run` and `benchmark` unless overridden by `--max-workspace-mb`).
- **`COELANOX_CLF_PATH`** — Path(s) for CLF backend discovery (packager and runtime).

---

## Example invocations (copy-paste for your site)

```bash
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

# Benchmark (text or JSON)
coelanox benchmark -f model.cnox -i 50
coelanox benchmark -f model.cnox --output json

# Environment
coelanox env

# Debug
coelanox debug -f model.cnox -o debug.json --trace-limit 5

# Extract standalone runtime (e.g. for ARM)
coelanox extract -o coelanox-run
coelanox extract -o coelanox-run-arm --arm
```

Use this doc to drive dropdowns, “command builder” UIs, and deep links to each command and option on your website.
