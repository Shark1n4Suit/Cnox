# Coelanox CLI — Full Mock Spec for Website

Use this to **mock the entire CLI** in the browser: parsing rules, all commands, and realistic stdout/stderr so the in-site terminal behaves like the real `coelanox` binary.

**Version:** `0.6.0-beta`  
**Binary name:** `coelanox`

---

## 1. How the CLI works

- **Invocation:** `coelanox [OPTIONS] <COMMAND> [COMMAND_OPTIONS]`
- **Global options** (before the subcommand): only `-h`/`--help` and `-V`/`--version`. No other top-level flags.
- **Subcommands:** One of `package` | `validate` | `verify` | `info` | `run` | `benchmark` | `env` | `debug` | `extract` | `help`.
- **Parsing:** Options use `-` or `--`; short form is single letter (e.g. `-f`), long form is `--file`. Values for options with a placeholder go in the next token or `=VALUE` (e.g. `--file model.cnox` or `--file=model.cnox`).
- **Help:** `coelanox`, `coelanox -h`, `coelanox --help`, and `coelanox help` all show top-level help. `coelanox help <COMMAND>` shows that command’s help.
- **Version:** `coelanox -V` or `coelanox --version` prints version only; no subcommand.

---

## 2. Command list (what to parse)

| Command     | Required args              | Optional args |
|------------|----------------------------|----------------|
| `package`  | `-i`/`--input`, `-o`/`--output` | `-m`, `--input-shapes`, `--output-shapes`, `-t`, `--optimization`, `--use-demo-translator`, `--fat`, `--auto`, `--backend-add`, `--fallback-only`, `--max-workspace` |
| `validate` | `-f`/`--file`              | — |
| `verify`   | `-f`/`--file`              | `-v`/`--verbose` |
| `info`     | `-f`/`--file`              | `--dump-plan` |
| `run`      | `-f`/`--file`, `-o`/`--output` | `-i`, `-b`, `--audit`, `--audit-output`, `--max-workspace-mb` |
| `benchmark`| `-f`/`--file`              | `-i`, `--loop-iterations`, `--output`, `--max-workspace-mb` |
| `env`      | —                          | — |
| `debug`    | `-f`/`--file`              | `-i`, `-o`, `--gdb`, `--dump-registers`, `-t` |
| `extract`  | —                          | `-o`, `--arm`, `--target` |
| `help`     | —                          | `[SUBCOMMAND]` |

**Mutual exclusivity (package only):** At most one of `--fat`, `--auto`, or explicit `--target` (other than `auto`).

---

## 3. Sample outputs for mock (copy-paste or template)

### 3.1 No args or `coelanox -h` / `coelanox --help` / `coelanox help`

```
COELANOX - Secure AI Container

Usage: coelanox <COMMAND>

Commands:
  package    Package a model into a COELANOX container
  validate   Validate a COELANOX container
  verify     Verify container integrity (pre-deployment check)
  info       Show information about a COELANOX container
  run        Run inference on a COELANOX container
  benchmark  Benchmark a COELANOX container
  env        Show system environment information
  debug      Debug a COELANOX container with detailed diagnostics
  extract    Extract runtime-only binary for deployment (same arch or cross-arch, e.g. ARM server). Produces a standalone runtime that matches this CLI version; deploy it with .cnox files only
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### 3.2 `coelanox -V` / `coelanox --version`

```
coelanox 0.6.0-beta
```

### 3.3 `coelanox env`

```
=== System Environment ===
OS: Ubuntu 24.04.3 LTS
Architecture: x86_64
CPU Vendor: AuthenticAMD
CPU Model: AMD Ryzen 7 9700X 8-Core Processor
CPU Cores: 16
Memory: 15 GB

=== GPU Information ===
Vendor: NVIDIA
Model: NVIDIA GeForce RTX 5070
Memory: 11 GB
Driver Version: 591.74

=== SIMD Capabilities ===
  SSE
  SSE2
  SSE3
  SSE41
  SSE42
  AVX
  AVX2
  AVX512

=== System Capabilities ===
  vulkan: ❌
  tensorrt: ❌
  onednn: ❌
  cuda: ❌
  opencl: ❌

=== Recommendations ===
Recommended target: Gpu
Recommended optimization: Maximum
```

*(For mock you can keep this as-is or substitute generic placeholders like "Host OS", "x86_64", "N/A", etc.)*

### 3.4 `coelanox validate -f model.cnox` (success)

```
✅ Container is valid
Version: 0.6.0-beta
Model: resnet_tiny_mnist
Format: Onnx
Size: 3176605 bytes
```

### 3.5 `coelanox validate -f nonexistent.cnox` (error)

```
Error: FileSystem(Os { code: 2, kind: NotFound, message: "No such file or directory" })
```

*(Exit code 1 for errors.)*

### 3.6 `coelanox verify -f model.cnox`

```
✅ Container integrity verified
```

### 3.7 `coelanox verify -f model.cnox -v` (verbose: same line or add one line like "Hash: ..." if you want)

```
✅ Container integrity verified
```

### 3.8 `coelanox info -f model.cnox`

```
=== COELANOX Container Information ===
File: /path/to/model.cnox
Version: 0.6.0-beta
Status: ✅ Valid

=== Model Information ===
Name: resnet_tiny_mnist
Format: Onnx
Created: 1771944831
Modified: 1771944831

=== Input/Output ===
Input shapes: [[1, 1, 28, 28]]
Output shapes: [[1, 10]]

=== Container Details ===
Total size: 3176605 bytes
Header size: 76 bytes
Manifest size: ~586 bytes
Machine code size: 27993 bytes
Workspace required: 414820 elements (~2 MB)

=== Performance ===
Compression ratio: N/A (original size not available)
```

*(For mock, replace `/path/to/model.cnox` with the path the user typed.)*

### 3.9 `coelanox run -f model.cnox -o out.json` (success)

```
Input size: 784 elements
Output size: 10 elements
Execution time: 106276 μs
```

### 3.10 `coelanox benchmark -f model.cnox -i 3` (success, text)

```
=== Benchmark Results ===
File: /path/to/model.cnox
Iterations: 3 (warm-up: 10)
Load time (to first execute-ready): 2 ms

=== Latency (μs) ===
  Average:  88583
  Min:      83157
  Max:      93032
  p50:      89560
  p95:      89560
  p99:      89560
  Std dev:  4090.22
  True cold (1st run, no warm-up): 106999 μs  (same metric as single "run" Execution time)
  First measured (after warm-up): 89560 μs
  Warm (avg after 1st measured): 88094 μs

=== Throughput ===
  Sustained: 11 inf/s
  Max burst: 12 inf/s (1/min_latency)

=== Memory ===
  Workspace (elements): 414820
  Weights (decompressed): 733992 bytes
  Executable memory: 0 bytes
  RSS after benchmark: 16207872 bytes (15.46 MiB)

=== Breakdown ===
  Load time: 2 ms
  Decompress (one-time): 6 ms
  Backend: plan
```

### 3.11 `coelanox debug -f model.cnox -o debug.json`

*(Minimal success: no stdout, or one line like "Debug output written to debug.json". For mock you can echo the command and say "Debug output written to <OUTPUT>".)*

### 3.12 `coelanox extract -o coelanox-run`

*(Success: no stdout, or one line like "Runtime extracted to coelanox-run".)*

### 3.13 `coelanox package -i in.onnx -o out.cnox` (success — mock only)

*(Real CLI prints packager progress. For mock you can use:)*

```
📁 Input: in.onnx
📁 Output: out.cnox
🔧 Creating packager...
✅ Packager created successfully
...
✅ Container written: out.cnox
```

### 3.14 `coelanox help package` (subcommand help)

*(Output = same as `coelanox package --help`: the full "Package a model into a COELANOX container" usage block with all options. Use the text from your existing CLI_REFERENCE.md or from `cargo run -q --bin coelanox -- package --help`.)*

---

## 4. Mock behavior summary

| User types              | Action |
|-------------------------|--------|
| (nothing) or `coelanox` | Show top-level usage (3.1). |
| `coelanox -h` / `--help` | Same as above. |
| `coelanox -V` / `--version` | Print version (3.2). |
| `coelanox help`         | Same as top-level usage. |
| `coelanox help <cmd>`    | Show that command’s help (e.g. `package --help` text). |
| `coelanox env`           | Show env block (3.3). |
| `coelanox validate -f <FILE>` | If FILE looks like a .cnox path: success (3.4); else error (3.5). |
| `coelanox verify -f <FILE>`  | Success (3.6); with `-v` same or slightly extended. |
| `coelanox info -f <FILE>`    | Success (3.8); substitute FILE into "File: ...". |
| `coelanox run -f <FILE> -o <OUT>` | Success (3.9); optionally substitute paths. |
| `coelanox benchmark -f <FILE> [-i N]` | Success (3.10); substitute FILE and optional iteration count. |
| `coelanox debug -f <FILE> [-o OUT]`   | Success (3.11). |
| `coelanox extract [-o OUT] [--arm]`   | Success (3.12). |
| `coelanox package -i <IN> -o <OUT> [opts]` | Success (3.13) for mock. |
| Missing required arg     | Print "Error: required argument ..." or show command help. |
| Unknown command         | Print "Error: unrecognized subcommand ..." and suggest help. |

---

## 5. Exit codes (for mock)

- **0** — Success.
- **1** — Error (file not found, validation failed, etc.).
- **2** — Usage/parse error (missing required, unknown option; clap often uses 2).

---

## 6. Optional: JSON for website

Attach the same structure as **cli_commands.json** (command id, required/optional args, types, defaults) so the site can:

- Build the command string from a form.
- Validate before "running" the mock.
- Show help per command from the same data.

Use **CLI_REFERENCE.md** for exact option names and **CLI_MOCK_SPEC.md** (this file) for invocation rules and sample outputs. Together they give you everything needed to mock the entire CLI on the website.

---

## 7. Mock output lookup (with cli_mock_outputs.json)

The file **cli_mock_outputs.json** contains ready-to-use strings for each response. Keys:

| Key | When to use |
|-----|----------------|
| `help_top` | No args, `-h`, `--help`, or `help` with no subcommand. |
| `version_out` | `-V` or `--version`. |
| `env_out` | Command is `env`. |
| `validate_ok` | Command is `validate` and file path looks valid (e.g. ends with `.cnox`). |
| `validate_err` | Command is `validate` and file is missing or invalid. |
| `verify_ok` | Command is `verify`. |
| `info_ok` | Command is `info`; replace `{{FILE}}` with the user’s `-f` value. |
| `run_ok` | Command is `run`; optional: substitute paths. |
| `benchmark_ok` | Command is `benchmark`; replace `{{FILE}}` and `{{ITERATIONS}}` (default 100). |
| `debug_ok` | Command is `debug`; replace `{{OUTPUT}}` (default `debug_output.json`). |
| `extract_ok` | Command is `extract`; replace `{{OUTPUT}}` (default `coelanox-run`). |
| `package_ok` | Command is `package`; replace `{{INPUT}}` and `{{OUTPUT}}`. |
| `usage_error` | Required option missing for the chosen command. |
| `unknown_command` | Unknown subcommand. |

**Template placeholders in JSON:** Replace `{{FILE}}`, `{{OUTPUT}}`, `{{INPUT}}`, `{{ITERATIONS}}` with the values from the parsed command (or defaults) before displaying.
