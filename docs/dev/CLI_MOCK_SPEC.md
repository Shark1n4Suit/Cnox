# Coelanox CLI — Full Mock Spec for Website

Use this to **mock the entire CLI** in the browser: parsing rules, all commands, and realistic stdout/stderr so the in-site terminal behaves like the real `coelanox` binary.

**Version:** `0.7.5`  
**Binary name:** `coelanox`

---

## 1. How the CLI works

- **Invocation:** `coelanox [OPTIONS] <COMMAND> [COMMAND_OPTIONS]`
- **Global options** (before the subcommand): only `-h`/`--help` and `-V`/`--version`. No other top-level flags.
- **Subcommands:** One of `package` | `validate` | `keygen` | `verify` | `info` | `run` | `serve` | `benchmark` | `env` | `uninstall` | `debug` | `extract` | `help`.
- **Parsing:** Options use `-` or `--`; short form is single letter (e.g. `-f`), long form is `--file`. Values for options with a placeholder go in the next token or `=VALUE` (e.g. `--file model.cnox` or `--file=model.cnox`).
- **Help:** `coelanox`, `coelanox -h`, `coelanox --help`, and `coelanox help` all show top-level help. `coelanox help <COMMAND>` shows that command's help.
- **Version:** `coelanox -V` or `coelanox --version` prints version only; no subcommand.

---

## 2. Command list (what to parse)

| Command     | Required args              | Optional args |
|------------|----------------------------|----------------|
| `package`  | `-i`/`--input`, `-o`/`--output` | `-m`, `--input-shapes`, `--output-shapes`, `-t`, `--optimization`, `--use-demo-translator`, `--fat`, `--auto`, `--backend-add`, `--fallback-only`, `--max-workspace` |
| `validate` | `-f`/`--file`              | — |
| `verify`   | `-f`/`--file`              | `-v`/`--verbose`, `--trusted-key`, `--require-signature` |
| `info`     | `-f`/`--file`              | `--dump-plan`, `--list-ops` |
| `run`      | `-f`/`--file`, `-o`/`--output` | `-i`, `-b`, `--audit`, `--audit-output`, `--max-workspace-mb`, `--no-verify`, `--trusted-key`, `--require-signature` |
| `serve`    | `-f`/`--file`              | `-b`, `--audit`, `--audit-output`, `--max-workspace-mb`, `--no-verify`, `--trusted-key`, `--require-signature` |
| `benchmark`| `-f`/`--file`              | `-i`, `--loop-iterations`, `--backend`, `--output`, `--max-workspace-mb`, `--no-verify`, `--trusted-key`, `--require-signature` |
| `env`      | —                          | — |
| `uninstall`| —                          | `--purge-audit` |
| `debug`    | `-f`/`--file`              | `-i`, `-o`, `--gdb`, `--dump-registers`, `-t` |
| `extract`  | —                          | `-o`, `--minimal`, `--arm`, `--target` |
| `keygen`   | —                          | `-o`/`--output-dir` |
| `help`     | —                          | `[SUBCOMMAND]` |

**Mutual exclusivity (package only):** At most one of `--fat`, `--auto`, or explicit `--target` (other than `auto`).

---

## 3. Sample outputs (for mock)

| Command | Success output |
|---------|----------------|
| (none), `-h`, `help` | Top-level usage: "Usage: coelanox \<COMMAND\>" + command list |
| `-V` | `coelanox 0.7.5` |
| `validate -f X.cnox` | `✅ Container is valid` |
| `verify -f X.cnox` | `✅ Container integrity verified` |
| `run -f X.cnox -o out.json` | `Execution time: N μs` |
| `info -f X.cnox` | Model name, shapes, sizes; substitute `{{FILE}}` with path |

Use [CLI_REFERENCE.md](../CLI_REFERENCE.md) for full option details; run `cargo run -q -p coelanox-cli -- CMD --help` (from `coelanox/`) for live help.

---

## 4. Exit codes

- **0** — Success.
- **1** — Error (file not found, validation failed, etc.).
- **2** — Usage/parse error (missing required, unknown option).
