# Integration: inference output and “human text”

**Last updated:** 2026-04-13

**Audience:** Developers integrating COELANOX into scripts, services, or apps that need **files**, **pipes**, or **user-visible text**—not only tensors inside the runtime.

---

## 1. What COELANOX actually emits

At the orchestrator boundary, inference is **always numeric**: a **single flattened `f32` vector** in **row-major order** for all outputs combined, consistent with the container manifest’s **output shapes** (see `coelanox info -f model.cnox`).

COELANOX does **not** ship a tokenizer, detokenizer, or label vocabulary. Turning logits into **natural language**, **class names**, or **structured records** is **application logic** (Python, Rust, your service, etc.) on top of those floats.

---

## 2. `coelanox run` — JSON on disk (most common integration)

| Direction | Format |
|-----------|--------|
| **Input** (`-i` / `--input`) | JSON file containing **one JSON array** of `f32` values: `[0.1, 0.2, …]`. Total length must match the sum of `product(shape)` over **all** input tensors (flattened, row-major, manifest order). |
| **Output** (`-o` / `--output`) | JSON file: **pretty-printed** single array of `f32` values — same flattening rules for **outputs**. |

**Example**

```bash
coelanox run -f model.cnox -i input.json -o output.json
```

`output.json` is text you can open in an editor or parse with any JSON library:

```json
[
  0.004,
  0.991,
  0.005
]
```

**Human-readable summary on the terminal:** the CLI also prints a short **summary to stdout** (element counts, timing, path to `output.json`). That is **log text**, not the tensor payload.

---

## 3. `coelanox serve` — binary on stdout (not JSON text)

For `coelanox serve`, **stdout is reserved for framed binary IPC** so clients can stream requests without corrupting frames. **Logs** go to **stderr**.

- Successful inference responses are **`CROk` frames** whose payload is **raw `f32` little-endian** bytes (flattened outputs), not UTF-8 text.
- Errors use **`CREr`** frames with a **UTF-8 message** for humans/SDKs.

**Integration pattern:** spawn the process, parse frames on stdout, decode **f32** from the bytes, then reshape using `output_shapes` from manifest (or your own metadata).  
Details: see `coelanox-ipc` (`serve.rs` / `IPC_PROTOCOL_VERSION`) and [non-technical/DATA_FLOW.md](non-technical/DATA_FLOW.md) §5.

---

## 4. `coelanox-run` (minimal binary) — debug-style stdout

`coelanox-run` prints results with Rust’s **`Debug`** formatting (e.g. `Output: [0.1, 0.2, …]`), not a machine-oriented JSON contract. For stable, parseable **text** output, prefer **`coelanox run -o out.json`** or a small wrapper that calls the **`coelanox-standalone`** Rust API and serializes JSON yourself.

---

## 5. `coelanox benchmark` — text or JSON

`--output text` (default) prints **human-readable tables** to stdout.  
`--output json` prints **pretty JSON** to stdout for tooling. This describes **benchmark statistics**, not the model’s inference tensor unless the command is wired to include it in that JSON (see CLI help for the current fields).

---

## 6. `coelanox debug` — structured JSON file

`coelanox debug` writes a **JSON document** to the path given by `--output` (default `debug_output.json`). On success it includes **`output`** as a JSON array of floats plus container metadata; on failure it records **`status`** and error info. **stdout** still carries human-readable lines for the session.

---

## 7. From floats to “real text” (classification, NLP, etc.)

**Pattern**

1. Run inference → obtain `Vec<f32>` (from JSON file, IPC decode, or API).
2. **Reshape** using manifest `output_shapes` (e.g. `[1, 1000]` for logits).
3. Apply your **post-processing**:
   - **Classification:** `argmax` → index → label from your own table.
   - **NLP:** map token IDs or logits through your tokenizer/vocab (not provided by COELANOX).
   - **CV:** interpret raw outputs per your head (boxes, masks, etc.).

COELANOX’s role is **deterministic numeric inference** on the artifact you packaged; **presentation layer** stays in your stack.

---

## Related documents

| Document | Relevance |
|----------|-----------|
| [QUICKSTART.md](QUICKSTART.md) | `run` with `-i` / `-o` JSON |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | All flags per subcommand |
| [non-technical/DATA_FLOW.md](non-technical/DATA_FLOW.md) | Where data lives (run vs serve) |
| [OPERATIONS.md](OPERATIONS.md) | Env, config, deployment |

← [Documentation hub](README.md)
