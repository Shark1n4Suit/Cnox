# COELANOX — Customer onboarding (model → `.cnox` → run)

**Goal:** Get from a **trained model export** to a **running `.cnox`** in **your** environment **without** vendor SSH access. Adjust paths and OS for your site.

---

## Prerequisites

- A **supported** source artifact (see [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)).  
- A **x86_64** (or supported) host for the **prebuilt CLI**, or **Rust stable** to build from source.  
- Enough **disk and RAM** for packaging and inference (model-dependent).

---

## Step 1 — Install the CLI

**Option A — Binary release (no compiler):**  
Download the release archive for your OS from your vendor or project releases, extract, add `coelanox` to **`PATH`**.

**Option B — Build from source:**  
From the repository: `cd coelanox && cargo build --release -p coelanox-cli`. Copy the `coelanox` binary to a directory on `PATH`.

**Verify:** `coelanox --version`

---

## Step 2 — (Optional) Prepare signing keys

If you use **Ed25519** signing:

```bash
coelanox keygen -o ./keys
```

Store **`keys/secret.bin`** in a **secrets manager**; distribute **`keys/public.bin`** only where verification should occur.

---

## Step 3 — Package the model

**ONNX example:**

```bash
coelanox package -i /path/to/model.onnx -o /path/to/model.cnox --target cpu
```

**If packaging fails with `Custom` ops:** See [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md) and fix the graph or translator.

**Scalar-only / portable container:**

```bash
coelanox package -i model.onnx -o model.cnox --target cpu --fallback-only
```

Demo bundles (ResNet tiny, etc.) use **`--use-demo-translator`** and format flags per [Quickstart](../QUICKSTART.md).

---

## Step 4 — Verify before you ship

```bash
coelanox verify -f /path/to/model.cnox
```

**Expected:** success message. **If hash fails:** do not deploy; re-copy or re-build.

**With signing:**

```bash
coelanox verify -f model.cnox --trusted-key keys/public.bin
```

---

## Step 5 — Inspect the container

```bash
coelanox info -f model.cnox
```

Confirm **input/output shapes**, sizes, and flags (e.g. audit required).

---

## Step 6 — Run inference locally

Prepare **input JSON** matching manifest shapes (flattened **f32**, row-major order as documented for your build).

```bash
coelanox run -f model.cnox -i input.json -o output.json
```

**Synthetic input (testing only):**

```bash
coelanox run -f model.cnox -o output.json
```

---

## Step 7 — Production configuration

Do **not** rely on defaults in production.

1. Create a **JSON config** or set **`COELANOX_*`** environment variables (see [Operations](../OPERATIONS.md)): limits for container size, input size, memory, timeout, **`COELANOX_ALLOW_ABSOLUTE_PATHS=false`** where policy requires.  
2. Point **`COELANOX_CONFIG_FILE`** at your config file on the runtime host.  
3. Set **`RUST_LOG`** / **`COELANOX_LOG_LEVEL`** so security-relevant events reach your log stack.

---

## Step 8 — Airgapped transfer

1. Run **verify** on the build side.  
2. Copy **`.cnox`** (and **public key** if signing) via approved transfer.  
3. Run **verify** again on the isolated host before **run**.

See [DATA_FLOW.md](DATA_FLOW.md).

---

## Step 9 — Long-lived serving (optional)

For process-to-process integration:

```bash
coelanox serve -f model.cnox -b scalar
```

(Or another backend name your build supports.) **IPC** is **stdin/stdout** framed binary; see [CLI reference](../CLI_REFERENCE.md) § serve. Wrap with your **supervisor** and **health** logic—there is no built-in HTTP server.

---

## Step 10 — Get help without SSH

| Problem | Self-serve resource |
|---------|---------------------|
| Command flags | [CLI_REFERENCE.md](../CLI_REFERENCE.md) |
| Ops not packaging | [ONNX_SUPPORTED_OPS.md](ONNX_SUPPORTED_OPS.md), [troubleshooting](../sdk/troubleshooting.md) |
| Slow inference | [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), CLF path in [Operations](../OPERATIONS.md) |
| Integrity / policy | [RUNTIME_SPECIFICATION.md](RUNTIME_SPECIFICATION.md), [Operations](../OPERATIONS.md) |

---

## Related documents

- [Quickstart](../QUICKSTART.md) — Shortest path (5 minutes)  
- [PRODUCT_BRIEF.md](PRODUCT_BRIEF.md) — What you are buying  
- [DATA_FLOW.md](DATA_FLOW.md) — Where data moves  

← [Non-technical hub](README.md)
