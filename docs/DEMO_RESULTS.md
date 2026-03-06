# COELANOX Demo Results

This document records benchmark results and reproduction steps for the full CLF execution path (verify → run).

---

## Demo workflow

- The `run` command always verifies container integrity before load; there is no bypass in the documented flow.
- Standard flow: `package` → `verify` → `run` (or `benchmark`).

---

## Full CLF MNIST (ResNet-tiny, batch 1)

| Item | Value |
|------|--------|
| **Container** | `mnist_clf.cnox` (packaged with `.clfc`, without `--fallback-only`) |
| **Backend** | `plan` (CLF execution plan; SIMD kernel dispatch) |
| **Cold run (single)** | 55,853 μs (~56 ms) |
| **Average (20 iterations, after warm-up)** | 61,374 μs (~61 ms) |
| **Scalar (same model, fallback-only package)** | ~57–66 ms (comparison baseline) |

### Scalar vs full CLF (this run)

| Path | Backend | Latency (μs) |
|------|---------|--------------|
| Scalar | legacy | ~57–66k |
| Full CLF | plan | ~56–61k |

Full CLF matches or improves on scalar for this model on the measured hardware.

---

## Reproduction

**Prerequisites:** Default CLF path `~/.coelanox/clf/clfc/` populated with the appropriate `.clfc` (no environment variable required). Alternatively, set `COELANOX_CLF_PATH` to the directory containing your `.clfc`.

```bash
cd coelanox
./scripts/demo_verify_run.sh
```

Or run verify and run explicitly:

```bash
coelanox verify --file models/output/mnist_clf.cnox
coelanox run --file models/output/mnist_clf.cnox --output out.json
```
