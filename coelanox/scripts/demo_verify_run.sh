#!/usr/bin/env bash
# Demo path: verify → run only (no --no-verify). Use for investor demo.
# Prerequisites: resnet_tiny_mnist.bundle and resnet_tiny_mnist_simd.clfc in COELANOX_CLF_PATH (or cnox root).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COELANOX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CNOX_ROOT="$(cd "$COELANOX_DIR/.." && pwd)"
CLI="cargo run --release -p coelanox-cli --"
OUT_DIR="${CNOX_ROOT}/models/output"
mkdir -p "$OUT_DIR"

# 1. Package MNIST with CLF (set CLF path so packager finds .clfc)
export COELANOX_CLF_PATH="${CNOX_ROOT}"
$CLI package --input "${CNOX_ROOT}/resnet_tiny_mnist.bundle" \
  --output "${OUT_DIR}/mnist_clf.cnox" \
  --model-type cv --target cpu --use-demo-translator

# 2. Verify (integrity check)
$CLI verify --file "${OUT_DIR}/mnist_clf.cnox"

# 3. Run (always verifies again before load)
$CLI run --file "${OUT_DIR}/mnist_clf.cnox" --output "${OUT_DIR}/mnist_clf_out.json"

echo "Demo complete: verify → run (Execution time printed above)."
