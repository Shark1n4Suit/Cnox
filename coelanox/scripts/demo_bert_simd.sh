#!/usr/bin/env bash
# Package BERT with simd.clfc and run inference.
# Prerequisites: bert_base_uncased.bundle (BRT1 format) and simd.clfc in COELANOX_CLF_PATH (cnox root).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COELANOX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CNOX_ROOT="$(cd "$COELANOX_DIR/.." && pwd)"
CLI="cargo run --release -p coelanox-cli --"
OUT_DIR="${CNOX_ROOT}/models/output"
BERT_BUNDLE="${BERT_BUNDLE:-${CNOX_ROOT}/bert_base_uncased.bundle}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$BERT_BUNDLE" ]]; then
  echo "Error: BERT bundle not found: $BERT_BUNDLE"
  echo "Set BERT_BUNDLE to your BRT1 bundle path, or place bert_base_uncased.bundle in $CNOX_ROOT"
  exit 1
fi
if [[ ! -f "${CNOX_ROOT}/simd.clfc" ]]; then
  echo "Error: simd.clfc not found in $CNOX_ROOT"
  exit 1
fi

# 1. Package BERT with SIMD CLF (packager discovers simd.clfc and embeds plan + blobs)
export COELANOX_CLF_PATH="${CNOX_ROOT}"
echo "Packaging BERT with simd.clfc..."
$CLI package --input "$BERT_BUNDLE" \
  --output "${OUT_DIR}/bert_base_uncased.cnox" \
  --model-type nlp --target cpu --use-demo-translator

# 2. Verify
echo "Verifying container..."
$CLI verify --file "${OUT_DIR}/bert_base_uncased.cnox"

# 3. Run inference (backend 'simd' uses the SIMD plan in the container)
echo "Running inference (backend simd)..."
export COELANOX_CLF_PATH="${CNOX_ROOT}"
$CLI run --file "${OUT_DIR}/bert_base_uncased.cnox" \
  --output "${OUT_DIR}/bert_simd_out.json" \
  --backend simd

echo "Done. Output: ${OUT_DIR}/bert_simd_out.json"
