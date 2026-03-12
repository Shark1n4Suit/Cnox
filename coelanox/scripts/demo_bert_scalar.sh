#!/usr/bin/env bash
# Package BERT with scalar backend (fallback-only), sign it, and run inference.
# Prerequisites: bert_base_uncased.bundle (BRT1 format).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COELANOX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CNOX_ROOT="$(cd "$COELANOX_DIR/.." && pwd)"
CLI="cargo run --release -p coelanox-cli --"
OUT_DIR="${CNOX_ROOT}/models/output"
KEY_DIR="${OUT_DIR}/keys"
BERT_BUNDLE="${BERT_BUNDLE:-${CNOX_ROOT}/bert_base_uncased.bundle}"
mkdir -p "$OUT_DIR" "$KEY_DIR"

if [[ ! -f "$BERT_BUNDLE" ]]; then
  echo "Error: BERT bundle not found: $BERT_BUNDLE"
  echo "Set BERT_BUNDLE to your BRT1 bundle path, or place bert_base_uncased.bundle in $CNOX_ROOT"
  exit 1
fi

# 1. Generate signing keypair if not present
SECRET_KEY="${KEY_DIR}/secret.bin"
PUBLIC_KEY="${KEY_DIR}/public.bin"
if [[ ! -f "$SECRET_KEY" ]] || [[ ! -f "$PUBLIC_KEY" ]]; then
  echo "Generating Ed25519 keypair..."
  $CLI keygen --output-dir "$KEY_DIR"
fi

# 2. Package BERT with scalar backend (fallback-only) and sign
export COELANOX_CLF_PATH="${CNOX_ROOT}"
echo "Packaging BERT with scalar backend (fallback-only) and signing..."
$CLI package --input "$BERT_BUNDLE" \
  --output "${OUT_DIR}/bert_base_uncased_scalar.cnox" \
  --model-type nlp --target cpu --use-demo-translator \
  --fallback-only \
  --signing-key "$SECRET_KEY"

# 3. Verify (hash + signature)
echo "Verifying container (hash + signature)..."
$CLI verify --file "${OUT_DIR}/bert_base_uncased_scalar.cnox" \
  --trusted-key "$PUBLIC_KEY"

# 4. Run inference with scalar backend
echo "Running inference (backend scalar)..."
$CLI run --file "${OUT_DIR}/bert_base_uncased_scalar.cnox" \
  --output "${OUT_DIR}/bert_scalar_out.json" \
  --backend scalar \
  --trusted-key "$PUBLIC_KEY"

echo "Done. Output: ${OUT_DIR}/bert_scalar_out.json"
