#!/usr/bin/env bash
# Release gate: build + smoke tests before tagging a release.
# Run from repo root: bash scripts/release_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/coelanox"

echo "== cargo test (workspace) =="
cargo test --workspace

echo "== cargo build release CLI + standalone =="
cargo build --release -p coelanox-cli -p coelanox-standalone

echo "== release gate OK =="
echo "Next: git tag vX.Y.Z and push tags, or run GitHub Actions release workflow."
