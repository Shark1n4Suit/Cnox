#!/bin/bash
# Install coelanox to /usr/local/bin (or specified dir)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-/usr/local/bin}"
mkdir -p "$INSTALL_DIR"
cp -f "$SCRIPT_DIR/coelanox" "$INSTALL_DIR/coelanox"
chmod +x "$INSTALL_DIR/coelanox"
echo "Installed coelanox to $INSTALL_DIR"
echo "Run: coelanox --help"
