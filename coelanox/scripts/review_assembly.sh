#!/bin/bash
# Extract and disassemble machine code from COELANOX container

CONTAINER_FILE="${1:-models/output/bert_model.cnox}"
OUTPUT_FILE="/tmp/coelanox_machine_code.bin"
DISASM_FILE="/tmp/coelanox_disassembly.asm"

echo "=== Extracting machine code from $CONTAINER_FILE ==="

# Create a Rust script to extract the machine code
cat > /tmp/extract_code.rs << 'EOF'
use std::fs;
use coelanox_core::format::CoelanoxContainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let container_file = &args[1];
    let output_file = &args[2];
    
    let container = CoelanoxContainer::read_binary_file(std::path::Path::new(container_file))?;
    
    fs::write(output_file, &container.optimized_code)?;
    println!("Extracted {} bytes of machine code", container.optimized_code.len());
    Ok(())
}
EOF

# Compile and run extraction
cd /home/shark1n4uit/Saic/coelanox && cargo run --bin coelanox --quiet 2>&1 | grep -q "Finished" && \
rustc --edition 2021 -L target/debug/deps /tmp/extract_code.rs -o /tmp/extract_code 2>/dev/null && \
/tmp/extract_code "$CONTAINER_FILE" "$OUTPUT_FILE" || {
    echo "Using Python fallback to extract machine code..."
    python3 << PYEOF
import struct
import sys

container_file = "$CONTAINER_FILE"
output_file = "$OUTPUT_FILE"

# Read container
with open(container_file, 'rb') as f:
    # Read header
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    flags = struct.unpack('<I', f.read(4))[0]
    container_size = struct.unpack('<Q', f.read(8))[0]
    manifest_offset = struct.unpack('<Q', f.read(8))[0]
    code_offset = struct.unpack('<Q', f.read(8))[0]
    
    # Read manifest length
    f.seek(manifest_offset)
    manifest_len = struct.unpack('<I', f.read(4))[0]
    manifest_json = f.read(manifest_len)
    
    # Read machine code
    f.seek(code_offset)
    # Read code length prefix (if any) or read until weights
    # For now, try to read reasonable amount
    import json
    manifest = json.loads(manifest_json)
    code_size = manifest.get('model_size', 0)
    
    if code_size > 0:
        machine_code = f.read(code_size)
        with open(output_file, 'wb') as out:
            out.write(machine_code)
        print(f"Extracted {len(machine_code)} bytes of machine code")
    else:
        print("Could not determine code size from manifest")
        sys.exit(1)
PYEOF
}

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "=== Disassembling machine code ==="
    
    # Try objdump first
    if command -v objdump &> /dev/null; then
        objdump -d -M intel --start-address=0 "$OUTPUT_FILE" > "$DISASM_FILE" 2>&1 || \
        objdump -D -M intel --start-address=0 "$OUTPUT_FILE" > "$DISASM_FILE" 2>&1
        echo "Disassembly saved to: $DISASM_FILE"
        echo ""
        echo "=== First 200 lines of disassembly ==="
        head -200 "$DISASM_FILE"
    else
        echo "objdump not found. Showing hex dump instead:"
        echo ""
        hexdump -C "$OUTPUT_FILE" | head -50
    fi
else
    echo "ERROR: Could not extract machine code"
    exit 1
fi

