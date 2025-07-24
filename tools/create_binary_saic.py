#!/usr/bin/env python3
"""
Create SAIC files with binary manifest format for instant performance testing
"""

import struct
import sys

def create_binary_saic_file(model_name, size_mb, filename):
    """Create a SAIC file with binary manifest format"""
    
    # Create binary manifest (64 bytes total)
    # Model name (32 bytes) + reserved (32 bytes)
    manifest = bytearray(64)
    
    # Copy model name (truncate if too long)
    name_bytes = model_name.encode('utf-8')
    copy_len = min(len(name_bytes), 32)
    manifest[:copy_len] = name_bytes[:copy_len]
    
    # Create model payload
    model_payload = b'X' * (size_mb * 1024 * 1024)
    
    with open(filename, 'wb') as f:
        # Header: magic + version
        f.write(b'SAIC')
        f.write(struct.pack('>4B', 0, 1, 0, 0))
        
        # Manifest length (64 bytes for binary manifest)
        f.write(struct.pack('>I', 64))
        
        # Binary manifest
        f.write(manifest)
        
        # Model payload
        f.write(model_payload)
    
    print(f"✅ Created {filename}: {len(manifest)} bytes binary manifest + {len(model_payload)} bytes payload")
    print(f"   Model name: '{model_name}'")
    print(f"   Total size: {len(manifest) + len(model_payload) + 12} bytes")

if __name__ == "__main__":
    # Create test files of different sizes
    test_cases = [
        (1, "test_1mb_binary.saic"),
        (10, "test_10mb_binary.saic"),
        (100, "test_100mb_binary.saic"),
        (500, "test_500mb_binary.saic"),
        (1000, "test_1000mb_binary.saic"),
    ]
    
    for size_mb, filename in test_cases:
        model_name = f"TestModel-{size_mb}MB"
        create_binary_saic_file(model_name, size_mb, filename)
        print() 