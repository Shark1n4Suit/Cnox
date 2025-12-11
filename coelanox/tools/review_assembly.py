#!/usr/bin/env python3
"""Extract and review machine code assembly from COELANOX container"""

import struct
import json
import sys

def extract_machine_code(container_file):
    """Extract machine code from COELANOX container"""
    with open(container_file, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'COEL':
            print(f"ERROR: Invalid magic: {magic}, expected COEL")
            return None
            
        version = struct.unpack('<I', f.read(4))[0]
        flags = struct.unpack('<I', f.read(4))[0]
        container_size = struct.unpack('<Q', f.read(8))[0]
        manifest_offset = struct.unpack('<Q', f.read(8))[0]
        code_offset = struct.unpack('<Q', f.read(8))[0]
        
        print(f"=== Container Header ===")
        print(f"Magic: {magic}")
        print(f"Version: {version:#x}")
        print(f"Manifest offset: {manifest_offset}")
        print(f"Code offset: {code_offset}")
        print()
        
        # Read manifest
        f.seek(manifest_offset)
        manifest_len = struct.unpack('<I', f.read(4))[0]
        manifest_json = f.read(manifest_len)
        try:
            manifest = json.loads(manifest_json.decode('utf-8'))
        except:
            # Try to find JSON in the data
            manifest_str = manifest_json.decode('utf-8', errors='ignore')
            # Find JSON object start
            json_start = manifest_str.find('{')
            if json_start >= 0:
                manifest_json_clean = manifest_str[json_start:]
                json_end = manifest_json_clean.rfind('}')
                if json_end >= 0:
                    manifest = json.loads(manifest_json_clean[:json_end+1])
                else:
                    raise
            else:
                raise
        
        print(f"=== Manifest ===")
        print(f"Model: {manifest.get('name', 'unknown')}")
        print(f"Code size (from manifest): {manifest.get('model_size', 0)} bytes")
        print()
        
        # Read machine code
        f.seek(code_offset)
        code_size = manifest.get('model_size', 0)
        if code_size == 0:
            print("ERROR: Code size is 0")
            return None
            
        machine_code = f.read(code_size)
        
        print(f"=== Extracted Machine Code ===")
        print(f"Size: {len(machine_code)} bytes")
        print()
        
        return machine_code

def disassemble_x86_64(code, max_bytes=200):
    """Simple x86-64 disassembler for common instructions"""
    instructions = {
        (0x55,): "push rbp",
        (0x48, 0x89, 0xE5): "mov rbp, rsp",
        (0x49, 0x89, 0xFA): "mov r10, rdi",
        (0x53,): "push rbx",
        (0x41, 0x54): "push r12",
        (0x41, 0x56): "push r14",
        (0x41, 0x57): "push r15",
        (0x4D, 0x89, 0xD4): "mov r12, r10",
        (0x49, 0x89, 0xF6): "mov r14, rsi",
        (0x49, 0x89, 0xD7): "mov r15, rdx",
        (0x48, 0x83, 0xEC, 0x20): "sub rsp, 32",
        (0x4C, 0x89, 0xE0): "mov rax, r12",
        (0x4C, 0x8B, 0x50, 0x08): "mov r10, [rax+8]",
        (0x4D, 0x89, 0x14, 0x24): "mov [rsp], r10",
        (0x48, 0x89, 0x4C, 0x24, 0x10): "mov [rsp+16], rcx",
        (0x4C, 0x89, 0x44, 0x24, 0x18): "mov [rsp+24], r8",
        (0x48, 0x8B, 0x04, 0x24): "mov rax, [rsp]",
        (0x48, 0x8D, 0x04, 0x98): "lea rax, [rax+r11*4]",
        (0x4C, 0x89, 0xF1): "mov rcx, r14",
        (0x48, 0x8D, 0x0C, 0x99): "lea rcx, [rcx+r11*4]",
        (0x48, 0x85, 0xDB): "test rbx, rbx",
        (0x0F, 0x84): "jz rel32",
        (0x48, 0x83, 0xC4, 0x20): "add rsp, 32",
        (0x41, 0x5F): "pop r15",
        (0x41, 0x5E): "pop r14",
        (0x41, 0x5C): "pop r12",
        (0x5B,): "pop rbx",
        (0x5D,): "pop rbp",
        (0xC3,): "ret",
    }
    
    i = 0
    output = []
    while i < min(len(code), max_bytes):
        matched = False
        for pattern, mnemonic in instructions.items():
            if i + len(pattern) <= len(code):
                if code[i:i+len(pattern)] == bytes(pattern):
                    hex_str = ' '.join(f'{b:02x}' for b in pattern)
                    output.append(f"{i:06x}: {hex_str:20s} {mnemonic}")
                    i += len(pattern)
                    matched = True
                    break
        
        if not matched:
            # Unknown instruction, show hex
            output.append(f"{i:06x}: {code[i]:02x}                   (unknown)")
            i += 1
    
    return output

def main():
    container_file = sys.argv[1] if len(sys.argv) > 1 else "models/output/bert_model.cnox"
    
    machine_code = extract_machine_code(container_file)
    if not machine_code:
        return 1
    
    print("=== Disassembly (First 300 bytes) ===")
    disasm = disassemble_x86_64(machine_code, max_bytes=300)
    for line in disasm:
        print(line)
    
    print()
    print("=== Hex Dump (First 100 bytes) ===")
    for i in range(0, min(100, len(machine_code)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in machine_code[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in machine_code[i:i+16])
        print(f"{i:06x}: {hex_part:<48s} {ascii_part}")
    
    # Save to file for objdump
    with open('/tmp/coelanox_machine_code.bin', 'wb') as f:
        f.write(machine_code)
    print()
    print("Machine code saved to: /tmp/coelanox_machine_code.bin")
    print("To disassemble with objdump: objdump -D -M intel /tmp/coelanox_machine_code.bin")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

