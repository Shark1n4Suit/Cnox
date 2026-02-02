# COELANOX Security Documentation

## Overview

COELANOX implements a comprehensive security model with defense-in-depth principles. This document describes the security features, guarantees, and best practices for using COELANOX containers.

## Security Architecture

### Defense-in-Depth Model

COELANOX uses a three-layer defense model:

```
┌─────────────────────────────────────────────────────────┐
│              Defense-in-Depth Security Model                │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Package-Time Security                          │
│  ├── SHA-256 hash calculation                            │
│  ├── Container integrity verification                    │
│  └── Secure container creation                            │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Pre-Deployment Verification                     │
│  ├── Standalone verify command                           │
│  ├── Hash verification before deployment                 │
│  └── Container validation                                │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Runtime Security                                │
│  ├── Automatic hash verification on load                 │
│  ├── Secure container loading                            │
│  └── Safe execution environment                          │
└─────────────────────────────────────────────────────────┘
```

## Security Hash System

### SHA-256 Integrity Verification

**Purpose**: Detect any modification to container content.

**Implementation**:
- **Algorithm**: SHA-256 (256-bit hash)
- **Scope**: Entire container content (header, manifest, code, weights, IR)
- **Storage**: 32-byte hash stored in container header

### Hash Calculation Process

1. **During Packaging**:
   ```
   Container Creation → Hash Calculation → Hash Storage → File Write
   ```

2. **Hash Calculation Steps**:
   - Container is built in memory
   - `security_hash` field is set to `[0; 32]`
   - Entire container is serialized to buffer
   - SHA-256 is calculated over buffer
   - Hash is stored in header
   - Complete container is written to disk

3. **Hash Verification Steps**:
   - Container is read from disk
   - `security_hash` field is temporarily zeroed
   - SHA-256 is recalculated
   - Calculated hash is compared with stored hash
   - Verification succeeds if hashes match

**Location**: `coelanox-core/src/format.rs`

**Key Functions**:
- `CoelanoxContainer::write_binary_file()` - Calculates hash during packaging
- `verify_security_hash()` - Verifies container integrity

### Security Guarantees

- **Integrity**: Any modification to container content is detected
- **Tamper Detection**: Changes to code, weights, metadata, or manifest invalidate hash
- **Completeness**: Hash covers entire container (no partial verification)

### Hash Verification Commands

#### Standalone Verification

```bash
coelanox verify container.cnox
```

**Use Cases**:
- Pre-deployment verification
- CI/CD pipeline validation
- Container integrity checks

#### Automatic Verification

```bash
coelanox run container.cnox --input data.bin
```

**Behavior**:
- Automatically verifies hash before loading
- Fails if hash mismatch detected
- Provides defense-in-depth (verification at multiple stages)

## Three-Command Security Model

### 1. Package Command Security

**Command**: `coelanox package`

**Security Features**:
- Secure container creation
- Hash calculation during packaging
- Input validation
- Path sanitization

**Security Checks**:
- Input file validation
- Output path validation
- Model format verification
- Weight compression integrity

### 2. Verify Command Security

**Command**: `coelanox verify`

**Security Features**:
- Standalone verification (no execution)
- Fast integrity check
- Detailed error reporting

**Security Checks**:
- File existence validation
- Container format validation
- Hash verification
- Manifest parsing validation

### 3. Run Command Security

**Command**: `coelanox run`

**Security Features**:
- Pre-execution verification
- Secure container loading
- Safe execution environment

**Security Checks**:
- Hash verification (defense in depth)
- Container format validation
- Input data validation
- Execution environment isolation

## Container Security

### File System Security

**Location**: `coelanox-runtime/src/security.rs`

**Protections**:
- **Path Traversal**: Blocks `..` patterns in paths
- **Symlink Attacks**: Prevents symlink access
- **File Permissions**: Validates file permissions
- **File Type**: Ensures regular files only (no devices/sockets)
- **Size Limits**: Enforces 1GB file limit, 1MB manifest limit

### Memory Safety

**Rust Ownership System**:
- Prevents buffer overflows
- Prevents use-after-free
- Prevents data races
- Zero-cost abstractions

**Safe Unsafe Blocks**:
- Executable memory allocation (mmap)
- Machine code execution
- FFI calls to backend libraries

### Format Validation

**Magic Number Check**:
- Validates container starts with "COEL"
- Prevents processing non-container files

**Version Validation**:
- Validates container version compatibility
- Prevents version mismatch issues

**Structure Verification**:
- Validates header structure
- Validates manifest JSON
- Validates data lengths

## Backend Security

### Backend Library Loading

**Security Considerations**:
- **Library Validation**: Validates backend library format
- **Symbol Verification**: Verifies required symbols exist
- **Version Checking**: Validates backend interface version
- **Path Validation**: Validates library paths (prevents path traversal)

**Current Implementation**:
- Backend loading is placeholder (not yet implemented)
- Future implementation will include:
  - Library signature verification
  - Sandboxed execution
  - Capability restrictions

### Backend Discovery Security

**Search Paths**:
- Default paths are system directories
- User paths require proper permissions
- Path validation prevents traversal attacks

**Discovery Process**:
- Scans only specified directories
- Validates file types (dynamic libraries only)
- Tracks failures for security auditing

## Security Best Practices

### Container Distribution

1. **Verify Before Distribution**:
   ```bash
   coelanox verify container.cnox
   ```

2. **Use Secure Channels**:
   - HTTPS for downloads
   - Signed distributions
   - Checksum verification

3. **Store Hashes Separately**:
   - Store hash in separate file
   - Verify hash matches before use
   - Use hash for integrity checks

### Container Deployment

1. **Pre-Deployment Verification**:
   ```bash
   coelanox verify container.cnox
   ```

2. **Runtime Verification**:
   - Run command automatically verifies
   - No need for manual verification
   - Defense in depth

3. **Monitor Verification Failures**:
   - Log verification failures
   - Alert on hash mismatches
   - Investigate tampering attempts

### Development Security

1. **Secure Packaging**:
   - Package in secure environment
   - Verify containers after packaging
   - Test hash verification

2. **Backend Development**:
   - Validate backend libraries
   - Test backend loading
   - Verify backend interfaces

3. **Testing**:
   - Test tamper detection
   - Test error handling
   - Test security edge cases

## Security Guarantees

### Integrity Guarantees

- **Container Integrity**: SHA-256 hash detects any modification
- **Code Integrity**: Machine code modifications are detected
- **Weight Integrity**: Weight modifications are detected
- **Metadata Integrity**: Manifest modifications are detected

### Availability Guarantees

- **100% Uptime**: Scalar fallback ensures execution always possible
- **Graceful Degradation**: Falls back to scalar if vendor backends fail
- **Error Recovery**: Handles errors gracefully without crashes

### Confidentiality

- **No Data Leakage**: Container format doesn't expose sensitive data
- **Secure Storage**: Weights are compressed but not encrypted (future: encryption support)
- **Execution Isolation**: Runtime provides execution isolation

## Security Limitations

### Current Limitations

1. **No Encryption**: Weights are compressed but not encrypted
   - **Future**: Encryption support planned

2. **No Code Signing**: Backend libraries are not signed
   - **Future**: Code signing support planned

3. **No Sandboxing**: Backend execution not sandboxed
   - **Future**: Sandboxing support planned

### Known Risks

1. **Backend Library Risk**: Vendor backends may have vulnerabilities
   - **Mitigation**: Validate backends, use fallback if needed

2. **Path Traversal**: If paths not properly validated
   - **Mitigation**: Path validation in security module

3. **Hash Collision**: Theoretically possible but extremely unlikely
   - **Mitigation**: SHA-256 provides 2^256 security level

## Security Testing

### Test Coverage

**Location**: `coelanox/tests/test_security_hash.rs`

**Tests**:
- Hash calculation during packaging
- Hash verification for valid containers
- Tamper detection (modified containers)
- Corrupted container detection
- Error handling for invalid files

### Security Validation

1. **Tamper Detection**: Tests verify tampered containers are detected
2. **Error Handling**: Tests verify errors are handled securely
3. **Edge Cases**: Tests verify security edge cases

## Security Updates

### Update Process

1. **Security Patches**: Applied to core library
2. **Hash Verification**: Updated containers require new hash
3. **Backward Compatibility**: Old containers remain verifiable

### Vulnerability Reporting

- Report security issues through secure channels
- Security patches released promptly
- CVE tracking for vulnerabilities

## Compliance

### Security Standards

- **Memory Safety**: Rust ownership system
- **Input Validation**: Comprehensive input validation
- **Error Handling**: Secure error handling
- **Logging**: Security event logging

### Audit Trail

- Verification events logged
- Error events logged
- Security failures tracked

## References

- [Architecture Documentation](ARCHITECTURE.md)
- [Production MVP Roadmap](../roadmaps/PRODUCTION_MVP_ROADMAP.md)
- [COELANOX Reference](../COELANOX_REFERENCE.md)

