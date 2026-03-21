# COELANOX Architecture Documentation

## Overview

COELANOX is a universal binary container system for AI models that provides secure, fast, and hardware-optimized model deployment. This document describes the architecture of the COELANOX system, focusing on Phase 1 features: security hash, backend manager, and AOT selection.

## Three-Command Architecture

COELANOX uses a three-command architecture for defense-in-depth security:

### 1. Package Command (`coelanox package`)

**Purpose**: Create a COELANOX container from a source model.

**Process**:
1. Load source model (ONNX, TensorFlow, PyTorch, etc.)
2. Translate to Universal IR
3. Optimize IR for target hardware
4. Generate machine code for target hardware
5. Compress weights
6. Calculate SHA-256 security hash
7. Write container to `.cnox` file

**Key Features**:
- AOT (Ahead-of-Time) compilation
- Hardware-specific code generation
- Weight compression (Zstd/LZ4)
- Security hash calculation

**Location**: `coelanox-packager/src/lib.rs`

### 2. Verify Command (`coelanox verify`)

**Purpose**: Standalone verification of container integrity.

**Process**:
1. Read container file
2. Calculate SHA-256 hash of container content
3. Compare with stored hash in header
4. Report verification result

**Key Features**:
- Pre-deployment verification
- Standalone operation (no execution)
- Fast verification (SHA-256 calculation)

**Location**: `coelanox-core/src/format.rs` (`verify_security_hash`)

### 3. Run Command (`coelanox run`)

**Purpose**: Verify, load, and execute a container.

**Process**:
1. Verify security hash (defense in depth)
2. Load container into runtime
3. Execute model with input data
4. Return output

**Key Features**:
- Automatic verification before execution
- Container loading and caching
- Execution with fallback support

**Location**: `coelanox-cli/src/main.rs` (`run_command`)

## Security Hash System

### SHA-256 Hash Calculation

The security hash is calculated during packaging:

1. **Container Creation**: Container is built in memory with all components:
   - Header (with `security_hash` initially set to `[0; 32]`)
   - Manifest
   - Machine code
   - Compressed weights
   - IR data

2. **Hash Calculation**: 
   - Entire container is written to an in-memory buffer
   - SHA-256 hash is calculated over the buffer (with hash field zeroed)
   - Hash is stored in the header's `security_hash` field
   - Complete container (with hash) is written to disk

3. **Hash Verification**:
   - Container is read from disk
   - Hash field is temporarily zeroed
   - SHA-256 is recalculated
   - Calculated hash is compared with stored hash

**Location**: `coelanox-core/src/format.rs`

**Key Functions**:
- `CoelanoxContainer::write_binary_file()` - Calculates and stores hash
- `verify_security_hash()` - Verifies container integrity

### Security Guarantees

- **Integrity**: Any modification to container content is detected
- **Tamper Detection**: Changes to code, weights, or metadata invalidate hash
- **Defense in Depth**: Verification happens at package time, verify command, and run command

## Backend Manager Architecture

### Overview

The Backend Manager discovers and manages CLF (Coelanox Library File) backends. It operates at **package time** to discover .clfc files for AOT compilation.

### Components

#### BackendManager

**Location**: `coelanox-orchestrator/src/backend_manager.rs`

**Responsibilities**:
- Discover CLF backends (.clfc files) from search paths
- Track discovered backends
- Provide backend selection strategies

**Key Methods**:
- `new()` - Create new BackendManager
- `discover_backends()` - Scan search paths for CLF files
- `validate_backend()` - Check if backend exists
- `select_backend_with_fallback()` - Select backend using fallback strategy

#### BackendInfo

**Structure**:
```rust
pub struct BackendInfo {
    pub name: String,                    // Backend name (e.g., "nvidia-h100")
    pub version: String,                 // Backend version
    pub library_path: PathBuf,          // Path to CLF file (.clfc)
    pub supported_targets: Vec<HardwareTarget>, // Supported hardware targets
}
```

#### DiscoveryResult

**Structure**:
```rust
pub struct DiscoveryResult {
    pub discovered: Vec<BackendInfo>,    // Successfully discovered backends
    pub failures: Vec<(PathBuf, String)>, // Failed paths with error messages
    pub total_scanned: usize,            // Total files scanned
}
```

### Backend Discovery Process

1. **Search Paths**: BackendManager scans default search paths:
   - `~/.coelanox/clf/` with subdirs `clfc/`, `clfmm/`, `clfmp/`, `clfe/`
   - `COELANOX_CLF_PATH` (override)
   - `COELANOX_BACKEND_PATH` (extra paths, colon-separated)

2. **CLF Detection**: Identifies CLF files (`.clfc`, legacy `.clf`)

3. **CLF Loading**: Reads CLF header to extract backend metadata and supported targets

4. **Registration**: Registers valid backends for use during packaging

### Backend Selection Strategies

**Location**: `coelanox-orchestrator/src/backend_manager.rs`

**Strategies**:
- `PrimaryOnly` - Use primary backend only (fail if unavailable)
- `PrimaryWithFallback` - Use primary with fallback to scalar
- `TryAll` - Try all backends in order until one succeeds

## AOT Binary Selection

### Overview

AOT (Ahead-of-Time) selection allows choosing which backends to compile into the container at package time. This enables:
- **Slim containers**: Single backend (smaller size)
- **Fat containers**: Multiple backends (larger size, more flexibility)
- **Auto selection**: Automatic backend detection

### Selection Modes

#### Slim Mode (Default)

**Usage**: `coelanox package --target cpu`

**Behavior**:
- Selects single primary backend
- Optionally adds additional backends with `--backend-add`
- Stores primary backend and additional backends in `BackendMetadata`

**Metadata Structure**:
```rust
BackendSelectionMode::Slim {
    primary: String,        // Primary backend name
    additional: Vec<String>, // Additional backends
}
```

#### Fat Mode

**Usage**: `coelanox package --fat`

**Behavior**:
- Includes all discovered backends
- Validates all backends exist before packaging
- Stores all backends in `BackendMetadata`

**Metadata Structure**:
```rust
BackendSelectionMode::Fat {
    backends: Vec<String>,  // All available backends
}
```

#### Auto Mode

**Usage**: `coelanox package --auto`

**Behavior**:
- Detects recommended hardware target from environment
- Selects backend for detected hardware
- Falls back to CPU if detected backend unavailable

**Metadata Structure**:
```rust
BackendSelectionMode::Auto {
    backends: Vec<String>,  // Auto-detected backends
}
```

### Backend Metadata

**Location**: `coelanox-core/src/format.rs`

**Structure**:
```rust
pub struct BackendMetadata {
    pub selection_mode: BackendSelectionMode,
    pub device_id_mapping: HashMap<usize, String>, // Device ID -> Backend name
}
```

**Device ID Mapping**:
- Primary backend: Device ID 0
- Additional backends: Device ID 1, 2, 3, ...
- Used for runtime backend selection

### Implementation

**Location**: `coelanox-packager/src/lib.rs`

**Key Function**: `determine_backend_selection()`
- Parses CLI flags (`--fat`, `--auto`, `--backend-add`)
- Validates backend existence
- Creates `BackendMetadata` structure
- Stores metadata in container manifest

## Container Format Structure

### Binary Layout

```
┌─────────────────────────────────────────────────────────┐
│                    COELANOX Container                      │
├─────────────────────────────────────────────────────────┤
│  Header (76 bytes)                                       │
│  ├── Magic Number: "COEL" (4 bytes)                     │
│  ├── Version: 0.1.0.0 (4 bytes)                         │
│  ├── Security Hash: SHA-256 (32 bytes)                  │
│  └── Reserved (36 bytes)                                │
├─────────────────────────────────────────────────────────┤
│  Manifest Length (4 bytes)                               │
├─────────────────────────────────────────────────────────┤
│  Manifest (JSON)                                         │
│  ├── Model metadata                                      │
│  ├── Input/output shapes                                 │
│  └── Backend metadata (AOT selection)                   │
├─────────────────────────────────────────────────────────┤
│  Machine Code Length (4 bytes)                           │
├─────────────────────────────────────────────────────────┤
│  Machine Code (binary)                                   │
├─────────────────────────────────────────────────────────┤
│  Compressed Weights Length (4 bytes)                     │
├─────────────────────────────────────────────────────────┤
│  Compressed Weights (binary, Zstd/LZ4)                  │
├─────────────────────────────────────────────────────────┤
│  IR Data Length (4 bytes)                                │
├─────────────────────────────────────────────────────────┤
│  IR Data (bincode serialized OptimizedIR)                │
└─────────────────────────────────────────────────────────┘
```

### Header Structure

**Location**: `coelanox-core/src/format.rs`

```rust
pub struct CoelanoxHeader {
    pub magic: [u8; 4],           // "COEL"
    pub version: [u8; 4],         // Version bytes
    pub security_hash: [u8; 32],  // SHA-256 hash
    pub reserved: [u8; 36],       // Reserved for future use
}
```

### Manifest Structure

**Location**: `coelanox-core/src/format.rs`

```rust
pub struct CoelanoxManifest {
    pub model_name: String,
    pub model_format: ModelFormat,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub machine_code_size: usize,
    pub backend_metadata: Option<BackendMetadata>, // AOT selection metadata
}
```

## Runtime Execution Flow

### Container Loading

1. **Verification**: Security hash is verified
2. **Parsing**: Header and manifest are parsed
3. **Caching**: Container is cached in runtime
4. **Preparation**: Machine code and weights are prepared

### Execution

1. **Backend Selection**: Select backend based on `BackendMetadata`
2. **Code Execution**: Execute machine code (if available)
3. **Fallback**: Fall back to scalar operations if machine code fails
4. **Result**: Return output data

### Scalar Fallback

**Location**: `coelanox-orchestrator/src/runtime.rs`

**Current Implementation**:
- Scalar fallback in `coelanox-orchestrator/src/fallback/` using `coelanox-core` scalar ops
- Pure Rust implementations
- 100% uptime guarantee
- Supports all OpType variants

## Component Dependencies

```
coelanox-core
├── format.rs          # Container format, security hash
├── ir.rs              # Universal IR structures
├── error.rs           # Error types
└── scalar/            # Scalar operations (reference implementation)

coelanox-packager
├── lib.rs             # Packaging logic, AOT selection
├── backend_trait/     # Backend translator trait, op mappings
├── optimizers/        # IR optimization
├── translators/       # Model translators (ONNX, BERT, ResNet)
└── memory/            # NVMe allocator, memory management

coelanox-orchestrator
├── runtime.rs         # Runtime execution, container loading
├── backend_manager.rs # Backend discovery and management
├── fallback/          # Scalar fallback (uses coelanox-core scalar)
└── execution_backend.rs # Execution dispatch

coelanox-cli
└── main.rs            # CLI commands (package, verify, run)
```

## Error Handling

### Error Types

**Location**: `coelanox-core/src/error.rs`

**Key Error Variants**:
- `SecurityHashMismatch` - Hash verification failed
- `BackendNotFound` - Requested backend not available
- `BackendLibraryNotFound` - CLF file not found
- `BackendDiscoveryFailed` - Backend discovery failed
- `DeviceIdOutOfRange` - Invalid device ID

### Error Propagation

- Errors are propagated using `CoelanoxResult<T>` type
- Detailed error messages for debugging
- Graceful degradation where possible (fallback to scalar)

## Performance Characteristics

### Security Hash

- **Calculation**: O(n) where n is container size
- **Verification**: O(n) where n is container size
- **Typical Time**: <1ms for typical containers

### Backend Discovery

- **Discovery**: O(m) where m is number of libraries scanned
- **Loading**: O(1) per library (lazy loading)
- **Typical Time**: <100ms for typical search paths

### AOT Selection

- **Validation**: O(b) where b is number of backends
- **Metadata Creation**: O(1)
- **Typical Time**: <1ms

## Future Enhancements

### Planned (Post-MVP)

1. **Dynamic Backend Loading** (Future)
   - Runtime backend loading (currently package-time only)
   - Hot-swappable backends
   - Backend health monitoring

## References

- [Security Documentation](SECURITY.md)
- [Production MVP Roadmap](../roadmaps/PRODUCTION_MVP_ROADMAP.md)
- [COELANOX Reference](../COELANOX_REFERENCE.md)

