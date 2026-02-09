# COELANOX Production MVP Roadmap

**Status:** Pre-Production → Production MVP  
**Target:** Production-ready deployment system for AI models  
**Timeline:** 12-16 weeks to MVP

## 📍 **Current Progress**

**Phase 1: Critical Fixes & Architecture Cleanup** ✅ **COMPLETED** (Weeks 1-3)
- ✅ **Week 1**: Security hash, backend cleanup, backend manager, AOT selection
- ✅ **Week 2**: Scalar backend refactoring (structure created, operations need implementation)
- ✅ **Week 3**: Testing & documentation complete

**Phase 2: Core Features** ⏳ **IN PROGRESS** (Weeks 4-8)
- ✅ **Week 4.1**: Max Shape Reader implemented
- ✅ **Week 4.2**: NVMe Pre-Allocation implemented (unified file approach)
- ✅ **Week 4.2.1**: Memory HAL & Protection HAL implemented in core (Memory HAL wired in packager + runtime; Protection HAL trait/registry in core, runtime integration pending)
- ⏳ **Week 4.3**: Static Load & Lock Architecture & InferenceContext (in progress)
  - **Architecture Decision**: "Static Load & Lock" - Custom I/O + Hardware MMU/MPU Protection
    - **Phase I: Load**: Custom I/O loads container data into physical memory regions
    - **Phase II: Lock**: Hardware MMU/MPU configured for read-only weight protection
    - **Phase III: Launch**: Hardware-enforced execution with crash-safe security
    - **Weights**: Hardware-protected read-only region (MMU/MPU enforced)
    - **Code**: Hardware-protected read-execute region (MMU/MPU enforced)
    - **Intermediates**: Read-write region (per-instance, mode-dependent)
  - **Week 4.3.1**: Platform-Specific MMU/MPU Firewall Implementation (ARM, x86, RISC-V)
  - **OS Independence**: Direct register access (ARM MPU, x86 Page Tables, RISC-V PMP)
  - **Cross-Platform**: 64KB alignment required (MMU/MPU granularity)
  - **AOT Code**: Must accept `InferenceContext` struct (physical memory region pointers)
- Next: Chunk streaming with double ring buffer, memory tiering

**Phase 3: AMD Backend Reference** ⏳ **NOT STARTED** (Weeks 9-13)

**Phase 4: Production Readiness** ⏳ **NOT STARTED** (Weeks 12-16)

---

## 📊 **Current State Assessment**

### ✅ **What's Working**

#### **Core Architecture**
- ✅ Universal container format (`.cnox`) with header, manifest, code, weights, IR
- ✅ Universal IR structure (graph-based, arena-allocated, bit-packed TensorId)
- ✅ ONNX translator (converts ONNX to Universal IR)
- ✅ Basic optimization (operator fusion, graph optimization)
- ✅ Weight compression (Zstd/LZ4 support)
- ✅ Container loading and validation (header-only, 23μs validation)
- ✅ Runtime execution framework (hypervisor managing containers)
- ✅ Trust-based validation (fast header checks)
- ✅ Security validations (path traversal, symlink attacks, file permissions)
- ✅ Executable memory allocation (custom I/O + hardware MMU/MPU protection)
- ✅ Performance metrics tracking

#### **Backend Infrastructure**
- ✅ Scalar fallback backend (pure Rust, 100% uptime)
- ✅ Backend SDK interface (8-trait interface for vendors)
- ✅ Backend translator pattern (vendor-specific codegen)
- ⚠️ Backend manager (needs implementation for vendor backend loading)

#### **Code Generation**
- ✅ CPU codegen (x86-64 assembly with SIMD)
- ✅ Stack-based buffer allocation (prologue/epilogue)
- ✅ Register rotation strategy (R11, R13 for intermediate tensors)
- ✅ SIMD loop generation (chunk processing with remainder handling)
- ✅ Operation codegen (arithmetic, activations, pooling, etc.)

#### **Hardware Abstraction Layers (HALs)**
- ✅ **Memory HAL** (`coelanox-core/src/memory_hal.rs`)
  - **Contract**: `MemoryTarget` (Nvme, Sd, Os, Custom), `MemoryTier` (Fast/Slow), `RegionOptions`, `MemoryRegion` trait, `MemoryHAL` trait (allocate_region, deallocate_region, move_memory, create_backing, get_capabilities), `MemoryBackendRegistry` (default_for_storage, default_for_executable).
  - **Packager**: Uses registry for container backing; `NvmeMemoryBackend` registered by default, implements `create_backing` via NvmeAllocator; fallback to NvmeAllocator when no backend.
  - **Runtime**: `memory_registry` on CoelanoxRuntime; executable allocation prefers HAL `default_for_executable()` then legacy mmap/VirtualAlloc; cleanup handles both Legacy and HAL blocks.
- ✅ **Protection HAL** (`coelanox-core/src/protection_hal.rs`)
  - **Contract**: `ProtectionTarget` (ArmMpu, X86PageTables, Esp32, Custom), `RegionPermissions` (read/write/execute), `ProtectionHAL` trait (configure_region, enable_protection, disable_protection, get_capabilities), `ProtectionBackendRegistry`.
  - **Status**: Defined in core and re-exported; not yet wired into runtime load/lock/launch (Phase II Lock integration pending).

### ⚠️ **What Needs Work**

#### **Critical Blockers**
- ✅ **Security hash (SHA-256)** - Implemented with three-command architecture (`package`, `verify`, `run`)
- ✅ **Backend architecture cleanup** - Hardware-specific backends removed, only `BackendTranslator` trait remains
- ✅ **Backend manager** - Implemented with discovery, validation, and AOT selection support
- ✅ **Scalar backend refactoring** - Authoritative scalar backend created in `coelanox-core/src/scalar/`
- ⚠️ **Scalar fallback module** - Structure created, needs full implementation in runtime

#### **Core Features Missing**
- ❌ **Chunk streaming** - No PCIe chunk streaming implementation
- ✅ **Memory HAL** - Implemented (trait, registry, packager create_backing, runtime executable allocation); NvmeMemoryBackend for storage; OS backend for executable when registered. Concrete tiering backends (async I/O, ring buffer) still Week 5+.
- ❌ **Max shape pre-allocation** - Max Shape Reader implemented; pre-allocation used at package time for backing size.
- ❌ **Double ring buffer** - Not implemented (Week 5)
- ❌ **Stack arena allocation** - Basic stack allocation exists, but not optimized (Week 7)
- ❌ **Memory tiering** - HAL in place; NVMe/RAM tiering with async I/O and ring buffer not yet implemented (Week 5–6)
- ⏳ **Protection HAL** - Trait and registry in core; runtime load/lock/launch integration (Phase II Lock) not yet wired

#### **Backend Implementations**
- ✅ **Backend manager** - Implemented with discovery, validation, and health check structure
- ⚠️ **Dynamic library loading** - Structure in place, needs platform-specific FFI implementation
- ❌ **AMD backend reference** - Reference implementation needed (or vendor-provided)

#### **Production Readiness**
- ❌ **Performance benchmarks** - No TensorRT comparison
- ❌ **PCIe utilization metrics** - No bandwidth measurement
- ❌ **Compute efficiency validation** - No 60% savings verification
- ❌ **Cold start validation** - No real-world testing
- ❌ **Vendor SDK documentation** - 8-trait interface not documented
- ❌ **Vendor testing framework** - No scalar reference validation

---

## 🎯 **Production MVP Goals**

### **Primary Objectives**
1. **Security**: SHA-256 container integrity verification
2. **Architecture**: Clean backend architecture (remove hardware-specific, add backend manager)
3. **Architecture**: Refactor scalar backend (single authoritative source in core)
4. **Core Features**: Chunk streaming, NVMe-backed memory, max shape pre-allocation
5. **Backend**: Backend manager for vendor backend loading/discovery
6. **Backend**: AOT binary selection (fat/slim containers at package time)
7. **Backend**: Two-stage selection (primary backend + additional backends)
8. **Backend**: AMD backend reference implementation (or vendor-provided)
9. **Validation**: Performance benchmarks vs TensorRT, PCIe utilization

### **Success Criteria**
- ✅ Container security hash implemented and verified
- ✅ Three-command architecture: `package`, `verify`, `run`
- ✅ Standalone verification command (`coelanox verify`)
- ✅ Run command with pre-deployment verification (`coelanox run`)
- ✅ Hardware-specific backends removed from runtime
- ✅ Backend manager implemented (vendor backend discovery/loading)
- ✅ AOT binary selection implemented (slim: `--h100`, fat: `--fat`, auto: `--auto`)
- ✅ Two-stage selection working (primary backend + `--backend-add amd,intel`)
- ✅ Container pre-compiled with selected backends (AOT compilation)
- ✅ Device ID mapping stored in container (primary = 0, added = 1, 2, 3...)
- ✅ Single authoritative scalar backend in `coelanox-core`
- ✅ Scalar fallback in dedicated module (100% uptime)
- ✅ Chunk streaming working with vendor backends (PCIe saturation)
- ✅ NVMe-backed memory with O(1) lookups (default: tiering enabled)
- ✅ Max shape pre-allocation (no runtime spikes)
- ✅ Memory tiering default with `--static-ram` flag for edge cases
- ✅ AMD backend reference implementation (or vendor-provided)
- ✅ Performance within 95% of TensorRT (with vendor backends)
- ✅ Compute efficiency savings validated (60% target)
- ✅ Vendor SDK documented and testable

---

## 📦 **Self-Contained Container Architecture** (Future Phase)

### **Vision: Container as Self-Contained Unit**

**Container** = IR + weights + code (CLF blobs) + embedded CLFMM + embedded CLFMP. Truly self-contained. No OS dependency for memory or protection.

**Runtime/Orchestrator** = OS-dependent. Manages multiple containers on the same card/server. Does **not** manage memory for containers. Schedules, routes I/O, load-balances.

### **Key Principles**

| Component | Role | OS-Dependent? |
|-----------|------|---------------|
| **Container** | Self-contained. Own memory via CLFMM, own protection via CLFMP. Uses CLF blobs (no ELF). | No |
| **Runtime** | Orchestrator only. Manages multiple containers. Scheduling, I/O routing. Does not allocate for containers. | Yes |
| **Drone** | No runtime. Container flashed directly. Bootloader/firmware loads container. Runs standalone. | N/A |

### **Changes Required**

1. **Container format** – Add CLFMM and CLFMP sections (embed or reference). Container carries its memory and protection logic.
2. **Packager** – Embed CLFMM and CLFMP when building. `--clfmm`, `--clfmp` or target-based discovery.
3. **Runtime** – Remove allocation for containers. Use embedded CLFMM/CLFMP from container. Runtime becomes orchestrator only (multi-container on same card).
4. **Bootstrap** – Minimal loader or container-as-loadable format so CLFMM blobs are executable before use.

### **Orchestrator Role on Server**

On a server, the runtime/orchestrator handles:

- **Loading containers** – Reads `.cnox` files, validates, caches.
- **Scheduling** – Decides which container(s) run, when, on which GPU/card.
- **I/O routing** – Sends inputs to containers, receives outputs.
- **Lifecycle** – Start/stop containers, cleanup.
- **Load balancing** – Distributes work across containers and cards.
- **Resource awareness** – Tracks which cards are busy, which are idle.

It does **not** allocate memory for containers; each container pulls its own via embedded CLFMM.

### **Production Implementation Suggestions (Working Product First)**

**Principle:** Avoid unnecessary features. Ship a working product first. Add optimizations only when needed.

| Suggestion | Why | Scope |
|------------|-----|-------|
| **Use `Arc<CoelanoxContainer>` in server container cache** | A 70GB container clone on every request would OOM the server. Store `HashMap<String, Arc<CoelanoxContainer>>` instead of `HashMap<String, CoelanoxContainer>`. Each request gets an `Arc` clone (8 bytes) not a full copy. | Server/orchestrator only |
| **Edge doesn't need Arc** | Edge devices run one container, loaded once. No request routing, no shared cache. | Edge/drone |
| **Cache eviction when using Arc** | Arc keeps memory until last reference drops. Implement eviction (TTL, LRU, max size) so you don't hold every loaded model forever. | Server |
| **Prioritize MVP over polish** | Get package → verify → run working end-to-end before adding chunk streaming, tiering, etc. | All |

### **Deployment Targets**

- **Server/Desktop (with OS):** Orchestrator manages multiple containers. Each container uses its own embedded CLFMM/CLFMP.
- **Drone/Edge (no OS):** Container flashed directly. No orchestrator. Bootloader loads container and runs.

### **Safety Model**

- Backend traits link ops only. AMD/Intel write kernels and pass tests vs scalar reference.
- Scalar is ground truth. Vendor kernels must match. Same container, different CLF per hardware.
- Container is Turing-incomplete (DAG, bounded). Safe for constrained environments.

---

## 🗺️ **Roadmap Phases**

### **Phase 1: Critical Fixes & Architecture Cleanup (Weeks 1-3)**
**Priority:** P0 - Blockers  
**Goal:** Fix critical issues and clean up backend architecture

#### **Week 1: Security & Backend Architecture**

**1.1 Implement SHA-256 Security Hash & Three-Command Architecture** ✅ **COMPLETED**
- **Location**: `coelanox-core/src/format.rs`, `coelanox-packager/src/lib.rs`, `coelanox-cli/src/main.rs`
- **Status**: ✅ Implemented with tests (`test_security_hash.rs`, `test_integration.rs`)
- **Tasks**:
  - Add `sha2` dependency to `coelanox-packager`
  - Calculate SHA-256 hash of entire container during packaging
  - Store hash in `CoelanoxHeader.security_hash` (set to `[0; 32]` during calculation, then write hash)
  - **Three-Command Architecture**:
    - `coelanox package`: Create container with hash (package time)
    - `coelanox verify`: Standalone verification command (pre-deployment check)
    - `coelanox run`: Verify + Load + Execute (defense in depth)
  - **Security Model**:
    - Container sits in cold storage (unverified, untrusted)
    - Hash verification happens at pre-deployment (before memory tier loading)
    - `verify` command: Reads container from cold storage, calculates hash, compares with stored hash
    - `run` command: Verifies hash, then loads into memory tier (NVMe + RAM), then executes
    - Once verified and loaded, container is trusted (no re-verification needed)
  - Implement shared verification logic (used by both `verify` and `run` commands)
  - Add hash mismatch error handling (don't load if verification fails)
  - **Hash Scope**: Entire container (header + manifest + code + weights + IR)
  - **Hash Timing**: Calculated after compression (hash final container as stored)
- **Success Criteria**:
  - Hash calculated during packaging (`coelanox package`)
  - Standalone verification command working (`coelanox verify`)
  - Run command verifies before loading (`coelanox run`)
  - Hash verified at pre-deployment (before memory tier loading)
  - Hash mismatch returns clear error (don't load container)
  - Shared verification logic (DRY principle)
  - No security vulnerabilities from hash bypass

**1.2 Remove Hardware-Specific Backends** ✅ **COMPLETED**
- **Location**: `coelanox-runtime/src/backends/`
- **Status**: ✅ All hardware-specific backends removed, only `BackendTranslator` trait remains
- **Tasks**:
  - Delete `gpu_backend.rs` (vendor-provided)
  - Delete `npu_backend.rs` (vendor-provided)
  - Delete `tpu_backend.rs` (vendor-provided)
  - Remove all hardware-specific backend code
  - Update imports and references
  - Clean up unused dependencies
- **Success Criteria**:
  - No hardware-specific backends in runtime
  - All references updated
  - Dependencies cleaned up
  - Runtime focuses on container management

**1.3 Create Backend Manager & AOT Binary Selection** ✅ **COMPLETED**
- **Location**: `coelanox-runtime/src/backend_manager.rs` (new), `coelanox-packager/src/lib.rs`
- **Status**: ✅ BackendManager implemented with discovery, validation, AOT selection (`--fat`, `--auto`, `--backend-add`). Tests: `test_backend_manager.rs`, `test_aot_selection.rs`
- **Note**: Dynamic library loading structure in place, needs platform-specific FFI implementation
- **Tasks**:
  - Create backend manager module
  - Implement vendor backend discovery (dynamic library loading)
  - Implement dynamic library loading (.so/.dylib/.dll)
  - Implement backend registry (register discovered backends)
  - Implement backend routing (route execution to appropriate backend)
  - Add backend health checks
  - Add backend fallback chain (vendor → scalar fallback)
  - **AOT Binary Selection (Package Time)**:
    - Implement slim binary selection: `--h100`, `--a100`, `--amd-gpu`, etc. (hardware-specific)
    - Implement fat binary selection: `--fat` (all backends included)
    - Implement two-stage selection: Primary backend + `--backend-add amd,intel,qualcomm` (additional backends)
    - Implement auto selection: `--auto` (auto-detect binary + auto-add available backends)
    - Container pre-compiled with selected backends (AOT compilation)
    - Store backend metadata in container manifest
    - Map device IDs to backends (primary = ID 0, added = ID 1, 2, 3...)
  - **Use Cases**:
    - Production: Slim container (hardware-specific, smaller size)
    - Home lab: Fat container (all backends, don't care about space)
- **Success Criteria**:
  - Backend manager implemented
  - Vendor backends discoverable
  - Dynamic loading working
  - Backend routing functional
  - Fallback chain working
  - Slim binary selection working (`--h100`, etc.)
  - Fat binary selection working (`--fat`)
  - Two-stage selection working (primary + `--backend-add`)
  - Auto selection working (`--auto`)
  - Container includes selected backends (AOT compiled)
  - Device ID mapping stored in container

#### **Week 2: Scalar Backend Refactoring & Fallback Module**

**2.1 Delete Packager Scalar Codegen** ✅ **COMPLETED**
- **Location**: `coelanox-packager/src/generators/cpu_codegen/operations/`
- **Status**: ✅ Removed `cpu_codegen` module using `iced-x86`, removed all direct code generators
- **Tasks**:
  - Identify all scalar operation codegen files
  - Remove scalar-specific codegen (keep SIMD codegen)
  - Update codegen to use core scalar operations
  - Remove `SCALAR_NOTE.md` and scalar-specific documentation
  - Update tests to use core scalar backend
- **Success Criteria**:
  - No scalar codegen in packager
  - Packager uses core scalar operations
  - Tests pass with core scalar backend
  - No duplication of scalar logic

**2.2 Create Authoritative Scalar Backend in Core** ✅ **COMPLETED**
- **Location**: `coelanox-core/src/scalar/`
- **Status**: ✅ Created with 1:1 OpType mapping, placeholder files for all operations (arithmetic, activations, math, pooling, normalization, reductions, matrix_ops, tensor_ops, comparisons, logical, misc)
- **Note**: Operations are placeholders, need implementation
- **Tasks**:
  - Create new `scalar/` module in `coelanox-core`
  - Move scalar operations from runtime to core
  - Implement pure Rust scalar operations (no SIMD, no SSE)
  - Organize by operation type (arithmetic, activations, pooling, etc.)
  - Add comprehensive documentation (golden reference)
  - Add unit tests for each operation
- **Success Criteria**:
  - Single authoritative scalar backend in core
  - Pure Rust implementation (no SIMD dependencies)
  - Comprehensive test coverage
  - Well-documented as golden reference

**2.3 Create Scalar Fallback Module** ⚠️ **PARTIAL**
- **Location**: `coelanox-runtime/src/fallback/` (new)
- **Status**: ⚠️ Structure created, needs full implementation and integration
- **Tasks**:
  - Create dedicated fallback module
  - Implement `FallbackBackend` struct
  - Use `coelanox-core` scalar operations
  - Implement backend traits for fallback
  - Integrate with backend manager
  - Verify 100% uptime with scalar fallback
- **Success Criteria**:
  - Fallback module created
  - Uses core scalar operations
  - Implements backend traits
  - Integrated with backend manager
  - 100% uptime maintained

#### **Week 3: Testing & Validation**

**3.1 Comprehensive Testing** ✅ **COMPLETED**
- **Status**: ✅ Tests created: `test_security_hash.rs`, `test_backend_manager.rs`, `test_aot_selection.rs`, `test_backend_cleanup.rs`, `test_integration.rs`, `test_scalar_fallback.rs`
- **Tasks**:
  - Add unit tests for security hash
  - Add tests for backend manager (discovery, loading, routing)
  - Add tests for scalar fallback backend
  - Add scalar backend correctness tests
  - Add integration tests for full pipeline (with vendor backends)
- **Success Criteria**:
  - All critical paths tested
  - No regressions introduced
  - Tests pass consistently
  - Coverage >90% for critical code

**3.2 Code Review & Documentation** ✅ **COMPLETED**
- **Status**: ✅ Documentation created: `docs/ARCHITECTURE.md`, `docs/SECURITY.md`, `docs/CODE_REVIEW_PHASE1.md`
- **Tasks**:
  - Review all changes for correctness
  - Update architecture documentation
  - Document scalar backend as golden reference
  - Update security documentation
- **Success Criteria**:
  - Code reviewed and approved
  - Documentation updated
  - Architecture clearly documented
  - Security model documented

---

### **Phase 2: Core Features (Weeks 4-8)**
**Priority:** P1 - Essential for MVP  
**Goal:** Implement core architectural features (chunk streaming, NVMe memory, max shape)

#### **Week 4: Max Shape Calculation & Pre-Allocation**

**4.1 Max Shape Reader**
- **Location**: `coelanox-core/src/optimization/`
- **Current State**: Shapes are already in the model → translator extracts them → IR has shapes
- **Tasks**:
  - Create `max_shape_reader.rs` module
  - Read shapes from Universal IR (already populated by translator)
  - Extract max bounds for dynamic dimensions (from model metadata)
  - Handle static shapes (fixed dimensions, already in IR)
  - Handle dynamic shapes (variable dimensions, max bounds from model)
  - Generate max shape metadata for container (for pre-allocation)
- **Architecture**:
  - **Shapes in Model**: Frontend model (ONNX, PyTorch, etc.) contains shapes
  - **Translator Extracts**: Translator reads shapes from model → populates IR
  - **IR Has Shapes**: `WeightTensor.shape`, `IntermediateTensor.shape`, `ModelMetadata.input_shapes/output_shapes`
  - **Max Shape Reader**: Reads shapes from IR (no calculation needed, just extraction)
  - **Dynamic Dimensions**: If model has max bounds (e.g., ONNX min/max), reader extracts them
- **Success Criteria**:
  - Max shapes read from IR for all tensors
  - Static shapes extracted correctly
  - Dynamic shape max bounds extracted (from model metadata)
  - Max shape metadata in container
  - No shape calculation needed (just reading from IR)

**4.2 NVMe Pre-Allocation**
- **Location**: `coelanox-packager/src/memory/`
- **Tasks**:
  - Create `nvme_allocator.rs` module
  - Calculate total NVMe size from max shapes
  - Pre-allocate zero-initialized space in unified `.cnox` container
  - Ensure 64KB alignment for all sections (cross-platform requirement)
  - Store NVMe backing offset and size in container manifest
  - **Architecture**: Unified file approach (backing section appended to container, not separate file)
- **Success Criteria**:
  - NVMe backing space pre-allocated in unified container
  - Max shape size allocation
  - 64KB alignment for all sections (Windows compatibility)
  - Allocation info in container manifest (offset + size)

**4.2.1 Memory HAL & Protection HAL (Implemented)** ✅
- **Location**: `coelanox-core/src/memory_hal.rs`, `coelanox-core/src/protection_hal.rs`
- **Purpose**: 100% abstraction for memory and protection backends (mirrors OpType + BackendTranslator pattern). Enables pluggable NVMe/SD/OS memory and ARM MPU/x86 MMU/ESP32 protection without hardcoding in core, packager, or runtime.
- **Memory HAL**:
  - **Core**: `MemoryTarget` (Nvme, Sd, Os, Custom), `MemoryTier` (Fast/Slow), `RegionOptions`, `MemoryRegion` trait, `MemoryHAL` trait (allocate_region, deallocate_region, move_memory, create_backing, get_capabilities), `MemoryBackendRegistry` (register, default_for_storage, default_for_executable).
  - **Packager**: `CoelanoxPackager` has `memory_registry`; package flow uses `default_for_storage()` for container backing; NvmeMemoryBackend registered by default (create_backing via NvmeAllocator); fallback to NvmeAllocator when no backend.
  - **Runtime**: `CoelanoxRuntime` has `memory_registry`; `allocate_and_copy_code` prefers HAL `default_for_executable()` then legacy mmap/VirtualAlloc; `ExecutableMemoryBlock` is Legacy or Hal; cleanup handles both.
- **Protection HAL**:
  - **Core**: `ProtectionTarget` (ArmMpu, X86PageTables, Esp32, Custom), `RegionPermissions`, `ProtectionHAL` trait (configure_region, enable_protection, disable_protection, get_capabilities), `ProtectionBackendRegistry`.
  - **Status**: Defined and re-exported from core; **not yet wired** into runtime load/lock/launch (Phase II Lock integration pending; Week 4.3 / 6.1).
- **Success Criteria** (Memory HAL): ✅ Trait and registry in core; packager uses HAL for create_backing; runtime uses HAL for executable when backend registered; NvmeMemoryBackend in packager. (Protection HAL): Trait and registry in core; runtime integration deferred to Static Load & Lock / Memory Tier Manager.

**4.3 Static Load & Lock Architecture & InferenceContext**
- **Location**: `coelanox-runtime/src/memory/`, `coelanox-core/src/runtime/`
- **Architecture Decision**: "Static Load & Lock" - Custom I/O + Hardware MMU/MPU Protection
  - **Phase I: Load (Custom I/O)**: Container data loaded into physical memory regions
    - Parse container header to identify section sizes
    - Allocate physical memory regions (Code, Weights, Scratchpad)
    - Decompress Code section → SRAM Region A (executable)
    - Decompress Weights section → SRAM Region B (read-only)
    - Zero-init Scratchpad → SRAM Region C (read-write)
  - **Phase II: Lock (Hardware MMU/MPU)**: Configure hardware protection registers
    - **Runtime Boot Code**: RX (Read + Execute)
    - **Container Kernels**: RX (Read + Execute)
    - **Model Weights**: RO (Read Only, No-Execute) - **Hardware Protected**
    - **Scratchpad/Stack**: RW (Read + Write, No-Execute)
    - Enable MMU/MPU (hardware gate closes, immutable memory map)
  - **Phase III: Launch (Execution)**: Hardware-enforced execution
    - Jump to container kernels entry point
    - Kernels read weights (hardware-protected, read-only)
    - Kernels write intermediates (scratchpad region)
    - **Security**: Write attempt to weights → Hardware exception (crash-safe)
- **Tasks**:
  - Create `static_load_lock.rs` module (platform abstraction: ARM MPU, x86 Page Tables, RISC-V PMP)
  - Implement Phase I: Custom I/O loader (container → physical memory)
    - **Platform-Specific Storage I/O** (see Task 4.3.2):
      - **ARM**: ARM storage controller registers (NVMe/Flash direct access via `arm_io.rs`)
      - **x86**: x86 storage controller registers (NVMe/Flash direct access via `x86_io.rs`)
      - **RISC-V**: RISC-V storage controller registers (NVMe/Flash direct access via `riscv_io.rs`)
      - **Same Platforms as MMU/MPU**: Uses same compile-time targets (`#[cfg(target_arch)]`)
      - **Trait-Based Abstraction**: `CustomIO` trait with platform implementations
      - **DMA Support**: Platform-specific DMA configuration for zero-copy transfers
      - **Physical Memory Allocation**: Platform-specific physical memory allocator
  - Implement Phase II: Hardware MMU/MPU configuration
    - Use platform-specific MMU/MPU firewall implementations (see Task 4.3.1)
    - **ARM**: MPU_RBAR/MPU_RASR registers (via `arm_mpu.rs`)
    - **x86**: Page Tables (CR3 register, via `x86_page_tables.rs`)
    - **RISC-V**: PMP (Physical Memory Protection, via `riscv_pmp.rs`)
  - Implement Phase III: Execution with hardware enforcement
  - Implement `InferenceContext` struct for AOT code:
    ```rust
    #[repr(C)]
    struct InferenceContext {
        weights_base: *const u8,  // Physical memory region (hardware-protected RO)
        backing_base: *mut u8,    // Scratchpad region (RW, per-instance)
        code_base: *const u8,     // Code region (hardware-protected RX)
    }
    ```
  - **Mode Detection**: Static RAM vs Tiering (affects scratchpad allocation)
  - **Static RAM Mode**: All intermediates in RAM (scratchpad region)
  - **Tiering Mode**: Async I/O for intermediates (ring buffer, see Week 5)
  - Pre-calculate TensorId → offset mapping at package time (store in manifest)
  - Load offset mapping at runtime (O(1) lookup via pre-calculated offsets)
- **Cross-Platform Considerations**:
  - **ARM**: MPU_RBAR/MPU_RASR registers (Cortex-M, Cortex-R, Cortex-A)
  - **x86**: Page Tables (CR3 register, standard x86-64)
  - **RISC-V**: PMP (Physical Memory Protection, RISC-V standard)
  - **Alignment**: All sections must be 64KB-aligned (MMU/MPU granularity requirement)
  - **Platform Abstraction**: Trait-based MMU/MPU configurator (ARM/x86/RISC-V implementations)
  - **Storage I/O**: Platform-specific storage controller access (ARM/x86/RISC-V implementations)
- **Success Criteria**:
  - Phase I: Custom I/O loader working (container → physical memory)
  - Phase II: Hardware MMU/MPU configuration working (weights RO, code RX, scratchpad RW)
  - Phase III: Hardware-enforced execution working (crash-safe on violation)
  - InferenceContext struct defined and used by AOT code
  - Pre-calculated offset mapping loaded at runtime
  - O(1) lookups via base pointer + pre-calculated offset
  - Cross-platform support (ARM, x86, RISC-V)
  - 64KB alignment enforced
  - OS independence validated (works on bare metal, firmware, embedded, RTOS)

**4.3.1 Platform-Specific MMU/MPU Firewall Implementation**
- **Location**: `coelanox-runtime/src/memory/mmu_firewall/`
- **Purpose**: Implement hardware MMU/MPU configuration code for each target architecture
- **Architecture**: Trait-based platform abstraction with architecture-specific implementations
- **Platforms**: ARM (aarch64), x86 (x86_64), RISC-V (riscv64) - compile-time targets via `#[cfg(target_arch)]`
- **Tasks**:
  - Create `mmu_firewall.rs` module with platform abstraction trait:
    ```rust
    pub trait MmuFirewall: Send + Sync {
        /// Configure memory protection regions
        fn configure_regions(&self, regions: &[MemoryRegion]) -> CoelanoxResult<()>;
        
        /// Enable MMU/MPU (hardware gate closes)
        fn enable(&self) -> CoelanoxResult<()>;
        
        /// Disable MMU/MPU (for shutdown/cleanup)
        fn disable(&self) -> CoelanoxResult<()>;
        
        /// Get architecture name
        fn architecture(&self) -> &str;
    }
    
    pub struct MemoryRegion {
        pub base: u64,
        pub size: usize,
        pub permissions: RegionPermissions,
    }
    
    pub enum RegionPermissions {
        ReadOnly,      // RO (Read Only, No-Execute) - for weights
        ReadExecute,  // RX (Read + Execute) - for code
        ReadWrite,    // RW (Read + Write, No-Execute) - for scratchpad
    }
    ```
  - **ARM MPU Implementation** (`arm_mpu.rs`):
    - Implement ARM MPU configuration (MPU_RBAR/MPU_RASR registers)
    - Support Cortex-M, Cortex-R, Cortex-A variants
    - Handle MPU region count limits (typically 8-16 regions)
    - Implement region attribute encoding (AP, XN, TEX, S, C, B bits)
    - Handle alignment requirements (32-byte minimum for Cortex-M)
    - Test with ARM Cortex-M4/M7 (embedded), Cortex-A53/A72 (application processors)
  - **x86 Page Tables Implementation** (`x86_page_tables.rs`):
    - Implement x86-64 page table structure (4-level paging)
    - Configure CR3 register (page table base)
    - Set page table entries (PDE, PTE) with protection bits
    - Handle 4KB/2MB/1GB page sizes
    - Implement page table walk and entry modification
    - Handle alignment requirements (4KB page boundaries)
    - Test with x86-64 processors (Intel, AMD)
  - **RISC-V PMP Implementation** (`riscv_pmp.rs`):
    - Implement RISC-V Physical Memory Protection (PMP)
    - Configure PMP registers (PMPADDR, PMPCFG)
    - Handle PMP region count limits (typically 8-16 regions)
    - Implement region matching logic (TOR, NAPOT, NA4 modes)
    - Handle alignment requirements (4-byte minimum)
    - Test with RISC-V processors (SiFive, Andes, etc.)
  - **Platform Detection & Factory**:
    - Implement architecture detection (compile-time via `#[cfg(target_arch)]`)
    - Create factory function to instantiate correct MMU/MPU implementation
    - Handle unsupported architectures gracefully (fallback to software-only)
  - **Testing & Validation**:
    - Unit tests for each architecture implementation
    - Integration tests for region configuration
    - Security tests (verify hardware exceptions on violation)
    - Cross-platform validation (test on real hardware when possible)
    - QEMU/emulator testing for architectures without physical hardware
- **Success Criteria**:
  - Platform abstraction trait defined and implemented
  - ARM MPU implementation working (Cortex-M, Cortex-R, Cortex-A)
  - x86 Page Tables implementation working (x86-64)
  - RISC-V PMP implementation working
  - Platform detection and factory working (compile-time `#[cfg(target_arch)]`)
  - All implementations tested and validated
  - Hardware exceptions verified on write attempts to read-only regions
  - Documentation complete for each architecture

**4.3.2 Platform-Specific Storage I/O Implementation**
- **Location**: `coelanox-runtime/src/memory/custom_io/`
- **Purpose**: Implement platform-specific storage controller access for direct hardware I/O (no OS)
- **Architecture**: Trait-based platform abstraction with architecture-specific implementations
- **Platforms**: ARM (aarch64), x86 (x86_64), RISC-V (riscv64) - **Same compile-time targets as MMU/MPU firewall**
- **Key Insight**: Uses same `#[cfg(target_arch)]` compile-time selection as MMU/MPU firewall (not additional platform work)
- **Tasks**:
  - Create `custom_io.rs` module with platform abstraction trait:
    ```rust
    pub trait CustomIO: Send + Sync {
        /// Read data from storage (direct hardware access, no OS)
        fn read_storage(&self, offset: u64, size: usize) -> CoelanoxResult<Vec<u8>>;
        
        /// Allocate physical memory (no virtual memory translation)
        fn allocate_physical_memory(&self, size: usize, alignment: usize) -> CoelanoxResult<*mut u8>;
        
        /// Configure DMA transfer (storage → physical memory)
        fn configure_dma(&self, source: u64, dest: *mut u8, size: usize) -> CoelanoxResult<()>;
        
        /// Get architecture name
        fn architecture(&self) -> &str;
    }
    ```
  - **ARM Storage I/O Implementation** (`arm_io.rs`):
    - Direct access to ARM storage controller registers (NVMe/Flash)
    - ARM-specific DMA configuration (DMA controller registers)
    - Physical memory allocation (ARM memory controller)
    - Support for ARM Cortex-M, Cortex-R, Cortex-A variants
    - Handle ARM-specific alignment requirements (32-byte minimum for DMA)
    - Test with ARM Cortex-M4/M7 (embedded), Cortex-A53/A72 (application processors)
  - **x86 Storage I/O Implementation** (`x86_io.rs`):
    - Direct access to x86 storage controller registers (NVMe/Flash)
    - x86-specific DMA configuration (DMA controller, PCIe)
    - Physical memory allocation (x86 memory controller)
    - Handle x86-specific alignment requirements (4KB page boundaries)
    - Support for Intel and AMD storage controllers
    - Test with x86-64 processors (Intel, AMD)
  - **RISC-V Storage I/O Implementation** (`riscv_io.rs`):
    - Direct access to RISC-V storage controller registers (NVMe/Flash)
    - RISC-V-specific DMA configuration (DMA controller)
    - Physical memory allocation (RISC-V memory controller)
    - Handle RISC-V-specific alignment requirements (4-byte minimum)
    - Test with RISC-V processors (SiFive, Andes, etc.)
  - **Platform Detection & Factory**:
    - Use same compile-time detection as MMU/MPU firewall (`#[cfg(target_arch)]`)
    - Create factory function to instantiate correct storage I/O implementation
    - Handle unsupported architectures gracefully (fallback to CPU copy if DMA unavailable)
  - **Integration with Static Load & Lock**:
    - Custom I/O used in Phase I: Load (container → physical memory)
    - Works with MMU/MPU firewall (Phase II: Lock)
    - Zero-copy architecture (DMA transfer, no OS page cache)
    - Physical memory addresses (no virtual memory translation)
  - **Testing & Validation**:
    - Unit tests for each architecture implementation
    - Integration tests for storage read operations
    - DMA transfer validation (verify zero-copy)
    - Performance tests (compare to OS file I/O)
    - Cross-platform validation (test on real hardware when possible)
    - QEMU/emulator testing for architectures without physical hardware
- **Success Criteria**:
  - Platform abstraction trait defined and implemented
  - ARM storage I/O implementation working (direct hardware access)
  - x86 storage I/O implementation working (direct hardware access)
  - RISC-V storage I/O implementation working (direct hardware access)
  - Platform detection and factory working (compile-time `#[cfg(target_arch)]`)
  - DMA transfers working (zero-copy, no OS overhead)
  - Physical memory allocation working (no virtual memory)
  - All implementations tested and validated
  - Performance validated (10-100x faster than OS file I/O)
  - Documentation complete for each architecture

#### **Week 5: Chunk Streaming Infrastructure**

**5.1 Chunk Streaming Core**
- **Location**: `coelanox-core/src/streaming/`
- **Tasks**:
  - Create `chunk_streaming.rs` module
  - Implement chunk size calculator (PCIe-optimized for GPU, memory-optimized for CPU)
  - Implement chunk buffer management
  - Implement chunk transfer coordination
  - Add chunk metadata to container
  - **OS Bypass Philosophy**: Manual chunk streaming (no OS prefetching, direct control)
  - **Purpose**: Bypass bandwidth bottlenecks (memory bandwidth for static RAM, storage bandwidth for tiering)
  - **Performance Justification**: Enables faster models on lower hardware (better memory bandwidth utilization)
- **Success Criteria**:
  - Chunk streaming infrastructure in core
  - PCIe-optimized chunk sizes (GPU backends)
  - Memory-optimized chunk sizes (CPU backends, static RAM)
  - Chunk buffer management working
  - Chunk metadata in container
  - OS bypassed (manual prefetching, no OS involvement)

**5.2 Double Ring Buffer & Async I/O for NVMe Backing**
- **Location**: `coelanox-core/src/streaming/`, `coelanox-runtime/src/memory/`
- **Architecture**: NVMe backing uses async I/O (custom I/O) for latency hiding
- **Tasks**:
  - Implement double ring buffer structure (Buffer A + Buffer B in RAM)
  - Implement async I/O integration:
    - **Linux**: `io_uring` for async read/write operations
    - **Windows**: `IOCP` / Overlapped I/O for async operations
  - Open container file with direct I/O flags:
    - **Linux**: `O_DIRECT` (bypass OS page cache)
    - **Windows**: `FILE_FLAG_NO_BUFFERING` (bypass OS cache)
  - Implement producer/consumer pattern (compute on one buffer, I/O on other)
  - Implement buffer rotation logic (switch between A and B)
  - Add buffer synchronization (thread-safe concurrent access)
  - **Key Insight**: Ring buffer in RAM, async I/O to/from NVMe backing section
  - **Purpose**: Overlap compute + transfer (prefetch chunk N+1 while computing chunk N)
  - **Latency Hiding**: Non-blocking async I/O enables true overlap (no page faults)
- **Success Criteria**:
  - Double ring buffer implemented (in RAM, custom I/O managed)
  - Async I/O working (io_uring on Linux, IOCP on Windows)
  - Direct I/O flags configured (bypass OS cache)
  - Producer/consumer pattern working
  - Buffer rotation smooth
  - Thread-safe concurrent access
  - Latency hiding validated (compute + I/O overlap)
  - No page faults during execution (predictable performance)

**5.3 Chunk Streaming for Static RAM & Tiering**
- **Location**: `coelanox-core/src/streaming/`
- **Tasks**:
  - Implement chunk streaming for static RAM (memory bandwidth optimization)
  - Implement chunk streaming for tiering (NVMe → RAM transfer optimization)
  - **Static RAM**: Prefetch within RAM (overlap compute + memory access)
  - **Tiering**: Prefetch from NVMe to RAM (overlap compute + NVMe transfer)
  - **OS Bypass**: Direct I/O, manual prefetching (no OS page cache, no OS prefetching)
  - **Performance**: Enables faster models on lower hardware (better bandwidth utilization)
  - **Architecture**: Same strategy (prefetch to avoid bottlenecks), different source (RAM vs NVMe)
- **Success Criteria**:
  - Chunk streaming working with static RAM (memory bandwidth optimization)
  - Chunk streaming working with tiering (storage bandwidth optimization)
  - Performance improvements validated (faster models on lower hardware)
  - OS bypassed (manual control, no OS jitter/latency)

**5.4 PCIe Chunk Streaming Integration**
- **Location**: `coelanox-core/src/streaming/`
- **Tasks**:
  - Create chunk streaming interface for vendor backends
  - Implement PCIe chunk transfer coordination (GPU backends)
  - Add chunk streaming to backend traits (SDK)
  - Implement chunk overlap coordination (compute + transfer)
  - Add PCIe utilization metrics
  - Document chunk streaming for vendors
  - **OS Bypass**: Manual PCIe transfer coordination (vendor-specific, direct control)
- **Success Criteria**:
  - Chunk streaming interface in core
  - Vendor backends can implement chunk streaming
  - PCIe transfer coordination working
  - Chunk overlap supported
  - PCIe utilization >90% (with vendor backends)
  - OS bypassed (manual control, predictable performance)

#### **Week 6: Memory Tiering System**

**6.1 Memory Tier Manager (Static Load & Lock)**
- **Location**: `coelanox-core/src/memory/`, `coelanox-runtime/src/memory/`
- **Abstraction**: Memory HAL (`coelanox-core/src/memory_hal.rs`) is the contract for memory backends; tier backends (NVMe, OS, SD) implement `MemoryHAL`. Protection HAL (`coelanox-core/src/protection_hal.rs`) is the contract for MMU/MPU; Lock phase will use `ProtectionBackendRegistry` when wired.
- **Architecture**: "Static Load & Lock" - Custom I/O + Hardware MMU/MPU Protection
  - **Weights (Both Modes)**: Hardware-protected read-only region (MMU/MPU enforced)
    - Loaded via Custom I/O (Phase I)
    - Locked as RO via hardware MMU/MPU (Phase II)
    - Hardware-enforced protection (crash-safe on write attempt)
  - **Code (Both Modes)**: Hardware-protected read-execute region (MMU/MPU enforced)
    - Loaded via Custom I/O (Phase I)
    - Locked as RX via hardware MMU/MPU (Phase II)
    - Hardware-enforced protection
  - **Static RAM Mode (`--static-ram`)**: All intermediates in RAM scratchpad
    - Small models (everything fits in RAM)
    - Scratchpad region allocated in RAM (per-instance, RW)
    - Simple, efficient, zero-copy, no async I/O needed
  - **Tiering Mode (default)**: Async I/O for intermediates
    - Large models (need NVMe backing)
    - Async I/O for intermediates (ring buffer, latency hiding)
    - Predictable performance, full control
- **Tasks**:
  - Create `memory_tier.rs` module
  - Implement mode detection (static RAM vs tiering)
  - **Weights (Both Modes)**:
    - Loaded via Custom I/O (Phase I: Load)
    - Locked as RO via hardware MMU/MPU (Phase II: Lock)
    - Hardware-enforced protection (Phase III: Launch)
    - **ARM**: MPU_RBAR/MPU_RASR registers
    - **x86**: Page Tables (CR3 register)
    - **RISC-V**: PMP (Physical Memory Protection)
    - **Result**: Hardware-protected read-only region, crash-safe
  - **Code (Both Modes)**:
    - Loaded via Custom I/O (Phase I: Load)
    - Locked as RX via hardware MMU/MPU (Phase II: Lock)
    - Hardware-enforced protection (Phase III: Launch)
  - **Static RAM Mode**:
    - Scratchpad region allocated in RAM (per-instance, RW)
    - Simple, efficient, zero-copy
  - **Tiering Mode**:
    - Async I/O for intermediates (ring buffer, see Week 5)
      - Ring buffer in RAM (Buffer A + Buffer B)
      - Async I/O to/from NVMe backing section
      - Direct I/O (bypass OS cache)
    - Latency hiding (overlap compute + I/O)
  - **Default behavior**: Tiering enabled by default (optimal for most use cases)
  - **Platform support**: ARM (MPU), x86 (Page Tables), RISC-V (PMP)
  - **OS Independence**: Direct register access (works on bare metal, firmware, embedded, RTOS)
  - **Smart fallback**: If tiering unavailable → fallback to static RAM automatically
  - **Chunk Streaming Integration**: Works with ring buffer (async I/O prefetch from NVMe to RAM)
- **Success Criteria**:
  - Weights hardware-protected (both modes, MMU/MPU enforced, crash-safe)
  - Code hardware-protected (both modes, MMU/MPU enforced)
  - Static RAM mode: Scratchpad in RAM (simple, efficient)
  - Tiering mode: Async I/O for intermediates (latency hiding, predictable)
  - Memory tiering system implemented
  - Cross-platform support (ARM, x86, RISC-V)
  - Graceful fallback if tiering unavailable
  - OS independence validated (bare metal, firmware, embedded, RTOS)

**6.2 Dynamic RAM Caching**
- **Location**: `coelanox-runtime/src/memory/`
- **Tasks**:
  - Implement RAM cache for active tensors
  - Implement cache eviction policy (LRU)
  - Implement cache hit/miss tracking
  - Add cache size management
  - Optimize for zero-copy operations
  - **Small model optimization**: All weights cached in RAM (transparent to user)
- **Success Criteria**:
  - RAM cache for active tensors
  - Cache eviction working correctly
  - Cache hit/miss tracking
  - Zero-copy operations optimized
  - Small models perform identically to all-RAM (100% cache hit rate)

**6.3 Static RAM Flag & Documentation**
- **Location**: `coelanox-runtime/src/runtime.rs`, `coelanox-cli/src/main.rs`
- **Architecture**: "Static Load & Lock" - Custom I/O + Hardware MMU/MPU Protection
  - **Weights (Both Modes)**: Hardware-protected read-only (MMU/MPU enforced, crash-safe)
  - **Code (Both Modes)**: Hardware-protected read-execute (MMU/MPU enforced)
  - **Static RAM Mode**: All intermediates in RAM scratchpad (per-instance, RW)
  - **Tiering Mode**: Async I/O for intermediates (ring buffer, latency hiding)
  - **OS Independence**: Direct register access (works on bare metal, firmware, embedded, RTOS)
- **Tasks**:
  - Add `--static-ram` command-line flag
  - Implement static RAM mode (scratchpad in RAM, weights/code hardware-protected)
  - Add flag validation (warn if model too large for all-RAM)
  - **Static RAM Strategy**: All data in RAM (weights/code hardware-protected, scratchpad RW)
    - Weights: Hardware-protected RO (MMU/MPU enforced, crash-safe)
    - Code: Hardware-protected RX (MMU/MPU enforced)
    - Intermediates: Scratchpad region in RAM (per-instance, RW)
    - No async I/O needed (everything already in RAM)
    - Simple, efficient, zero-copy
  - **Tiering Strategy**: Weights/code hardware-protected, async I/O for intermediates
    - Weights: Hardware-protected RO (MMU/MPU enforced, crash-safe)
    - Code: Hardware-protected RX (MMU/MPU enforced)
    - Intermediates: Async I/O with ring buffer (latency hiding, predictable)
  - **Documentation**:
    - Document `--static-ram` flag in CLI help
    - Add use case examples (IoT, embedded, development, small models)
    - Document when to use static RAM vs default tiering
    - Document "Static Load & Lock" architecture (Phase I: Load, Phase II: Lock, Phase III: Launch)
    - Document hardware MMU/MPU protection (weights RO, code RX, crash-safe)
    - Document OS independence (works on bare metal, firmware, embedded, RTOS)
    - Document platform-specific behavior (ARM MPU, x86 Page Tables, RISC-V PMP)
    - Add troubleshooting guide for memory issues
- **Success Criteria**:
  - `--static-ram` flag implemented
  - Static RAM mode working correctly (hardware-protected weights/code, scratchpad in RAM)
  - Tiering mode working correctly (hardware-protected weights/code, async I/O for intermediates)
  - Flag validation with helpful error messages
  - Comprehensive documentation in CLI help
  - Use case examples documented
  - Troubleshooting guide complete
  - Clear distinction between static RAM (scratchpad in RAM) and tiering (async I/O intermediates)

**6.4 Memory Access Optimizer**
- **Location**: `coelanox-core/src/memory/`
- **Tasks**:
  - Implement access pattern analysis
  - Implement prefetching for predicted access
  - Implement memory access optimization
  - Add bandwidth optimization
  - Test with real workloads
- **Success Criteria**:
  - Access pattern analysis working
  - Prefetching implemented
  - Memory access optimized
  - Bandwidth utilization improved

#### **Week 7: Stack Arena Optimization**

**7.1 CSV Optimization Parameter Integration**
- **Location**: `coelanox-packager/src/optimization/`
- **Current State**: CSV loader exists but is not wired into optimizer or code generation
- **Tasks**:
  - Integrate `CsvOptimizationLoader` into the optimizer pipeline
  - Load CSV parameters during IR optimization
  - Attach CSV parameters to nodes as attributes (`tile_size_x`, `tile_size_y`, `unroll_factor`, `simd_width`, `latency`)
  - Support vendor-specific CSV (`data/vendor/{vendor}/{hardware}.csv`), default CSV (`data/default/cpu.csv`), and embedded defaults
  - Validate CSV parameters per operation; handle missing/invalid rows with graceful defaults
- **CSV Format**:
  ```
  operation,precision,tile_size_x,tile_size_y,unroll_factor,simd_width,latency
  MatMul,FP32,256,256,2,8,1.2ms
  Add,FP32,64,64,2,8,0.1ms
  Convolution,FP32,32,32,4,8,0.5ms
  ```
- **Success Criteria**:
  - CSV loader invoked in optimizer
  - Parameters validated and added as node attributes
  - Vendor → default → embedded fallback chain works
  - Missing CSV handled without failing the build

**7.2 Template Wrapper Code Generation**
- **Location**: `coelanox-packager/src/generators/` (or vendor `BackendTranslator` impls)
- **Architecture**: Template wrapper around bit-exact operation cores
- **Tasks**:
  - Keep operation cores bit-exact and portable (no hardware-specific logic inside)
  - Generate wrapper code from CSV parameters:
    - Loop tiling (`tile_size_x`, `tile_size_y`)
    - Loop unrolling (`unroll_factor`)
    - SIMD vectorization (`simd_width`)
    - Memory layout tweaks as attributes
  - Use `ir.metadata.input_shapes` (and `custom_input_shapes` override) to set correct loop bounds and signatures
  - Produce final machine code via the template wrapper; core code remains untouched
- **Success Criteria**:
  - Bit-exact cores unchanged
  - Wrappers reflect CSV params (tiling, unrolling, SIMD)
  - Input shapes (including custom overrides) flow into wrapper generation
  - Generated code maintains bit-exact correctness while applying hardware tuning

**7.3 BackendTranslator Integration for CSV/Template**
- **Location**: `coelanox-packager/src/backend_trait/`
- **Tasks**:
  - Ensure `BackendTranslator` receives CSV-driven params via `CoelanoxAiOperationParams`
  - **AOT Code Generation**: Generated code must accept `InferenceContext` struct as parameter
    - Context contains: `weights_base` (hardware-protected RO region), `backing_base` (scratchpad/ring buffer)
    - Code accesses weights via: `weights_base + offset` (from pre-calculated mapping)
    - Code accesses intermediates via: `backing_base + offset` (from pre-calculated mapping)
    - **Architecture**: Position-independent code (relative addressing via context pointers)
  - Document the template wrapper pattern for vendors
  - (Optional) Provide a minimal reference showing how to consume CSV params in template generation
- **Success Criteria**:
  - CSV parameters available to vendor translators
  - AOT code accepts `InferenceContext` parameter
  - Code uses context pointers for memory access (weights_base, backing_base)
  - Template-wrapper pattern clearly documented for partners
  - Reference flow validated end-to-end (CSV → optimizer attrs → translator/template → code)

**7.4 Stack Arena Allocator (Execution Environment)**
- **Location**: `coelanox-core/src/memory/`
- **Architecture**: Stack arena is the execution environment (like game engines)
  - **Container-Internal**: Generated code allocates one large stack frame (single `sub rsp`)
  - **All Intermediates**: Allocated from this single stack arena (no malloc)
  - **Per-Instance**: Each inference gets its own stack arena (isolated execution environment)
  - **Integration**: Works with ring buffer (stack arena for small tensors, ring buffer for large streaming data)
- **Tasks**:
  - Create `stack_arena.rs` module
  - Design stack arena pattern/algorithm for vendor codegen (helper library)
  - Calculate total arena size from max shapes (at package time)
  - Calculate tensor offsets within arena (at package time, stored in manifest)
  - Implement alignment handling (16-byte alignment for SIMD, 64KB for cross-platform)
  - **Vendor Implementation**: Vendors generate code with stack arena allocation in their `BackendTranslator`
    - Prologue: Single `sub rsp, TOTAL_ARENA_SIZE` (one allocation for all intermediates)
    - Body: All operations use arena offsets (pre-calculated, O(1) access)
    - Epilogue: Single `add rsp, TOTAL_ARENA_SIZE` (one deallocation)
- **Architecture**:
  - **Pattern**: Single arena allocation (one `sub rsp` for all intermediates)
  - **Offsets**: Pre-calculated at package time (TensorId → offset mapping)
  - **Benefits**: Faster allocation, better cache locality, simpler lifetime management, zero runtime allocation
  - **Game Engine Pattern**: All objects from one arena, single deallocation at end
- **Success Criteria**:
  - Stack arena pattern/algorithm documented
  - Offset calculation at package time (stored in manifest)
  - Vendor codegen generates stack arena allocation (prologue/epilogue)
  - Alignment handled properly (16-byte SIMD, 64KB cross-platform)
  - Performance optimized (single allocation, O(1) offset access)

**7.5 Buffer Pool Abstraction**
- **Location**: `coelanox-core/src/memory/`
- **Purpose**: Runtime buffer reuse across multiple executions (different from stack arena)
- **Tasks**:
  - Create `buffer_pool.rs` module
  - Implement buffer pool for intermediate tensors
  - Implement buffer reuse logic (zero-copy across executions)
  - Add buffer pool metrics (hit rate, reuse efficiency)
  - Optimize for zero-copy (reuse buffers when shapes match)
- **Architecture**:
  - **Stack Arena**: Package-time allocation strategy (how buffers are allocated in generated code)
  - **Buffer Pool**: Runtime optimization (reuse buffers across multiple `execute()` calls)
  - **Relationship**: Stack arena allocates buffers, buffer pool reuses them
  - **Use Case**: When same model runs multiple times with same shapes, reuse buffers
- **Success Criteria**:
  - Buffer pool implemented
  - Buffer reuse working (zero-copy when shapes match)
  - Pool metrics tracked
  - Zero-copy optimized
  - Performance improvement measured

**7.6 Integration with Codegen**
- **Location**: `coelanox-packager/src/generators/cpu_codegen/`
- **Tasks**:
  - Integrate stack arena with codegen
  - Update prologue/epilogue for arena allocation (single allocation)
  - Optimize buffer reuse in codegen (register rotation strategy)
  - Test with real models
- **Current Codegen**:
  - Prologue: Allocates all intermediate buffers on stack (sequential)
  - Register rotation: R11, R13 for first 2 tensors, stack offsets for rest
  - Epilogue: Deallocates stack buffers
- **Optimized Codegen**:
  - Prologue: Single arena allocation (one `sub rsp` for all intermediates)
  - Buffer mapping: TensorId → arena offset (O(1) lookup)
  - Epilogue: Single deallocation (one `add rsp`)
- **Success Criteria**:
  - Stack arena integrated with codegen
  - Prologue/epilogue updated (single allocation)
  - Buffer reuse optimized
  - Real models working
  - Performance improvement measured

#### **Week 8: Phase 2 Completion & Polish**

**8.1 Feature Integration & Validation**
- **Location**: Integration across all Phase 2 components
- **Tasks**:
  - Integrate all Phase 2 features (max shapes, NVMe pre-allocation, chunk streaming, memory tiering, stack arena)
  - Validate end-to-end pipeline (package → verify → run)
  - Test feature interactions (tiering + chunk streaming, stack arena + buffer pool)
  - Fix any integration issues
  - Performance baseline measurements
- **Success Criteria**:
  - All Phase 2 features working together
  - End-to-end pipeline validated
  - No feature conflicts
  - Performance baselines established

**8.2 Documentation & API Finalization**
- **Location**: Documentation updates, API stabilization
- **Tasks**:
  - Document all Phase 2 features
  - Finalize public APIs
  - Add usage examples
  - Update architecture documentation
  - Create migration guide (if needed)
- **Success Criteria**:
  - All features documented
  - APIs stable and documented
  - Examples provided
  - Architecture docs updated

**8.3 Performance Optimization Pass**
- **Location**: Performance tuning across Phase 2 features
- **Tasks**:
  - Profile and optimize critical paths
  - Optimize memory access patterns
  - Reduce allocation overhead
  - Optimize chunk streaming performance
  - Tune tiering promotion/demotion thresholds
- **Success Criteria**:
  - Performance improvements measured
  - Critical paths optimized
  - Memory overhead minimized
  - Chunk streaming optimized

**8.4 Phase 2 Testing & Validation**
- **Location**: Comprehensive testing of Phase 2
- **Tasks**:
  - Unit tests for all Phase 2 features
  - Integration tests for feature combinations
  - Performance regression tests
  - Memory leak detection
  - Edge case testing
- **Success Criteria**:
  - All tests passing
  - No performance regressions
  - No memory leaks
  - Edge cases handled

---

### **Phase 3: AMD Backend Reference Implementation (Weeks 9-13)**

#### **Week 9: Testing & Integration (Phase 2 Final Validation)**

**9.1 Integration Testing**
- **Tasks**:
  - Test max shape pre-allocation with real models
  - Test chunk streaming with GPU workloads
  - Test memory tiering with large models
  - Test stack arena with various workloads
  - Measure performance improvements
- **Success Criteria**:
  - All features working together
  - Performance improvements validated
  - No regressions
  - Real-world workloads tested

**9.2 Performance Validation**
- **Tasks**:
  - Measure memory allocation spikes (should be zero)
  - Measure PCIe utilization (target >90%)
  - Measure bandwidth improvements
  - Measure cache hit rates
  - Compare before/after performance
- **Success Criteria**:
  - No allocation spikes
  - PCIe utilization >90%
  - Bandwidth improvements measured
  - Cache hit rates >80%
  - Performance improvements validated

---

### **Phase 3: AMD Backend Reference Implementation (Weeks 9-13)**
**Priority:** P1 - Essential for MVP  
**Goal:** Reference AMD backend implementation (or vendor-provided) with ROCm/HIP, pure SASS, PCIe streaming

**Note:** This can be either a reference implementation in a separate crate or vendor-provided. The runtime will load it via the backend manager.

#### **Week 10: ROCm/HIP Integration**

**10.1 AMD Backend Reference Structure**
- **Location**: `coelanox-backend-amd/` (new crate) or vendor-provided
- **Tasks**:
  - Create AMD backend crate (or vendor provides)
  - Implement ROCm context initialization
  - Implement HIP device detection
  - Implement HIP memory management
  - Add ROCm error handling
  - Implement backend traits (8-trait interface)
- **Success Criteria**:
  - AMD backend structure created
  - ROCm context initialized
  - HIP devices detected
  - Memory management working
  - Backend traits implemented

**10.2 HIP Kernel Execution**
- **Location**: `coelanox-backend-amd/src/` or vendor implementation
- **Tasks**:
  - Implement HIP kernel launch
  - Implement HIP stream management
  - Implement HIP event synchronization
  - Add kernel execution metrics
  - Test with simple kernels
- **Success Criteria**:
  - HIP kernels launching
  - Stream management working
  - Event synchronization correct
  - Execution metrics tracked

**10.3 PCIe Chunk Streaming for AMD**
- **Location**: `coelanox-backend-amd/src/` or vendor implementation
- **Tasks**:
  - Implement chunk streaming interface
  - Implement PCIe transfers with ROCm
  - Implement chunk overlap for AMD
  - Add AMD-specific optimizations
  - Test PCIe utilization
- **Success Criteria**:
  - Chunk streaming working with AMD
  - PCIe transfers optimized
  - Chunk overlap implemented
  - PCIe utilization >90%

#### **Week 11: Pure SASS Codegen**

**11.1 SASS Code Generator**
- **Location**: `coelanox-packager/src/generators/amd_codegen/` (new)
- **Tasks**:
  - Create AMD codegen module
  - Implement SASS assembly generation
  - Implement AMD-specific optimizations
  - Add Tensor Core support (if available)
  - Generate optimized kernels
- **Success Criteria**:
  - SASS codegen implemented
  - AMD optimizations applied
  - Kernels generated correctly
  - Performance optimized

**11.2 Operation Kernels**
- **Location**: `coelanox-packager/src/generators/amd_codegen/operations/`
- **Tasks**:
  - Implement MatMul kernel (SASS)
  - Implement Convolution kernel (SASS)
  - Implement Activation kernels (SASS)
  - Implement Pooling kernels (SASS)
  - Test each kernel
- **Success Criteria**:
  - All operation kernels implemented
  - Kernels working correctly
  - Performance optimized
  - Tests passing

**11.3 Kernel Optimization**
- **Location**: `coelanox-packager/src/generators/amd_codegen/`
- **Tasks**:
  - Implement kernel autotuning
  - Implement register optimization
  - Implement shared memory optimization
  - Implement warp-level optimizations
  - Test performance improvements
- **Success Criteria**:
  - Kernel autotuning working
  - Register usage optimized
  - Shared memory optimized
  - Performance improvements validated

#### **Week 12: Integration & Testing**

**12.1 Backend Integration**
- **Tasks**:
  - Load AMD backend via backend manager
  - Test end-to-end pipeline with AMD backend
  - Test with real models
  - Measure performance
  - Compare with scalar fallback
  - Test fallback chain (AMD → scalar)
- **Success Criteria**:
  - AMD backend loaded via backend manager
  - End-to-end pipeline working
  - Real models running
  - Performance improvements measured
  - Fallback chain working

**12.2 Performance Validation**
- **Tasks**:
  - Measure GPU utilization (target 110-130%)
  - Measure PCIe utilization (target >90%)
  - Measure compute efficiency
  - Compare with TensorRT (target 95%+)
  - Validate 60% compute savings
- **Success Criteria**:
  - GPU utilization 110-130%
  - PCIe utilization >90%
  - Compute efficiency validated
  - Performance within 95% of TensorRT
  - 60% compute savings validated

---

### **Phase 4: Production Readiness (Weeks 12-16)**
**Priority:** P2 - Important for MVP  
**Goal:** Production-ready system with documentation, testing, and validation

#### **Week 13: Vendor SDK Documentation**

**12.1 SDK Interface Documentation**
- **Location**: `coelanox-sdk/docs/`
- **Tasks**:
  - Document 8-trait interface
  - Create vendor onboarding guide
  - Create backend implementation guide
  - Add code examples
  - Add best practices
- **Success Criteria**:
  - SDK interface fully documented
  - Vendor onboarding guide complete
  - Implementation guide clear
  - Examples working
  - Best practices documented

**12.2 Scalar Reference Documentation**
- **Location**: `coelanox-core/src/scalar/`
- **Tasks**:
  - Document scalar backend as golden reference
  - Create correctness testing guide
  - Add operation specifications
  - Add numerical precision requirements
  - Create validation framework
- **Success Criteria**:
  - Scalar backend documented
  - Testing guide complete
  - Operation specs clear
  - Precision requirements defined
  - Validation framework ready

#### **Week 13: Performance Benchmarking**

**13.1 Benchmark Suite**
- **Location**: `coelanox/benches/`
- **Tasks**:
  - Create comprehensive benchmark suite
  - Add TensorRT comparison benchmarks
  - Add PCIe utilization benchmarks
  - Add compute efficiency benchmarks
  - Add cold start benchmarks
- **Success Criteria**:
  - Benchmark suite complete
  - TensorRT comparisons working
  - All metrics measured
  - Benchmarks reproducible

**13.2 Performance Validation**
- **Tasks**:
  - Run benchmarks on target hardware
  - Validate performance claims
  - Measure real-world improvements
  - Document performance characteristics
  - Create performance report
- **Success Criteria**:
  - Benchmarks run successfully
  - Performance claims validated
  - Real-world improvements measured
  - Performance documented
  - Report generated

#### **Week 14: Testing & Quality Assurance**

**14.1 Comprehensive Testing**
- **Tasks**:
  - Add unit tests for all new features
  - Add integration tests for full pipeline
  - Add performance tests
  - Add stress tests
  - Add concurrency tests
- **Success Criteria**:
  - Test coverage >90%
  - All tests passing
  - Performance tests validated
  - Stress tests passing
  - Concurrency tests passing

**14.2 Quality Assurance**
- **Tasks**:
  - Code review all changes
  - Security audit
  - Performance audit
  - Documentation review
  - Final validation
- **Success Criteria**:
  - Code reviewed
  - Security validated
  - Performance validated
  - Documentation complete
  - Ready for production

#### **Week 15: Documentation & Examples**

**15.1 User Documentation**
- **Tasks**:
  - Create user guide
  - Create API documentation
  - Create deployment guide
  - Create troubleshooting guide
  - Add examples
- **Success Criteria**:
  - User guide complete
  - API documentation complete
  - Deployment guide clear
  - Troubleshooting guide helpful
  - Examples working

**15.2 Developer Documentation**
- **Tasks**:
  - Create architecture documentation
  - Create contribution guide
  - Create development setup guide
  - Create testing guide
  - Add developer examples
- **Success Criteria**:
  - Architecture documented
  - Contribution guide clear
  - Development setup working
  - Testing guide complete
  - Developer examples working

#### **Week 16: Final Validation & Release**

**16.1 Final Testing**
- **Tasks**:
  - Run full test suite
  - Run performance benchmarks
  - Run security audits
  - Run compatibility tests
  - Validate all success criteria
- **Success Criteria**:
  - All tests passing
  - Performance validated
  - Security validated
  - Compatibility confirmed
  - All success criteria met

**16.2 Release Preparation**
- **Tasks**:
  - Create release notes
  - Tag release version
  - Create deployment packages
  - Prepare marketing materials
  - Announce release
- **Success Criteria**:
  - Release notes complete
  - Version tagged
  - Packages created
  - Materials prepared
  - Release announced

---

## 📋 **Detailed Task Breakdown**

### **Critical Fixes (Phase 1)**

#### **Security Hash Implementation & Three-Command Architecture**
```rust
// coelanox-packager/src/lib.rs
use sha2::{Sha256, Digest};

fn calculate_security_hash(container_data: &[u8]) -> [u8; 32] {
    // Hash entire container (header + manifest + code + weights + IR)
    // Header.security_hash set to [0; 32] during calculation
    let mut hasher = Sha256::new();
    hasher.update(container_data);
    hasher.finalize().into()
}

// coelanox-core/src/format.rs (shared verification logic)
pub fn verify_security_hash(container_path: &Path) -> CoelanoxResult<()> {
    // Read container from cold storage
    let container_data = std::fs::read(container_path)?;
    
    // Calculate hash of entire file
    let calculated_hash = calculate_security_hash(&container_data);
    
    // Extract stored hash from header
    let stored_hash = extract_hash_from_header(&container_data)?;
    
    // Compare hashes
    if calculated_hash != stored_hash {
        return Err(CoelanoxError::Security("Container hash mismatch - file may have been tampered with".to_string()));
    }
    
    Ok(())
}

// coelanox-cli/src/main.rs
// Three commands:
// 1. coelanox package model.onnx --h100 --output model.cnox
//    → Creates container with hash (package time)
//
// 2. coelanox verify model.cnox
//    → Standalone verification (pre-deployment check)
//    → Reads from cold storage, verifies hash
//    → Does NOT load into memory tier
//
// 3. coelanox run model.cnox
//    → Verifies hash (pre-deployment)
//    → If verified: Loads into memory tier (NVMe + RAM)
//    → Executes from memory tier (trusted, no re-verification)
```

**Security Model:**
- **Cold Storage**: Container file on disk (unverified, untrusted)
- **Pre-Deployment Verification**: Hash check before memory tier loading
- **Memory Tier Deployment**: Load into NVMe (weights, persistent "cheap RAM") + RAM (code, hot)
- **Runtime**: Execute from memory tier (trusted, already verified)
- **Defense in Depth**: `verify` command (standalone) + `run` command (also verifies)

#### **Backend Manager & AOT Binary Selection**
```rust
// coelanox-runtime/src/backend_manager.rs
pub struct BackendManager {
    backends: HashMap<String, Box<dyn Backend>>,
    fallback: FallbackBackend,
    device_id_mapping: HashMap<usize, String>, // Device ID → Backend name
}

impl BackendManager {
    pub fn discover_backends(&mut self) -> CoelanoxResult<()> {
        // Discover vendor backends from container (AOT compiled)
        // Load dynamic libraries
        // Register backends from container metadata
    }
    
    pub fn execute(&self, container: &CoelanoxContainer, device_id: usize) -> CoelanoxResult<()> {
        // Route to appropriate backend based on device_id
        // Backend selected from container (pre-compiled at package time)
        // Fallback to scalar if needed
    }
}

// coelanox-packager/src/lib.rs
// AOT Binary Selection (Package Time)
pub enum BinarySelection {
    Slim { 
        primary: HardwareTarget,  // --h100, --a100, etc.
        additional: Vec<HardwareTarget>,  // --backend-add amd,intel
    },
    Fat,  // --fat (all backends)
    Auto,  // --auto (auto-detect + auto-add)
}

impl CoelanoxPackager {
    pub fn package_with_selection(
        &self,
        model: &Path,
        selection: BinarySelection,
        output: &Path
    ) -> CoelanoxResult<()> {
        // Determine backends to include based on selection
        // Generate code for selected backends (AOT compilation)
        // Store backend metadata in container
        // Map device IDs: primary = 0, additional = 1, 2, 3...
    }
}
```

**Package Time Selection:**
- **Slim**: `--h100` (primary) + `--backend-add amd,intel` (additional)
- **Fat**: `--fat` (all backends)
- **Auto**: `--auto` (auto-detect binary + auto-add backends)

**Container Contents:**
- Pre-compiled code for selected backends
- Backend metadata in manifest
- Device ID mapping (primary = 0, added = 1, 2, 3...)

**Runtime Behavior:**
- Uses backends from container (pre-compiled at package time)
- `--device-id=0` selects backend from container
- No backend selection needed (AOT compiled)

#### **Scalar Backend Structure**
```
coelanox-core/src/scalar/
├── mod.rs                 # Module exports
├── arithmetic.rs          # Add, Sub, Mul, Div
├── activations.rs         # ReLU, Sigmoid, Tanh, etc.
├── pooling.rs            # MaxPool, AvgPool
├── normalization.rs      # BatchNorm, LayerNorm
└── tensor_ops.rs         # Reshape, Transpose, etc.

coelanox-runtime/src/fallback/
├── mod.rs                 # Module exports
├── fallback_backend.rs    # FallbackBackend struct
└── scalar_executor.rs     # Uses coelanox-core scalar
```

### **AOT Binary Selection Architecture (Phase 1)**

#### **Fat vs Slim Container Selection (Package Time)**
**Location**: `coelanox-packager/src/lib.rs`

**Design Philosophy:**
- **AOT Compilation**: Backend selection happens at package time, not runtime
- **Container Pre-compilation**: Container includes pre-compiled code for selected backends
- **User Choice**: Production (slim) vs Home lab (fat)

**Selection Modes:**

**1. Slim Container (Production):**
```bash
# Two-stage selection
coelanox-packager model.onnx --h100 --backend-add amd,intel --output model.cnox
# Stage 1: --h100 (primary backend, device ID 0)
# Stage 2: --backend-add amd,intel (additional backends, device ID 1, 2)
# Container: Pre-compiled with H100 + AMD + Intel backends
# Size: Medium (~300MB, selected backends only)
```

**2. Fat Container (Home Lab):**
```bash
# Single-stage selection
coelanox-packager model.onnx --fat --output model.cnox
# All backends included (NVIDIA, AMD, Intel, Qualcomm, CPU, etc.)
# Container: Pre-compiled with all backends
# Size: Large (~500MB, all backends)
```

**3. Auto Selection:**
```bash
# Auto-detect + auto-add
coelanox-packager model.onnx --auto --output model.cnox
# Stage 1: Auto-detect hardware → Select binary (e.g., --h100)
# Stage 2: Auto-detect available backends → Add (e.g., --backend-add amd)
# Container: Pre-compiled with detected backends
```

**Container Metadata:**
- Backend list (which backends are included)
- Primary backend (device ID 0)
- Device ID mapping (ID → backend name)
- Optimization parameters (from CSV for each backend)

**Runtime Behavior:**
- Container uses pre-compiled backends (no selection needed)
- `--device-id=0` selects backend from container
- All backends available are those included at package time
- No dynamic backend loading at runtime (AOT compiled)

**Use Cases:**
- **Production**: Slim container (`--h100`) - Hardware-specific, smaller size
- **Home Lab**: Fat container (`--fat`) - All backends, don't care about space
- **Hybrid**: Slim + additions (`--h100 --backend-add amd,intel`) - Optimized primary + flexibility

### **Backend SDK Architecture (Phase 1)**

#### **Backend SDK Overview**

The Backend SDK provides a trait-based interface for hardware vendors to implement code generation at **package time** (AOT compilation). Vendors implement the `BackendTranslator` trait to generate hardware-specific binary machine code from Universal IR operations.

**Architecture Flow:**
```
Universal IR (OpType operations)
    ↓
[Backend Translator] ← Vendor Implementation
    ├─ translate_operation() → Binary machine code (Vec<u8>)
    ├─ optimize_code() → Hardware-optimized code
    ├─ apply_optimization_hints() → Apply hints (SIMD, parallelization, etc.)
    └─ Returns optimized binary machine code
    ↓
Binary Machine Code (Vec<u8>)
    ↓
[Packager] ← Your Code
    ├─ Stitches operation code together
    ├─ Adds prologue/epilogue (stack allocation)
    └─ Packages into .cnox container
    ↓
.cnox Container (pre-compiled, ready to execute)
```

**Key Point**: This is **package-time code generation** (AOT), not runtime execution. The runtime executes the pre-compiled machine code directly - no backend interface needed at runtime.

#### **Core Trait: BackendTranslator**

**Location**: `coelanox-packager/src/generators/backend_translators/backend_translator.rs`

**Single Trait with 8 Required Methods** - All hardware vendors implement this:

```rust
pub trait BackendTranslator: Send + Sync {
    // 1. Translate operation to hardware-specific binary machine code
    // Returns Vec<u8> containing actual machine code bytes (not assembly strings)
    fn translate_operation(
        &self, 
        operation: &OpType, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<Vec<u8>>;
    
    // 2. Optimize code for specific hardware
    // Takes binary machine code → returns optimized binary machine code
    fn optimize_code(
        &self, 
        code: &[u8], 
        target: HardwareTarget
    ) -> CoelanoxResult<Vec<u8>>;
    
    // 3. Get supported hardware targets
    fn supported_hardware(&self) -> Vec<HardwareTarget>;
    
    // 4. Get translator name
    fn name(&self) -> &str;
    
    // 5. Get translator version
    fn version(&self) -> &str;
    
    // 6. Get performance characteristics for operation
    fn get_performance_estimate(
        &self, 
        operation: &OpType, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<CoelanoxPerformanceEstimate>;
    
    // 7. Apply optimization hints to code
    // Takes binary machine code → returns optimized binary machine code
    fn apply_optimization_hints(
        &self, 
        code: &[u8], 
        hints: &[CoelanoxOptimizationHint]
    ) -> CoelanoxResult<Vec<u8>>;
    
    // 8. Validate operation parameters
    fn validate_parameters(
        &self, 
        operation: &OpType, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<()>;
    
    // Optional: Check if translator supports native COELANOX operations
    fn supports_native_coelanox(&self) -> bool {
        false // Default to translated operations
    }
}
```

#### **What Vendors Must Implement**

**1. Binary Machine Code Generation:**
- `translate_operation()`: Convert `OpType` (Add, MatMul, Convolution, etc.) → hardware-specific binary machine code
- Returns `Vec<u8>` (actual machine code bytes, not assembly strings)
- For AOT compilation: Code is generated at package time, not runtime

**2. Hardware Optimization:**
- `optimize_code()`: Optimize generated code for specific hardware (SIMD, tensor cores, etc.)
- `apply_optimization_hints()`: Apply hints (vectorizable, parallelizable, memory-efficient, etc.)

**3. Performance Estimation:**
- `get_performance_estimate()`: Estimate execution time, memory usage, power consumption
- Used by packager for optimization decisions

**4. Parameter Validation:**
- `validate_parameters()`: Validate operation parameters (shapes, attributes, etc.)
- Ensures operation is valid for hardware

#### **What SDK Provides (Your Code)**

**1. Trait Interface:**
- Defines contract (`BackendTranslator` trait)
- Vendors implement it (you don't maintain hardware-specific codegen)

**2. Backend Translator Manager:**
- Manages registered translators
- Routes operations to appropriate translators
- Handles translator discovery

**3. Operation Parameters:**
- `CoelanoxAiOperationParams`: Input/output shapes, attributes, optimization hints
- `CoelanoxOptimizationHint`: Hints for optimization (vectorizable, parallelizable, etc.)

**4. Code Stitching:**
- Packager stitches operation code together
- Adds prologue/epilogue (stack allocation, register management)
- Creates complete executable function

#### **Package-Time vs Runtime**

**Package Time (AOT Compilation):**
- Backend translators generate binary machine code
- Code is pre-compiled into `.cnox` container
- No backend interface needed at runtime

**Runtime:**
- Executes pre-compiled machine code directly
- No backend interface - just direct execution
- Code is already optimized for hardware

#### **Example: NVIDIA Backend**

```rust
pub struct NvidiaBackendTranslator {
    cuda_context: CudaContext,
}

impl BackendTranslator for NvidiaBackendTranslator {
    fn translate_operation(&self, operation: &OpType, params: &CoelanoxAiOperationParams) -> CoelanoxResult<Vec<u8>> {
        match operation {
            OpType::MatMul => {
                // Generate CUDA PTX/SASS code for matrix multiplication
                // Compile to binary machine code
                // Return Vec<u8> with machine code
            }
            OpType::Convolution => {
                // Generate CUDA code for convolution
                // Use cuDNN or custom kernels
                // Return Vec<u8> with machine code
            }
            // ... other operations
        }
    }
    
    fn optimize_code(&self, code: &[u8], target: HardwareTarget) -> CoelanoxResult<Vec<u8>> {
        // Optimize for specific GPU (H100, A100, etc.)
        // Apply hardware-specific optimizations
    }
    
    // ... implement other methods
}
```

#### **Backend Manager Integration**

**Package Time:**
- Backend manager discovers vendor backends (dynamic library loading)
- Packager uses backend translator to generate code
- Code is pre-compiled into container

**Runtime:**
- Container already contains pre-compiled code
- Runtime executes code directly (no backend interface)
- Backend manager only needed for vendor backend discovery at package time

### **Frontend SDK Architecture (Phase 1)**

#### **Frontend Translator SDK Overview**

The Frontend SDK provides a trait-based interface for converting various model formats (ONNX, PyTorch, TensorFlow, etc.) into the Universal IR. Vendors and the community implement translators; you maintain the trait interface and plugin manager.

**Architecture Flow:**
```
Frontend Binary (ONNX/PyTorch/TensorFlow)
    ↓
[Frontend Translator] ← Vendor/Community Implementation
    ├─ Parses frontend binary format (protobuf, pickle, etc.)
    ├─ Extracts weights → populates WeightTensor.data in IR
    ├─ Builds graph structure → converts to OpType operations
    ├─ Extracts shapes → static shapes from model, infers dynamic
    └─ Returns UniversalIR (Rust struct)
    ↓
UniversalIR (Rust struct)
    ↓
[Packager] ← Your Code
    ├─ Validates IR: ir.validate()
    ├─ Optimizes IR for hardware
    ├─ Generates machine code
    ├─ Extracts weights from IR (or uses extract_weights())
    ├─ Compresses weights
    └─ Packages into .cnox container
```

#### **Core Trait: TranslatorPlugin**

**Location**: `coelanox-core/src/plugins.rs`

**Single Trait Interface** - All frontend translators implement this:

```rust
pub trait TranslatorPlugin: Send + Sync + 'static {
    // REQUIRED METHODS (must implement):
    
    /// Parse frontend binary → produce UniversalIR
    /// - Parses frontend format (ONNX protobuf, PyTorch pickle, etc.)
    /// - Extracts weights → populates WeightTensor.data in IR
    /// - Builds graph → converts operations to OpType, connects via TensorId
    /// - Extracts shapes → static shapes from model, infers dynamic
    /// - Returns complete UniversalIR with weights populated
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>;
    
    /// Extract weights from frontend binary (for compression)
    /// - Flattens all weights into Vec<f32>
    /// - Used by packager for compression
    /// - Alternative: Extract from ir.weight_tensors (if weights in IR)
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>;
    
    /// Which formats this translator supports
    fn supported_formats(&self) -> Vec<ModelFormat>;
    
    /// Metadata
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn author(&self) -> &str;
    
    // OPTIONAL METHODS (have default implementations):
    
    /// Translate from file path (handles sharded models)
    fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR> {
        // Default: read file → call translate()
    }
    
    /// Translate with context (custom shapes, model type)
    fn translate_with_context(&self, model_data: &[u8], model_type: &str, 
                              custom_input_shapes: Option<&str>, 
                              custom_output_shapes: Option<&str>) -> CoelanoxResult<UniversalIR> {
        // Default: call translate()
    }
    
    /// Plugin capabilities (sharded models, custom shapes, etc.)
    fn capabilities(&self) -> PluginCapabilities { ... }
}
```

#### **Plugin Manager**

**Location**: `coelanox-packager/src/plugins/mod.rs`

**Manages Registered Translators**:

```rust
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn TranslatorPlugin>>,
    config: PluginConfig,
}

impl PluginManager {
    /// Register a translator plugin
    pub fn register_plugin(&mut self, name: String, plugin: Box<dyn TranslatorPlugin>);
    
    /// Find translator for model format
    pub fn find_plugin(&self, format: ModelFormat) -> Option<&dyn TranslatorPlugin>;
    
    /// Load default plugins (ONNX, BERT, etc.)
    pub fn load_default_plugins(&mut self) -> CoelanoxResult<()>;
}
```

#### **What Translators Must Do**

**1. Parse Binary Format:**
- ONNX Translator: Parses ONNX protobuf (uses `prost` or `onnx` crate)
- PyTorch Translator: Parses pickle format (uses `pytorch` crate or custom)
- TensorFlow Translator: Parses SavedModel (protobuf + checkpoint files)
- Each translator contains its own binary parser (vendor/community responsibility)

**2. Populate UniversalIR:**
- **Weight Tensors**: Extract weights from frontend format → populate `WeightTensor.data` in IR
- **Graph Structure**: Convert frontend operations → `OpType` operations, connect via `TensorId`
- **Shapes**: Extract static shapes from model, infer dynamic shapes
- **Metadata**: Populate `ModelMetadata` (name, input/output shapes, data types)

**3. Return Complete IR:**
- IR must be valid (can call `ir.validate()` to check)
- Weights should be in `WeightTensor.data` (packager can extract from IR)
- Graph must be well-formed (nodes connected via TensorIds)

#### **What SDK Provides (Your Code)**

**1. Trait Interface:**
- Defines contract (`TranslatorPlugin` trait)
- Vendors implement it (you don't maintain frontend parsers)

**2. Plugin Manager:**
- Manages registered translators
- Finds translators by format
- Handles plugin discovery

**3. UniversalIR:**
- Rust struct (not a trait)
- Serialized with serde/bincode (no custom parser needed)
- Contains graph, weights, shapes, metadata

**4. Validation:**
- `ir.validate()` checks IR structure
- Packager validates after translation

**5. Utilities:**
- Format detection (magic bytes)
- Weight extraction from IR
- IR serialization/deserialization

#### **No Binary Parser Needed for UniversalIR**

**Key Point**: UniversalIR is a Rust struct with `Serialize`/`Deserialize` traits. You don't parse it - you serialize/deserialize it:

- **Serialization**: `bincode::serialize(&ir)` → Rust struct → binary bytes (for container storage)
- **Deserialization**: `bincode::deserialize(&bytes)` → binary bytes → Rust struct (at runtime)

This is **not parsing** - serde/bincode handles the binary format automatically. The translator produces a Rust struct; the packager works with the struct directly.

#### **Complete Flow Example**

```rust
// 1. Packager detects format
let format = detect_format(&magic_bytes)?; // ModelFormat::Onnx

// 2. Plugin manager finds translator
let translator = plugin_manager.find_plugin(format)?; // OnnxTranslator

// 3. Translator parses ONNX → produces UniversalIR
let ir = translator.translate_from_path(model_path)?;
// OnnxTranslator internally:
//   - Parses ONNX protobuf (uses prost/onnx crate)
//   - Extracts weights → populates WeightTensor.data
//   - Builds graph → OpType operations
//   - Extracts shapes → static + dynamic
//   - Returns UniversalIR (Rust struct)

// 4. Packager validates IR
ir.validate()?;

// 5. Packager extracts weights (from IR or separate call)
let weights = extract_weights_from_ir(&ir)?; // Or: translator.extract_weights()

// 6. Packager compresses weights
let compressed = compress_weights(&weights)?;

// 7. Packager serializes IR (serde/bincode, not parsing)
let ir_bytes = bincode::serialize(&ir)?;

// 8. Packager creates container
let container = CoelanoxContainer::new(manifest, code, compressed, ir_bytes);
```

#### **Responsibilities Summary**

**Translator (Vendor/Community):**
- Parse frontend binary format (ONNX protobuf, PyTorch pickle, etc.)
- Extract weights → populate `WeightTensor.data` in IR
- Build graph structure → convert to `OpType` operations
- Extract shapes and metadata
- Return complete `UniversalIR`

**SDK (Your Code):**
- Define trait interface (`TranslatorPlugin`)
- Manage plugins (`PluginManager`)
- Validate IR structure
- Serialize/deserialize IR (serde/bincode)
- Extract weights from IR
- Provide utilities

**No Second Parser:**
- UniversalIR is a Rust struct
- Serialized with serde/bincode (automatic)
- No custom binary parser needed

### **Core Features (Phase 2)**

#### **Static Load & Lock Architecture (Formal Specification)**

**Process Name**: "Load, Lock, Launch"  
**Architecture**: Custom I/O + Hardware MMU/MPU Protection  
**OS Independence**: Direct register access (works on bare metal, firmware, embedded, RTOS)

**Three-Phase Security Model:**

1. **Phase I: Load (Custom I/O)**
   - Parse container header to identify section sizes
   - Allocate physical memory regions (Code, Weights, Scratchpad)
   - Decompress Code section → SRAM Region A (executable)
   - Decompress Weights section → SRAM Region B (read-only target)
   - Zero-init Scratchpad → SRAM Region C (read-write)
   - **System State**: All RAM is RWX (open state, only during loading)

2. **Phase II: Lock (Hardware MMU/MPU Configuration)**
   - Configure hardware protection registers:
     - **Runtime Boot Code**: RX (Read + Execute)
     - **Container Kernels**: RX (Read + Execute)
     - **Model Weights**: RO (Read Only, No-Execute) - **Hardware Protected**
     - **Scratchpad/Stack**: RW (Read + Write, No-Execute)
   - **Platform-Specific**:
     - **ARM**: MPU_RBAR/MPU_RASR registers (Cortex-M, Cortex-R, Cortex-A)
     - **x86**: Page Tables (CR3 register, standard x86-64)
     - **RISC-V**: PMP (Physical Memory Protection, RISC-V standard)
   - **Enable MMU/MPU**: Hardware gate closes, memory map becomes immutable
   - **Result**: Hardware-enforced protection (crash-safe on violation)

3. **Phase III: Launch (Execution)**
   - Drop privilege (optional): Switch from Supervisor to User Mode
   - Jump to container kernels entry point
   - **Inference Loop**:
     - Kernels read weights (hardware-protected RO) → Success
     - Kernels write intermediates (scratchpad RW) → Success
     - **Security**: Write attempt to weights → Hardware exception (crash-safe)

**Key Benefits:**
- **Hardware-Level Protection**: MMU/MPU provides true hardware enforcement
- **OS Independence**: Direct register access (no OS dependencies)
- **Crash-Safe Security**: Hardware exception on violation (cannot be bypassed)
- **Universal Deployment**: Works on bare metal, firmware, embedded, RTOS, full OS
- **Clear Security Model**: Load → Lock → Launch (three distinct phases)

**Memory Region Layout:**
```
Region 0: Runtime Boot Code    → RX (Read + Execute)
Region 1: Container Kernels    → RX (Read + Execute)
Region 2: Model Weights         → RO (Read Only, No-Execute) [HARDWARE PROTECTED]
Region 3: Scratchpad/Stack     → RW (Read + Write, No-Execute)
```

**InferenceContext Structure:**
```rust
#[repr(C)]
struct InferenceContext {
    weights_base: *const u8,  // Hardware-protected RO region (Phase II: Lock)
    backing_base: *mut u8,    // Scratchpad region (RW, per-instance, mode-dependent)
    code_base: *const u8,     // Hardware-protected RX region (Phase II: Lock)
}
```

#### **Max Shape Reader**
```rust
// coelanox-core/src/optimization/max_shape_reader.rs
pub struct MaxShapeReader;

impl MaxShapeReader {
    /// Read max shapes from Universal IR (shapes already in IR from translator)
    /// - Static shapes: Fixed dimensions (already in IR)
    /// - Dynamic shapes: Max bounds from model metadata (already in IR)
    /// - No calculation needed: Just extract from IR
    pub fn read_max_shapes(ir: &UniversalIR) -> HashMap<TensorId, Shape> {
        let mut max_shapes = HashMap::new();
        
        // Read weight tensor shapes (static, already in IR)
        for (idx, tensor) in ir.weight_tensors.iter().enumerate() {
            let tensor_id = TensorId::weight(idx);
            max_shapes.insert(tensor_id, tensor.shape.clone());
        }
        
        // Read intermediate tensor shapes (static or dynamic, already in IR)
        for (idx, tensor) in ir.intermediate_tensors.iter().enumerate() {
            let tensor_id = TensorId::intermediate(idx);
            max_shapes.insert(tensor_id, tensor.shape.clone());
        }
        
        // For dynamic shapes, check model metadata for max bounds
        // (Translator already extracted max bounds from model)
        
        max_shapes
    }
}
```

**Key Point**: Shapes are **already in the model** → translator extracts them → IR has shapes → reader just extracts from IR (no calculation needed).

#### **Static Load & Lock Architecture (Hardware-Enforced Protection)**

**Architecture Decision**: "Static Load & Lock" - Custom I/O + Hardware MMU/MPU Protection

**Formal Process Specification: "Load, Lock, Launch"**

### **Phase I: The Loader (Custom I/O)**

*The Runtime reads the `.cnox` container from storage (NVMe/Flash) and decompresses it into physical SRAM.*

**Platform-Specific Storage I/O:**
- **ARM**: Direct ARM storage controller register access (via `arm_io.rs`)
- **x86**: Direct x86 storage controller register access (via `x86_io.rs`)
- **RISC-V**: Direct RISC-V storage controller register access (via `riscv_io.rs`)
- **Same Platforms**: Uses same compile-time targets as MMU/MPU firewall (`#[cfg(target_arch)]`)

1. **Parse Header:** Read the `.cnox` header to identify the size of Kernels, Weights, and Memory requirements.
   - **Storage Access**: Direct hardware register access (no OS file system)
   - **Platform-Specific**: ARM/x86/RISC-V storage controller implementations
2. **Allocate Physical Regions:** Calculate physical start addresses for each section (e.g., Weights start at `0x2000_0000`).
   - **Physical Memory**: Direct physical address allocation (no virtual memory)
   - **Platform-Specific**: ARM/x86/RISC-V memory controller implementations
3. **Decompress Code (Kernels):**
   - **Source:** Compressed Code section in `.cnox`.
   - **Dest:** SRAM Region A (e.g., `0x1000_0000`).
   - *Action:* `custom_io_read_and_decompress(source, dest_A)`
   - **DMA Transfer**: Platform-specific DMA configuration (zero-copy)
4. **Decompress Weights:**
   - **Source:** Compressed Weights section in `.cnox`.
   - **Dest:** SRAM Region B (e.g., `0x2000_0000`).
   - *Action:* `custom_io_read_and_decompress(source, dest_B)`
   - **DMA Transfer**: Platform-specific DMA configuration (zero-copy)
5. **Prepare Scratchpad:**
   - **Dest:** SRAM Region C (e.g., `0x3000_0000`).
   - *Action:* Zero-init this memory for intermediate calculations.

**System State:** All RAM is **Read/Write/Execute (RWX)**. This is the only time the system is "open" to allow loading.

---

### **Phase II: The Lockdown (Hardware Enforcement)**

*The critical security step. The Runtime configures the hardware protection registers before any AI logic runs.*

**Action:** Write to CPU Configuration Registers (MPU_RBAR / MPU_RASR on ARM, or Page Tables on x86/RISC-V).

| Region | Physical Address Range | Content | **Hardware Permission Set** |
| --- | --- | --- | --- |
| **0** | `0x0000` - `0x0FFF` | **Runtime Boot Code** | **RX** (Read + Execute) |
| **1** | `0x1000_0000` - `0x1FFF_FFFF` | **Container Kernels** | **RX** (Read + Execute) |
| **2** | `0x2000_0000` - `0x2FFF_FFFF` | **Model Weights** | **RO** (Read Only, No-Execute) |
| **3** | `0x3000_0000` - `0x3FFF_FFFF` | **Scratchpad/Stack** | **RW** (Read + Write, No-Execute) |

**Platform-Specific Implementation:**
- **ARM**: MPU_RBAR/MPU_RASR registers (Cortex-M, Cortex-R, Cortex-A)
- **x86**: Page Tables (CR3 register, standard x86-64)
- **RISC-V**: PMP (Physical Memory Protection, RISC-V standard)

**Final Step:** **ENABLE MPU/MMU.**

* *Effect:* The hardware "gate" slams shut. From this CPU cycle forward, the memory map is immutable.

---

### **Phase III: The Launch (Execution)**

*Control is handed over to the decompressed kernels.*

1. **Drop Privilege (Optional):** If required for security, switch CPU from Supervisor to User Mode.
2. **Jump:** The Runtime executes a jump instruction to the entry point of the **Container Kernels** (in Region 1).
3. **Inference Loop:**
   - The Kernels read input.
   - The Kernels read Weights (from Region 2). *Success.*
   - The Kernels write intermediates (to Region 3). *Success.*
   - **Security Test:** If a Kernel bug tries to write to Region 2 (Weights)? -> **HARDWARE EXCEPTION (Crash Safe).**

---

**InferenceContext for AOT Code**:
```rust
#[repr(C)]
struct InferenceContext {
    weights_base: *const u8,  // Hardware-protected RO region (Phase II: Lock)
    backing_base: *mut u8,    // Scratchpad region (RW, per-instance, mode-dependent)
    code_base: *const u8,     // Hardware-protected RX region (Phase II: Lock)
}
```

**AOT Code Generation**:
- Generated code accepts `InferenceContext` as parameter
- Access weights: `weights_base + offset` (from pre-calculated mapping, hardware-protected RO)
- Access intermediates: `backing_base + offset` (from pre-calculated mapping, mode-dependent)
- Access code: `code_base + offset` (from pre-calculated mapping, hardware-protected RX)
- Position-independent code (relative addressing via context pointers)
- Works for both modes (weights/code always hardware-protected, intermediates mode-dependent)

**Cross-Platform Considerations**:
- **64KB Alignment**: All sections must be 64KB-aligned (MMU/MPU granularity requirement)
- **Platform Abstraction**: Trait-based MMU/MPU configurator (ARM/x86/RISC-V implementations)
- **Mode Detection**: Runtime determines static RAM vs tiering mode
- **OS Independence**: Direct register access (works on bare metal, firmware, embedded, RTOS)

**Key Insight**: "Static Load & Lock" provides hardware-level protection (MMU/MPU) without OS dependencies. Weights are hardware-protected as read-only (crash-safe on write attempt), code is hardware-protected as read-execute, and intermediates are in a read-write scratchpad region. This architecture works universally (bare metal to full OS) and provides true hardware enforcement.

**Platform-Specific Implementation**: See Task 4.3.1 for detailed MMU/MPU firewall code implementation for ARM (MPU), x86 (Page Tables), and RISC-V (PMP) architectures.

#### **Chunk Streaming (OS Bypass)**
```rust
// coelanox-core/src/streaming/chunk_streaming.rs
pub struct ChunkStreamer {
    chunk_size: usize,
    ring_buffer: DoubleRingBuffer,
    bandwidth_utilization: f64,  // PCIe for GPU, memory for CPU
}

impl ChunkStreamer {
    // OS Bypass: Manual prefetching (no OS, direct control)
    pub fn stream_chunks(&mut self, data: &[u8]) -> CoelanoxResult<()> {
        // Manual chunk streaming (OS bypass)
        // Overlap compute + transfer (prefetch chunk N+1 while computing chunk N)
        // Bypass bandwidth bottlenecks (memory bandwidth for static RAM, storage for tiering)
    }
}

// Works with both static RAM and tiering:
// - Static RAM: Prefetch within RAM (memory bandwidth optimization)
// - Tiering: Prefetch from NVMe to RAM (storage bandwidth optimization)
// - Both: OS bypassed (manual control, predictable performance)
// - Performance: Enables faster models on lower hardware
```

#### **Memory Tiering**
```rust
// coelanox-core/src/memory/memory_tier.rs
pub enum MemoryTier {
    Nvme,  // Large, persistent, slow
    Ram,   // Small, fast, dynamic
}

pub struct MemoryTierManager {
    nvme_tier: NvmeTier,
    ram_tier: RamTier,
}

// Default: Tiering enabled (optimal for most use cases)
// Flag: --static-ram (for IoT/embedded/storage-constrained devices)
// Platform: ARM (MPU), x86 (Page Tables), RISC-V (PMP) - Direct register access (OS independent)
// Fallback: If tiering unavailable → all-RAM automatically
```

### **AMD Backend Reference (Phase 3)**

#### **ROCm Integration**
```rust
// coelanox-backend-amd/src/lib.rs (or vendor-provided)
use rocm_sys::*;

pub struct AmdBackend {
    hip_context: HipContext,
    chunk_streamer: ChunkStreamer,
}

impl Backend for AmdBackend {
    // Implement 8-trait interface
}
```

#### **SASS Codegen**
```rust
// coelanox-packager/src/generators/amd_codegen/mod.rs
pub struct AmdCodeGenerator {
    sass_emitter: SassEmitter,
    optimizer: AmdOptimizer,
}
```

---

## 🎯 **Success Metrics**

### **Phase 1 Success Criteria**
- ✅ Security hash implemented and verified
- ✅ Three-command architecture working (`package`, `verify`, `run`)
- ✅ Standalone verification command (`coelanox verify`) for pre-deployment checks
- ✅ Run command with pre-deployment verification (`coelanox run` verifies before loading)
- ✅ Shared verification logic (used by both `verify` and `run` commands)
- ✅ Hardware-specific backends removed from runtime
- ✅ Backend manager implemented (vendor backend discovery/loading)
- ✅ AOT binary selection implemented (slim: `--h100`, fat: `--fat`, auto: `--auto`)
- ✅ Two-stage selection working (primary + `--backend-add amd,intel`)
- ✅ Container pre-compiled with selected backends
- ✅ Device ID mapping stored in container (primary = 0, added = 1, 2, 3...)
- ✅ Single authoritative scalar backend in core
- ✅ Scalar fallback module created
- ✅ All critical issues fixed
- ✅ Test coverage >90%

### **Phase 2 Success Criteria**
- ✅ Max shape pre-allocation working
- ✅ Chunk streaming with PCIe >90% utilization
- ✅ NVMe-backed memory with O(1) lookups
- ✅ Memory tiering system functional (default: tiering enabled)
- ✅ `--static-ram` flag implemented for edge cases (IoT/embedded)
- ✅ Comprehensive documentation for memory tiering
- ✅ No allocation spikes

### **Phase 3 Success Criteria**
- ✅ AMD backend reference implementation (or vendor-provided)
- ✅ AMD backend loaded via backend manager
- ✅ ROCm/HIP integration working
- ✅ Pure SASS codegen implemented
- ✅ GPU utilization 110-130%
- ✅ Performance within 95% of TensorRT

### **Phase 4 Success Criteria**
- ✅ Vendor SDK documented
- ✅ Performance benchmarks validated
- ✅ Test coverage >90%
- ✅ Documentation complete
- ✅ Production-ready release

---

## 📊 **Risk Assessment**

### **Technical Risks**
1. **AMD Backend Complexity** - ROCm/HIP integration may be complex
   - **Mitigation**: Start with simple kernels, iterate; prefer vendor-provided backend
   - **Contingency**: Use vendor-provided backend if available; reference implementation optional

2. **Performance Validation** - May not meet 95% TensorRT target
   - **Mitigation**: Optimize incrementally, measure early
   - **Contingency**: Adjust targets based on real-world data

3. **Memory Tiering Complexity** - NVMe/RAM tiering may be complex
   - **Mitigation**: Start simple, add complexity incrementally
   - **Contingency**: Use simpler memory model if needed

### **Timeline Risks**
1. **Scope Creep** - Features may expand beyond MVP
   - **Mitigation**: Strict prioritization, defer non-essential features
   - **Contingency**: Extend timeline if necessary

2. **Integration Issues** - Features may not integrate smoothly
   - **Mitigation**: Test integration early, iterate
   - **Contingency**: Simplify integration if needed

### **Quality Risks**
1. **Test Coverage** - May not reach 90% coverage
   - **Mitigation**: Write tests alongside code
   - **Contingency**: Prioritize critical path tests

2. **Documentation** - May not be complete
   - **Mitigation**: Document as you go
   - **Contingency**: Prioritize user-facing documentation

---

## 🚀 **Next Steps**

### **Immediate Actions (Week 1)**
1. **Set up development environment** - Ensure all tools ready
2. **Create feature branches** - One per phase
3. **Set up CI/CD** - Automated testing and validation
4. **Begin Phase 1** - Start with security hash implementation

### **Week 1 Deliverables**
- ✅ Security hash implemented
- ✅ Hardware-specific backends removed
- ✅ Backend manager implemented
- ✅ Initial tests passing

### **Communication Plan**
- **Daily standups** - Progress updates
- **Weekly reviews** - Phase completion reviews
- **Bi-weekly demos** - Feature demonstrationsm
- **Monthly reports** - Overall progress reports

---

## 📚 **References**

### **Architecture Documents**
- `COELANOX_REFERENCE.md` - Architecture reference
- `coelanox-sdk/docs/ARCHITECTURE.md` - SDK architecture
- `README.md` - Project overview

### **Existing Roadmaps**
- `roadmaps/coelanox-v1-roadmap.md` - Original V1 roadmap
- `roadmaps/coelanox-constellation-roadmap.md` - Long-term roadmap

### **Code References**
- `coelanox-core/src/format.rs` - Container format
- `coelanox-runtime/src/backend_manager.rs` - Backend manager
- `coelanox-runtime/src/fallback/` - Scalar fallback backend
- `coelanox-packager/src/generators/` - Code generation

---

**Last Updated:** 2024  
**Status:** Active Development  
**Next Review:** End of Phase 1

