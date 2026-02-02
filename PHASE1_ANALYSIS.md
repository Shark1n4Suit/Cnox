# Phase 1 Analysis & Action Plan

## 📊 Current State Assessment

### ✅ What's Already in Place

1. **Project Structure**
   - ✅ Workspace with 4 main crates: `coelanox-core`, `coelanox-packager`, `coelanox-runtime`, `coelanox-cli`
   - ✅ `sha2` dependency already in `coelanox-core` (version 0.10)
   - ✅ Security hash field exists in `CoelanoxHeader` (`security_hash: [u8; 32]`)
   - ✅ CLI structure with `Package`, `Validate`, `Run` commands

2. **Container Format**
   - ✅ `CoelanoxHeader` structure with `security_hash` field (currently `[0; 32]`)
   - ✅ Binary format support with `write_binary` and `read_binary` methods
   - ✅ Header size calculation: 76 bytes (includes 32-byte hash)

3. **Backend Infrastructure**
   - ✅ Backend translator pattern exists in packager
   - ✅ CPU backend translator implemented
   - ⚠️ Hardware-specific backends (GPU/NPU/TPU) appear to be disabled/commented out in packager
   - ⚠️ Runtime backends directory exists but appears empty

4. **Scalar Operations**
   - ✅ Runtime has scalar fallback operations (in `runtime.rs`)
   - ✅ Packager has SIMD codegen (which is correct - SIMD is for codegen, scalar is for fallback)

### ❌ What Needs to Be Done

#### Week 1 Tasks

**1.1 Security Hash Implementation**
- ❌ SHA-256 hash calculation during packaging
- ❌ Hash verification logic in core
- ❌ `verify` command in CLI (currently only `Validate` exists)
- ❌ Shared verification function for `verify` and `run` commands
- ❌ Hash mismatch error handling

**1.2 Remove Hardware-Specific Backends**
- ✅ **FILES DON'T EXIST** - No files to delete
- ❌ Remove references from `coelanox-runtime/src/backends.rs`:
  - Remove `pub mod gpu_backend;`
  - Remove `pub mod npu_backend;`
  - Remove `pub mod tpu_backend;`
  - Remove `pub use gpu_backend::*;`
  - Remove `pub use npu_backend::*;`
  - Remove `pub use tpu_backend::*;`
- ❌ Keep only `cpu_backend` (if it exists) or remove all if none exist
- ❌ Clean up unused dependencies

**1.3 Backend Manager & AOT Selection**
- ❌ Backend manager module doesn't exist
- ❌ No dynamic library loading system
- ❌ No AOT binary selection (`--h100`, `--fat`, `--auto`, `--backend-add`)
- ❌ No device ID mapping in container

#### Week 2 Tasks

**2.1 Delete Packager Scalar Codegen**
- ✅ **NO ACTION NEEDED** - Packager uses SSE scalar instructions (movss, addss, etc.)
- ✅ SSE scalar is correct for codegen (generating x86-64 assembly)
- ✅ This is different from runtime scalar fallback (pure Rust)
- ⚠️ **Clarification**: "Scalar codegen" in roadmap refers to removing scalar fallback paths in packager, but packager doesn't have scalar fallback - it generates SSE scalar assembly which is correct
- ✅ **Task may be unnecessary** - verify with roadmap author if this task is still needed

**2.2 Create Scalar Backend in Core**
- ❌ `coelanox-core/src/scalar/` module doesn't exist
- ❌ Need to extract scalar operations from runtime
- ❌ Organize by operation type

**2.3 Create Fallback Module**
- ❌ `coelanox-runtime/src/fallback/` module doesn't exist
- ❌ Need to create `FallbackBackend` struct

## 🎯 Detailed Implementation Plan

### Task 1.1: Security Hash & Three-Command Architecture

**Files to Modify:**
1. `coelanox-packager/src/lib.rs` - Add hash calculation during packaging
2. `coelanox-core/src/format.rs` - Add shared verification function
3. `coelanox-cli/src/main.rs` - Add `verify` command

**Implementation Steps:**
1. Calculate SHA-256 hash of entire container (after compression, before writing)
2. Set `security_hash` to `[0; 32]` during calculation
3. Write hash to header after calculation
4. Create `verify_security_hash()` function in `format.rs`
5. Add `verify` command to CLI
6. Update `run` command to verify before loading

**Key Considerations:**
- Hash must include entire container (header + manifest + code + weights + IR)
- Hash calculated after compression (hash final container as stored)
- Verification happens at pre-deployment (before memory tier loading)
- Shared logic between `verify` and `run` commands

### Task 1.2: Remove Hardware-Specific Backends

**Files to Check/Modify:**
1. `coelanox-runtime/src/backends.rs` - Check what's exported
2. `coelanox-runtime/src/backends/` - Check if files exist
3. `coelanox-packager/src/generators/backend_translators/` - Already disabled/commented

**Implementation Steps:**
1. Verify which backend files exist
2. Delete `gpu_backend.rs`, `npu_backend.rs`, `tpu_backend.rs` if they exist
3. Remove exports from `backends.rs`
4. Clean up unused dependencies
5. Update any references

### Task 1.3: Backend Manager & AOT Selection

**New Files to Create:**
1. `coelanox-runtime/src/backend_manager.rs` - Backend manager implementation
2. `coelanox-runtime/src/fallback/mod.rs` - Fallback module (Week 2, but structure needed)

**Files to Modify:**
1. `coelanox-packager/src/lib.rs` - Add AOT binary selection logic
2. `coelanox-cli/src/main.rs` - Add selection flags (`--h100`, `--fat`, `--auto`, `--backend-add`)
3. `coelanox-core/src/format.rs` - Add backend metadata to manifest

**Implementation Steps:**
1. Create `BackendManager` struct with:
   - Backend registry (HashMap<String, Box<dyn Backend>>)
   - Fallback backend
   - Device ID mapping
2. Implement dynamic library loading (.so/.dylib/.dll)
3. Implement backend discovery
4. Implement backend routing
5. Add AOT selection to packager:
   - Parse selection flags
   - Determine backends to include
   - Store metadata in container
   - Map device IDs

### Task 2.1: Delete Packager Scalar Codegen

**Files to Check:**
1. `coelanox-packager/src/generators/cpu_codegen/operations/` - Check for scalar-specific code
2. Look for files with "scalar" in name or scalar fallback paths

**Implementation Steps:**
1. Identify all scalar codegen files
2. Remove scalar-specific codegen (keep SIMD)
3. Update codegen to reference core scalar operations
4. Remove scalar documentation

### Task 2.2: Create Scalar Backend in Core

**New Files to Create:**
1. `coelanox-core/src/scalar/mod.rs`
2. `coelanox-core/src/scalar/arithmetic.rs`
3. `coelanox-core/src/scalar/activations.rs`
4. `coelanox-core/src/scalar/pooling.rs`
5. `coelanox-core/src/scalar/normalization.rs`
6. `coelanox-core/src/scalar/tensor_ops.rs`

**Files to Modify:**
1. `coelanox-core/src/lib.rs` - Export scalar module
2. `coelanox-runtime/src/runtime.rs` - Extract scalar operations

**Implementation Steps:**
1. Extract scalar operations from runtime
2. Organize by operation type
3. Implement pure Rust (no SIMD, no SSE)
4. Add comprehensive documentation
5. Add unit tests

### Task 2.3: Create Fallback Module

**New Files to Create:**
1. `coelanox-runtime/src/fallback/mod.rs`
2. `coelanox-runtime/src/fallback/fallback_backend.rs`
3. `coelanox-runtime/src/fallback/scalar_executor.rs`

**Implementation Steps:**
1. Create `FallbackBackend` struct
2. Use `coelanox-core` scalar operations
3. Implement backend traits
4. Integrate with backend manager

## ✅ Codebase Investigation Complete

### Findings:

1. **Runtime Backends:**
   - ❌ `gpu_backend.rs`, `npu_backend.rs`, `tpu_backend.rs` **DO NOT EXIST** (files not found)
   - ⚠️ `backends.rs` **references these non-existent modules** (will cause compilation errors)
   - ✅ Only need to remove references from `backends.rs` (no files to delete)

2. **Scalar Codegen:**
   - ✅ Packager uses **SSE scalar instructions** (movss, addss, etc.) - this is CORRECT
   - ✅ SSE scalar is for codegen (generating assembly), not runtime fallback
   - ✅ No separate scalar codegen module to delete (scalar is part of SIMD codegen)
   - ⚠️ **No action needed** - packager scalar codegen is actually SSE scalar instructions (correct)

3. **Runtime Scalar Operations:**
   - ✅ Found extensive scalar fallback operations in `runtime.rs`:
     - Arithmetic: Add, Multiply, Divide, Subtract
     - Activations: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish
     - Pooling: MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool
     - Normalization: BatchNorm, LayerNorm
     - Reductions: ReduceSum, ReduceMean, ReduceMax, ReduceMin
     - Matrix ops: MatMul, Gemm, Convolution
   - ✅ All operations are pure Rust (no SIMD dependencies)
   - ✅ Need to extract these to `coelanox-core/src/scalar/`

4. **Backend Traits:**
   - ⚠️ Need to investigate backend trait interface
   - ⚠️ Roadmap mentions "8-trait interface" but need to verify what exists

## 📝 Next Steps

1. **Investigate codebase** - Complete the checks above
2. **Create TODO list** - Break down tasks into actionable items
3. **Start with Task 1.1** - Security hash is critical blocker
4. **Implement incrementally** - One task at a time, test as we go
5. **Document changes** - Update architecture docs as we progress

## ⚠️ Risks & Considerations

1. **Hash Calculation Timing** - Must hash after compression but before writing
2. **Backend Interface** - Need to understand existing backend traits before creating manager
3. **Scalar Extraction** - Need to ensure no functionality is lost when moving to core
4. **Testing** - Need comprehensive tests for each change
5. **Backward Compatibility** - Old containers may not have hashes (need migration strategy)

## 📚 References

- Roadmap: `/home/shark/Saic/roadmaps/PRODUCTION_MVP_ROADMAP.md`
- Format definition: `coelanox-core/src/format.rs`
- CLI structure: `coelanox-cli/src/main.rs`
- Runtime structure: `coelanox-runtime/src/runtime.rs`

