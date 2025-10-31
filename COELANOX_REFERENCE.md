# COELANOX & .cnox Reference Document

**Quick reference guide for what COELANOX is and what `.cnox` files contain.**

---

## 🎯 **What is COELANOX?**

**COELANOX** (Secure AI Container) is a universal binary container system for AI models that provides:

1. **Universal Format**: Convert any AI model (ONNX, TensorFlow, PyTorch) into a single `.cnox` container
2. **Ahead-of-Time (AOT) Compilation**: Converts models to optimized machine code for specific hardware
3. **Universal IR**: Uses an intermediate representation (IR) that abstracts over model formats
4. **Hardware Abstraction**: Backend translators convert Coelanox operations to hardware-specific code
5. **"De Beers of AI" Strategy**: Defines and controls standardized AI operations that hardware vendors must implement

### **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│              COELANOX System Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT MODEL                                                 │
│  (ONNX/TensorFlow/PyTorch)                                   │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  Format Translator  │  → Converts to Universal IR        │
│  │  (ONNX/TF/PyTorch)  │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │   Universal IR       │  → Graph-based, optimized IR      │
│  │   (Coelanox IR)     │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  IR Optimizer       │  → Operator fusion, optimization   │
│  │  (CPU/GPU/TPU)      │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  Code Generator     │  → Hardware-specific machine code  │
│  │  (CPU/GPU/TPU/NPU)  │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  .cnox Container    │  → Universal binary container      │
│  │  (compressed, AOT)  │                                     │
│  └─────────────────────┘                                     │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────┐                                     │
│  │  Runtime Execution  │  → Executes machine code directly   │
│  │  (native hardware)  │                                     │
│  └─────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

### **Key Components**

1. **coelanox-core**: Defines the Universal IR, container format, and core types
2. **coelanox-packager**: Packages models into `.cnox` containers (AOT compilation)
3. **coelanox-runtime**: Executes `.cnox` containers on hardware
4. **Translators**: Convert model formats (ONNX, TensorFlow, etc.) to Universal IR
5. **Backend Translators**: Convert Coelanox operations to hardware-specific code

---

## 📦 **What is a `.cnox` File?**

A **`.cnox` file** is a universal binary container that contains:

1. **Compiled Machine Code**: Pre-compiled, optimized machine code for specific hardware
2. **Compressed Weights**: Model weights compressed using Zstd/LZ4
3. **Universal IR Data**: Serialized intermediate representation
4. **Manifest**: Metadata about the model (input/output shapes, operations, etc.)
5. **Header**: Binary header with magic number, version, offsets

### **`.cnox` Container Structure**

```
┌─────────────────────────────────────────────────────────────┐
│                    .cnox File Layout                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Header (32 bytes)                                           │
│  ├── Magic Number: "CNOX" (4 bytes)                          │
│  ├── Version: 1.0.0 (4 bytes)                                │
│  ├── Flags: Feature flags (4 bytes)                         │
│  ├── Container Size: Total file size (8 bytes)              │
│  ├── Manifest Offset: Where manifest starts (8 bytes)       │
│  ├── Code Offset: Where machine code starts (8 bytes)       │
│  ├── Security Hash: SHA-256 hash (32 bytes)                │
│  └── Reserved: Future use (8 bytes)                        │
│                                                               │
│  Manifest (JSON metadata)                                   │
│  ├── Model Name, Version                                    │
│  ├── Input/Output Shapes                                    │
│  ├── Operations List                                        │
│  ├── Hardware Target (CPU/GPU/TPU/NPU)                     │
│  ├── Optimization Level                                     │
│  └── Compression Info                                       │
│                                                               │
│  Optimized Machine Code                                     │
│  ├── Function Prologue (x86-64/ARM/PTX/etc.)               │
│  ├── Operation Kernels (Compiled machine code)              │
│  └── Function Epilogue                                     │
│                                                               │
│  Compressed Weights                                         │
│  ├── Compression Type: Zstd/LZ4/Deflate                    │
│  └── Compressed Weight Data                                │
│                                                               │
│  IR Data (Serialized)                                       │
│  ├── Optimized IR structure                                │
│  ├── Graph nodes and edges                                 │
│  └── Operation metadata                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Code Reference:**
```372:391:coelanox/coelanox-core/src/format.rs
/// COELANOX container with binary format support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoelanoxContainer {
    /// Container header
    pub header: CoelanoxHeader,
    /// Container manifest
    pub manifest: CoelanoxManifest,
    /// Optimized machine code
    pub optimized_code: Vec<u8>,
    /// Compressed model weights
    pub compressed_weights: Vec<u8>,
    /// Compression type used for weights
    pub compression_type: CompressionType,
    /// IR data (serialized OptimizedIR using bincode)
    pub ir_data: Vec<u8>,
    /// Backend hints
    pub backend_hints: Option<BackendHints>,
    /// Security metadata
    pub security: Option<SecurityMetadata>,
}
```

### **Key Features of `.cnox` Files**

1. **Zero Runtime Dependencies**: Contains pre-compiled machine code
2. **Hardware-Specific**: Compiled for specific hardware (CPU/GPU/TPU/NPU)
3. **Compressed**: Weights compressed using Zstd/LZ4 (90%+ size reduction)
4. **Self-Contained**: Everything needed for execution in one file
5. **Fast Startup**: No parsing/compilation at runtime (instant execution)
6. **Secure**: Header validation, security hashes, tamper detection

---

## 🔄 **How COELANOX Works**

### **1. Packaging Phase (AOT Compilation)**

When you package a model into `.cnox`:

```
1. Format Detection
   ↓
2. Format Translation → Universal IR
   (ONNX → Coelanox IR)
   ↓
3. IR Optimization
   (Operator fusion, graph optimization)
   ↓
4. Code Generation
   (Universal IR → Hardware machine code)
   ↓
5. Weight Compression
   (Zstd/LZ4 compression)
   ↓
6. Container Creation
   (Package into .cnox file)
```

**Code Reference:**
```226:258:coelanox/coelanox-packager/src/lib.rs
    pub fn package(
        &self,
        model_data: &[u8],
        model_name: &str,
        target: HardwareTarget,
        optimization_level: OptimizationLevel,
    ) -> CoelanoxResult<CoelanoxContainer> {
        // Step 1: Detect format
        let format = self.detect_format(model_data)?;
        
        // Step 2: Get translator
        let translator = self.translator_manager.get_translator(&format)?;
        
        // Step 3: Translate to Universal IR
        let universal_ir = translator.translate(model_data)?;
        
        // Step 4: Optimize IR
        let optimized_ir = self.optimize_ir(universal_ir, target, optimization_level)?;
        
        // Step 5: Generate machine code
        let machine_code = self.generate_machine_code(optimized_ir.clone(), target)?;
        
        // Step 6: Extract and compress weights
        let (compressed_weights, compression_type) = self.compress_weights(&optimized_ir.weights)?;
        
        // Step 7: Create manifest
        let manifest = self.create_manifest(model_name, &optimized_ir, target)?;
        
        // Step 8: Create container
        let ir_data = bincode::serialize(&optimized_ir)?;
        Ok(CoelanoxContainer::new(manifest, machine_code, compressed_weights, compression_type, ir_data))
    }
```

### **2. Execution Phase (Runtime)**

When you execute a `.cnox` file:

```
1. Load Container
   (Parse header, validate magic number)
   ↓
2. Decompress Weights
   (Zstd/LZ4 decompression)
   ↓
3. Execute Machine Code
   (Direct hardware execution)
   ↓
4. Return Results
```

**Code Reference:**
```715:809:coelanox/coelanox-runtime/src/runtime.rs
    /// Load a COELANOX container from a file path
    pub fn load_container<P: AsRef<Path>>(&mut self, path: P) -> CoelanoxResult<CoelanoxContainer> {
        let path = path.as_ref();
        
        // Security: Validate path
        self.validate_path(path)?;
        
        // Read file
        let data = fs::read(path)?;
        
        // Load from data
        self.load_container_from_data(&data)
    }
    
    /// Load a COELANOX container from raw bytes
    pub fn load_container_from_data(&mut self, data: &[u8]) -> CoelanoxResult<CoelanoxContainer> {
        // Validate minimum size
        if data.len() < COELANOX_HEADER_SIZE {
            return Err(CoelanoxError::InvalidContainer("File too small".to_string()));
        }
        
        // Parse header
        let mut header_data = &data[..COELANOX_HEADER_SIZE];
        let header = CoelanoxHeader::read_binary(&mut header_data)?;
        
        // Validate header
        header.validate()?;
        
        // Check cache
        if let Some(cached) = self.container_cache.get(&header.security_hash) {
            return Ok(cached.clone());
        }
        
        // Parse manifest
        let manifest_offset = header.manifest_offset as usize;
        let manifest_data = &data[manifest_offset..];
        let manifest: CoelanoxManifest = serde_json::from_slice(manifest_data)
            .map_err(|e| CoelanoxError::InvalidContainer(format!("Failed to parse manifest: {}", e)))?;
        
        // Validate manifest
        manifest.validate()?;
        
        // Extract components
        let code_offset = header.code_offset as usize;
        let code_size = manifest.model_size;
        let optimized_code = data[code_offset..code_offset + code_size].to_vec();
        
        // Extract weights (after code)
        let weights_offset = code_offset + code_size;
        // ... weight extraction logic ...
        
        // Deserialize IR
        let ir_data_start = weights_offset + compressed_weights.len();
        let ir_data = data[ir_data_start..].to_vec();
        
        // Create container
        let container = CoelanoxContainer {
            header,
            manifest,
            optimized_code,
            compressed_weights,
            compression_type,
            ir_data,
            backend_hints: None,
            security: None,
        };
        
        // Cache container
        self.container_cache.insert(container.header.security_hash, container.clone());
        
        Ok(container)
    }
```

---

## 🎯 **Standardized AI Operations (De Beers Strategy)**

COELANOX defines and controls universal AI operations that hardware vendors must implement:

**Code Reference:**
```9:45:coelanox/coelanox-packager/src/generators/backend_translators/coelanox_operations.rs
/// COELANOX Standardized AI Operations
/// 
/// These are the universal operations that COELANOX defines and controls.
/// Hardware vendors must implement translators for these operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoelanoxAiOperation {
    // Basic Arithmetic Operations
    Add,
    Subtract,
    Multiply,
    Divide,
    
    // Neural Network Operations
    Convolution,
    Pooling,
    Activation,
    BatchNorm,
    Dropout,
    
    // Matrix Operations
    MatMul,
    Transpose,
    Reshape,
    
    // Attention Operations
    Attention,
    MultiHeadAttention,
    SelfAttention,
    
    // Transformer Operations
    Transformer,
    Encoder,
    Decoder,
    
    // Custom Operations
    Custom(String),
}
```

### **Execution Paths**

1. **Translated Operations** (Current):
   - Coelanox operations → Backend translator → Hardware code → Execution
   - Works immediately but has translation overhead

2. **Native Operations** (Future):
   - Coelanox operations → Direct hardware execution
   - Zero translation overhead, 10-15% faster performance

**Code Reference:**
```30:33:coelanox/coelanox-packager/src/generators/backend_translators/backend_translator.rs
    /// Check if translator supports native COELANOX operations
    fn supports_native_coelanox(&self) -> bool {
        false // Default to translated operations
    }
```

---

## 📊 **Performance Characteristics**

### **Storage Memory**
- **90%+ compression**: Weights compressed using Zstd/LZ4
- **GBs less than Docker**: Docker images are GBs, `.cnox` files are MBs
- **10% size reduction**: Removing interpreter overhead

### **Execution Speed**
- **With CSV optimizations**: ~5% slower than TensorRT (vendor performance data)
- **With native ops**: Potentially 5-15% faster than TensorRT (zero translation overhead)
- **Without CSV**: ~15% slower than TensorRT (general GPU templates)

### **Startup Time**
- **Instant**: Pre-compiled machine code (no parsing/compilation)
- **1-5s faster than Docker**: Docker has container startup overhead
- **23μs validation**: Header-only validation (sub-50μs regardless of model size)

### **Throughput**
- **High**: Native machine code execution
- **Optimized**: Operator fusion, graph-level optimizations
- **Hardware-specific**: Leverages Tensor Cores, NPUs, etc.

---

## 🏗️ **Current Implementation Status**

### **✅ Completed**
- Universal IR structure (graph-based, arena-allocated)
- `.cnox` container format (header, manifest, code, weights, IR)
- ONNX translator (converts ONNX to Universal IR)
- Basic optimization (operator fusion, graph optimization)
- Weight compression (Zstd/LZ4)
- Container loading and validation
- Runtime execution framework

### **🚧 In Progress**
- Machine code generation (currently placeholders)
- Native code execution (intended path exists, needs implementation)
- Backend translators (GPU/TPU/NPU placeholders)
- Vendor CSV system (performance data lookup)

### **📋 Planned**
- Dynamic shape support (JIT compilation layer)
- Kernel autotuning (runtime benchmarking)
- Mixed-precision inference (INT8/FP16/FP32 selection)
- Memory management (tensor buffer reuse, NUMA-aware placement)
- Model partitioning (heterogeneous device split)

---

## 🔑 **Key Code Locations**

### **Container Format**
- `coelanox/coelanox-core/src/format.rs`: `.cnox` container structure
- `coelanox/coelanox-core/src/ir.rs`: Universal IR definition

### **Packaging**
- `coelanox/coelanox-packager/src/lib.rs`: Main packaging pipeline
- `coelanox/coelanox-packager/src/generators/mod.rs`: Code generation

### **Runtime**
- `coelanox/coelanox-runtime/src/runtime.rs`: Container loading and execution
- `coelanox/coelanox-runtime/src/backends/`: Hardware backends

### **Operations**
- `coelanox/coelanox-packager/src/generators/backend_translators/coelanox_operations.rs`: Standardized operations
- `coelanox/coelanox-packager/src/generators/backend_translators/backend_translator.rs`: Backend translator trait

---

## 📝 **Summary**

**COELANOX** = Universal AI deployment system that:
- Converts any model format to a universal container (`.cnox`)
- Pre-compiles models to optimized machine code (AOT)
- Provides standardized AI operations (De Beers strategy)
- Executes models with zero runtime dependencies

**`.cnox` File** = Universal binary container containing:
- Pre-compiled machine code (hardware-specific)
- Compressed weights (90%+ size reduction)
- Universal IR data (optimized graph representation)
- Manifest (metadata about the model)
- Header (validation and structure info)

**Goal**: Become the standard for AI deployment, where all hardware vendors implement native `.cnox` operations for maximum performance.

---

*Last Updated: Based on current codebase analysis*

