# COELANOX SDK Best Practices

Recommended practices for building translators and backends.

## 🎯 General Principles

### **1. Use Builder API**
❌ **Don't construct IR directly**
```rust
// BAD - Direct construction (not exposed)
let ir = UniversalIR::new(...);
```

✅ **DO use Builder API**
```rust
// GOOD - Use builder
let mut builder = UniversalIRBuilder::new();
builder.set_model_name("my_model");
// ... add operations ...
let ir = builder.build()?;
```

### **2. Validate Early**
❌ **Don't defer validation**
```rust
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
    let model = parse(data)?;
    // ... lots of processing ...
    // Validate only at the end
    validate(model)?;
}
```

✅ **DO validate at start**
```rust
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
    // Validate format first
    self.validate_format(data)?;
    let model = parse(data)?;
    // ... rest of processing ...
}
```

### **3. Handle Errors Properly**
❌ **Don't use plain strings**
```rust
fn extract_weights(&self, data: &[u8]) -> Result<Vec<f32>> {
    Ok(vec![]) // No error handling
}
```

✅ **DO return proper errors**
```rust
fn extract_weights(&self, data: &[u8]) -> CoelanoxResult<Vec<f32>> {
    let model = parse(data).map_err(|e| {
        CoelanoxError::format(&format!("Failed to parse: {}", e))
    })?;
    Ok(extract_weights(&model))
}
```

## 🔧 Translator Best Practices

### **1. Complete Metadata**
Always include all available metadata:

```rust
let mut builder = UniversalIRBuilder::new();
builder.set_model_name(&model.name);
builder.set_description(model.description.as_ref().unwrap_or(&String::new()));
builder.set_input_shapes(&model.input_shapes);
builder.set_output_shapes(&model.output_shapes);
```

### **2. Proper Operation Mapping**
Map all operations, even if to custom:

```rust
match operation {
    Op::Add => builder.add_operation(OpType::Add, params),
    Op::Mul => builder.add_operation(OpType::Multiply, params),
    Op::Unsupported => builder.add_operation(
        OpType::Custom(format!("{:?}", operation)), 
        params
    ),
}
```

### **3. Handle Edge Cases**
- Empty models
- Single-node graphs
- Missing metadata
- Unsupported operations

## 🖥️ Backend Best Practices

### **1. Hardware Initialization**
Initialize hardware early and reuse context:

```rust
impl BackendTranslator for MyBackend {
    fn new() -> Self {
        let hardware = initialize_hardware();
        MyBackend { hardware }
    }
    
    fn translate_operation(&self, op: &CoelanoxAiOperation) -> Result<String> {
        // Use pre-initialized hardware context
        generate_code(op, &self.hardware)
    }
}
```

### **2. Performance Hints**
Use CSV optimization data to improve performance:

```rust
fn apply_optimization_hints(&self, code: &str, hints: &[CoelanoxOptimizationHint]) -> Result<String> {
    let mut optimized = code.to_string();
    for hint in hints {
        match hint {
            CoelanoxOptimizationHint::TileSize(size) => {
                optimized = optimize_tiles(&optimized, *size);
            }
            _ => {}
        }
    }
    Ok(optimized)
}
```

### **3. Validate Parameters**
Always validate before execution:

```rust
fn validate_parameters(&self, op: &CoelanoxAiOperation, params: &CoelanoxAiOperationParams) -> Result<()> {
    match op {
        CoelanoxAiOperation::Convolution => {
            check_convolution_params(params)?;
        }
        _ => {}
    }
    Ok(())
}
```

## 🧪 Testing Best Practices

### **1. Unit Tests for Each Component**
```rust
#[test]
fn test_parsing() {
    let translator = MyTranslator;
    let data = include_bytes!("fixtures/model.myformat");
    assert!(translator.validate(data).is_ok());
}
```

### **2. Integration Tests**
```rust
#[test]
fn test_full_pipeline() {
    let translator = MyTranslator;
    let ir = translator.translate(test_data).unwrap();
    assert!(!ir.is_empty());
}
```

### **3. Edge Case Tests**
- Empty models
- Large models
- Invalid data
- Missing fields

## 📈 Performance Best Practices

### **1. Lazy Evaluation**
Parse only what you need:

```rust
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
    // Only parse required parts first
    let header = parse_header(data)?;
    let operations = parse_operations(data, header)?;
    // ... rest
}
```

### **2. Memory Efficiency**
Reuse buffers, avoid unnecessary allocations:

```rust
// Reuse buffer
let mut buffer = Vec::with_capacity(expected_size);
// ... reuse buffer multiple times ...
```

### **3. Parallel Processing**
Use parallel processing where possible:

```rust
use rayon::prelude::*;

operations.par_iter()
    .map(|op| translate_operation(op))
    .collect()
```

## 🔒 Security Best Practices

### **1. Input Validation**
Always validate input before processing:

```rust
fn validate(&self, data: &[u8]) -> CoelanoxResult<()> {
    if data.len() < MIN_HEADER_SIZE {
        return Err(CoelanoxError::InvalidFormat);
    }
    // ... more validation ...
}
```

### **2. Sanitize File Paths**
Always sanitize file paths:

```rust
fn translate_from_path(&self, path: &Path) -> CoelanoxResult<UniversalIR> {
    // Check for path traversal attacks
    if path.to_string_lossy().contains("..") {
        return Err(CoelanoxError::Security("Path traversal detected"));
    }
    // ... rest
}
```

### **3. Resource Limits**
Enforce resource limits:

```rust
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR> {
    if data.len() > MAX_MODEL_SIZE {
        return Err(CoelanoxError::ModelTooLarge);
    }
    // ... rest
}
```

## 📚 Documentation Best Practices

### **1. Document All Public Methods**
```rust
/// Translate model from format X to Universal IR.
///
/// # Arguments
/// * `data` - Raw model data in format X
///
/// # Returns
/// * `UniversalIR` - Translated Universal IR
///
/// # Errors
/// * `InvalidFormat` - If data is not in format X
/// * `ParseError` - If parsing fails
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR>;
```

### **2. Include Examples**
```rust
/// # Example
/// ```rust
/// let translator = MyTranslator;
/// let ir = translator.translate(&model_data)?;
/// ```
fn translate(&self, data: &[u8]) -> CoelanoxResult<UniversalIR>;
```

### **3. Document Edge Cases**
```rust
/// Translates model to Universal IR.
///
/// # Edge Cases
/// * Empty models return minimal IR with no operations
/// * Missing metadata uses default values
/// * Unsupported operations map to Custom operations
```

## ✅ Summary Checklist

- [ ] Use Builder API for IR construction
- [ ] Validate input early
- [ ] Handle all error cases
- [ ] Include complete metadata
- [ ] Map all operations
- [ ] Initialize hardware once
- [ ] Use CSV optimization data
- [ ] Validate parameters
- [ ] Write comprehensive tests
- [ ] Document all public APIs
- [ ] Include usage examples
- [ ] Sanitize inputs
- [ ] Enforce resource limits

