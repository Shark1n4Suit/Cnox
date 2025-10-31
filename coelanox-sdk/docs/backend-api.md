# Backend API Reference

Partner-only API for hardware vendors building execution backends.

## Access Control

**This API is restricted to verified hardware partners.**

- Registration required with COELANOX team
- Security review and approval process
- Sign partnership agreement
- Receive API credentials and documentation package

## Overview

Backends translate COELANOX operations to hardware-specific code and execute them on target hardware.

## BackendTranslator Trait

```rust
use coelanox_core::backends::BackendTranslator;
use coelanox_core::operations::{CoelanoxAiOperation, CoelanoxAiOperationParams};

pub trait BackendTranslator: Send + Sync {
    fn translate_operation(
        &self, 
        operation: &CoelanoxAiOperation, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<String>;
    
    fn optimize_code(&self, code: &str, target: HardwareTarget) -> CoelanoxResult<String>;
    
    fn supported_hardware(&self) -> Vec<HardwareTarget>;
    
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    
    fn supports_native_coelanox(&self) -> bool;
    
    fn get_performance_estimate(
        &self, 
        operation: &CoelanoxAiOperation, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<CoelanoxPerformanceEstimate>;
    
    fn apply_optimization_hints(
        &self, 
        code: &str, 
        hints: &[CoelanoxOptimizationHint]
    ) -> CoelanoxResult<String>;
    
    fn validate_parameters(
        &self, 
        operation: &CoelanoxAiOperation, 
        params: &CoelanoxAiOperationParams
    ) -> CoelanoxResult<()>;
}
```

## Required Methods

### translate_operation()

Translates COELANOX operations to hardware code.

```rust
fn translate_operation(&self, op: &CoelanoxAiOperation, params: &CoelanoxAiOperationParams) -> CoelanoxResult<String> {
    match op {
        CoelanoxAiOperation::Convolution => {
            generate_convolution_kernel(params, &self.hardware)
        }
        CoelanoxAiOperation::MatMul => {
            generate_matmul_kernel(params, &self.hardware)
        }
        _ => Err(CoelanoxError::UnsupportedOperation)
    }
}
```

### optimize_code()

Optimizes generated code for target hardware.

```rust
fn optimize_code(&self, code: &str, target: HardwareTarget) -> CoelanoxResult<String> {
    apply_hardware_specific_optimizations(code, target)
}
```

### supported_hardware()

Returns supported hardware targets.

```rust
fn supported_hardware(&self) -> Vec<HardwareTarget> {
    vec![
        HardwareTarget::GPU("NVIDIA"),
        HardwareTarget::GPU("AMD"),
    ]
}
```

## Configuration

```rust
#[derive(Debug, Clone)]
pub struct CoelanoxPerformanceEstimate {
    pub estimated_latency_us: u64,
    pub estimated_throughput: f64,
    pub memory_usage_bytes: usize,
}

#[derive(Debug)]
pub enum CoelanoxOptimizationHint {
    TileSize(usize),
    MemoryLayout(String),
    Parallelism(usize),
}
```

## Examples

See `examples/` for complete backend implementations.

## Reference

- [Getting Started Guide](./getting-started.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./best-practices.md)
