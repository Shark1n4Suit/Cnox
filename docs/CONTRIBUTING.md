# Contributing to Coelanox SAIC

Thank you for your interest in contributing to **Coelanox SAIC**! We welcome contributions from developers, researchers, and AI practitioners who want to help build the future of secure AI deployment.

## 🎯 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **💡 Feature Requests**: Suggest new capabilities and improvements
- **📝 Documentation**: Improve guides, tutorials, and API documentation
- **🔧 Code Contributions**: Implement features, fix bugs, and optimize performance
- **🧪 Testing**: Expand test coverage and validate across different hardware
- **📊 Benchmarks**: Performance testing and optimization analysis
- **🎨 Examples**: Create sample models and integration demonstrations
- **🚀 Performance Optimization**: Help us maintain world-class performance standards

## 🚀 Getting Started

### Prerequisites

- **Rust 1.70+** with Cargo
- **Git** for version control
- **ONNX models** for testing (optional)
- **Hardware**: CPU (required), NVIDIA GPU (optional for TensorRT testing)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/saic.git
   cd saic
   git remote add upstream https://github.com/coelanox/saic.git
   ```

2. **Install Dependencies**
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Build the project
   cargo build --features full-inference
   ```

3. **Run Tests**
   ```bash
   # Unit tests
   cargo test
   
   # Integration tests
   cargo test --test integration
   
   # Backend-specific tests
   cargo test --features cpu-backend
   ```

4. **Verify Installation**
   ```bash
   # Test packager
   cargo run --bin packager -- --help
   
   # Test runner
   cargo run --bin runner -- --help
   ```

## 🏆 Performance Standards

SAIC maintains **world-class performance** standards. All contributions must preserve or improve our current performance achievements:

### **Current Performance Benchmarks**
- **INSTANTANEOUS Loading**: <1ms for all model sizes
- **World-Class Throughput**: 1000+ GB/s parse speed
- **Memory Efficiency**: Zero-copy operations throughout
- **Audit Compliance**: Full logging when needed, silent when not

### **Performance Requirements**
- **No regression**: New features must not degrade current performance
- **Zero-copy first**: Prefer zero-copy operations over buffered I/O
- **Memory mapping**: Use memory mapping for file operations
- **Conditional logging**: Audit logging must be optional and zero-overhead
- **Binary formats**: Prefer binary formats over text parsing

### **Performance Testing**
```bash
# Run performance benchmarks
python3 tools/benchmark_world_class.py

# Test loading performance
time ./runner/target/release/runner run mobilenetv2.saic

# Verify audit mode doesn't impact performance
time ./runner/target/release/runner run mobilenetv2.saic --audit
```

## 📋 Development Workflow

### Branch Naming Convention

- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/issue-description`
- **Documentation**: `docs/section-name`
- **Performance**: `perf/optimization-area`
- **Backend support**: `backend/hardware-name`

### Commit Message Format

We follow the [Conventional Commits](https://conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Examples:**
```bash
feat(runner): add AMD ROCm backend support

Implement new AmdBackend struct with ROCm integration for AMD GPU acceleration.
Includes memory management, model loading, and inference execution.

Closes #123
```

```bash
fix(packager): resolve manifest parsing error for large models

Fixed JSON parsing issue when processing models with manifests larger than 1MB.
Added proper error handling and validation for manifest size limits.

Fixes #456
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🔧 Code Guidelines

### Rust Style

- Follow official [Rust style guidelines](https://doc.rust-lang.org/1.0.0/style/)
- Use `rustfmt` for code formatting: `cargo fmt`
- Use `clippy` for linting: `cargo clippy`
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Organization

```rust
// File structure example for new backends
src/
├── backends/
│   ├── mod.rs           // Backend trait definition
│   ├── cpu.rs          // OpenVINO CPU backend
│   ├── gpu.rs          // TensorRT GPU backend
│   └── amd.rs          // New AMD backend (example)
├── format/
│   ├── parser.rs       // SAIC format parsing
│   └── validator.rs    // Format validation
└── main.rs            // CLI interface
```

### Documentation Standards

- **Public APIs**: Comprehensive rustdoc comments
- **Complex functions**: Inline comments explaining logic
- **Error handling**: Document all error conditions
- **Examples**: Include usage examples in documentation

```rust
/// Execute AI model inference using the specified backend
/// 
/// This function loads the model from the SAIC container, initializes
/// the backend hardware, and executes inference with the provided input.
/// 
/// # Arguments
/// 
/// * `model_payload` - Raw model data from SAIC container
/// * `model_name` - Human-readable model identifier for logging
/// 
/// # Returns
/// 
/// Returns `Ok(())` on successful execution, or `BackendError` if:
/// - Backend hardware is unavailable
/// - Model format is incompatible
/// - Execution fails due to resource constraints
/// 
/// # Examples
/// 
/// ```rust
/// let mut backend = CpuBackend::new();
/// backend.execute(&model_data, "ResNet50")?;
/// ```
pub fn execute(&mut self, model_payload: &[u8], model_name: &str) -> Result<(), BackendError> {
    // Implementation here
}
```

### Error Handling

- Use `Result<T, E>` for all fallible operations
- Provide informative error messages with context
- Create custom error types for different error categories
- Log errors with appropriate levels (error!, warn!, info!, debug!)

```rust
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Backend initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
}
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and modules
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_saic_header_parsing() {
           let header = [b'S', b'A', b'I', b'C', 0, 1, 0, 0];
           let result = parse_header(&header);
           assert!(result.is_ok());
       }
   }
   ```

2. **Integration Tests**: Test component interactions
   ```bash
   # Create integration test files in tests/
   tests/
   ├── packager_integration.rs
   ├── runner_integration.rs
   └── backend_tests.rs
   ```

3. **Performance Tests**: Benchmark critical paths
   ```rust
   #[cfg(test)]
   mod benchmarks {
       use criterion::{black_box, criterion_group, criterion_main, Criterion};
       
       fn benchmark_model_loading(c: &mut Criterion) {
           c.bench_function("load_model", |b| {
               b.iter(|| load_model(black_box(&model_data)))
           });
       }
   }
   ```

### Test Data

- Use small, representative test models
- Include both valid and invalid SAIC files
- Test edge cases (empty files, corrupted data, large models)
- Mock external dependencies when possible

## 📊 Performance Considerations

### Benchmarking

We use [Criterion.rs](https://crates.io/crates/criterion) for performance benchmarks:

```bash
# Run benchmarks
cargo bench

# Compare against baseline
cargo bench --baseline main
```

### Memory Safety

- Avoid `unsafe` code unless absolutely necessary
- Document all `unsafe` blocks with safety justification
- Use tools like `valgrind` and `miri` for memory validation
- Profile memory usage with `heaptrack` or similar tools

### Backend Performance

- Minimize memory allocations in hot paths
- Use zero-copy operations where possible
- Implement proper resource pooling for GPU backends
- Add performance logging for optimization analysis

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for duplicates
2. **Test with latest version** from main branch
3. **Reproduce consistently** with minimal example
4. **Check hardware compatibility** (CPU/GPU requirements)

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Create SAIC file with: `saic-packager --model model.onnx --name test --output test.saic`
2. Run with: `saic-runner execute test.saic --backend cpu`
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Rust version: [e.g., 1.70.0]
- SAIC version: [e.g., v0.5.0-beta]
- Hardware: [e.g., Intel i7-12700K, NVIDIA RTX 3080]
- Backend: [e.g., OpenVINO 2023.1, TensorRT 8.6]

**Additional Context**
- Error messages and logs
- Model information (format, size, framework)
- Any other relevant details
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe your specific use case and why this feature would be valuable.

**Proposed Implementation**
If you have ideas for how this could be implemented, please describe them.

**Alternatives Considered**
What alternative solutions have you considered?

**Priority**
How important is this feature to you? (Low/Medium/High/Critical)
```

## 🎯 Backend Development

### Adding New Backends

1. **Create backend module**: `src/backends/your_backend.rs`
2. **Implement Backend trait**: All required methods
3. **Add feature flag**: In `Cargo.toml` under `[features]`
4. **Add conditional compilation**: Use `#[cfg(feature = "your-backend")]`
5. **Update documentation**: CLI help, README, and docs
6. **Add tests**: Unit, integration, and performance tests
7. **Update CI/CD**: Add backend to build matrix

### Backend Trait Implementation

```rust
impl Backend for YourBackend {
    fn execute(&mut self, model_payload: &[u8], model_name: &str) -> Result<(), BackendError> {
        // Your implementation here
    }
    
    fn name(&self) -> &str {
        "Your Backend"
    }
    
    fn is_available(&self) -> bool {
        // Check if hardware/software is available
    }
}
```

## 📝 Documentation

### Types of Documentation

- **API Documentation**: Rustdoc comments for all public items
- **User Guides**: Tutorials and how-to guides in `docs/`
- **Examples**: Working code samples in `examples/`
- **Architecture Docs**: Design decisions and system architecture

### Documentation Standards

- Write for your audience (beginners vs. experts)
- Include practical examples
- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include diagrams where helpful

## 🔄 Pull Request Process

### Before Submitting

1. **Create feature branch** from latest `main`
2. **Implement your changes** following code guidelines
3. **Add/update tests** for your changes
4. **Update documentation** if needed
5. **Run full test suite**: `cargo test --all-features`
6. **Check formatting**: `cargo fmt --check`
7. **Run linter**: `cargo clippy -- -D warnings`

### Pull Request Template

```markdown
**Description**
Brief description of changes and why they were made.

**Type of Change**
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings introduced
```

### Review Process

1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one maintainer approval
3. **Performance Check**: No significant regression
4. **Documentation Review**: Updates match code changes
5. **Final Testing**: Manual verification if needed

## 🏆 Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Significant contributions highlighted
- **GitHub**: Contributor statistics and graphs
- **Community**: Shoutouts in Discord and social media

## 📞 Getting Help

- **GitHub Discussions**: [github.com/coelanox/saic/discussions](https://github.com/coelanox/saic/discussions)
- **Discord**: [discord.gg/coelanox](https://discord.gg/coelanox)
- **Email**: [contribute@coelanox.com](mailto:contribute@coelanox.com)
- **Documentation**: [docs.coelanox.com/saic](https://docs.coelanox.com/saic)

## 📄 License

By contributing to SAIC, you agree that your contributions will be licensed under the Business Source License 1.1 (BSL 1.1). See the [LICENSE](../LICENSE) file for details.

---

**Thank you for contributing to the future of secure AI deployment!** 🚀

*The Coelanox SAIC Team* 