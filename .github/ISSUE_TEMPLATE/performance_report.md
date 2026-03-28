---
name: Performance report
about: Report performance issues or improvements
title: '[PERFORMANCE] '
labels: ['performance']
assignees: ''

---

**Performance Issue Type**
- [ ] Load time slower than expected (<1ms)
- [ ] Parse throughput below 1000+ GB/s
- [ ] Memory usage higher than expected
- [ ] Backend execution slow
- [ ] Performance regression from previous version
- [ ] Performance improvement achieved
- [ ] Other: [describe]

**Environment**
- OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 11]
- CNOX Version: [e.g. v0.7.0]
- Rust Version: [e.g. 1.70.0]
- Backend: [e.g. CPU (OpenVINO), GPU (TensorRT)]
- Hardware: [e.g. Intel i7-10700K, NVIDIA RTX 3080, 32GB RAM]

**Model Information**
- Model format: [e.g. ONNX, PyTorch, TensorFlow]
- Model size: [e.g. 13MB, 1GB, 5GB]
- Model name/type: [e.g. MobileNetV2, ResNet50, BERT]

**Performance Measurements**

### Current Performance
```
# Load time measurement
time cnox-runner run model.cnox
# Result: [actual time]

# Parse throughput calculation
# File size: [size in bytes]
# Load time: [time in seconds]
# Throughput: [calculated GB/s]

# Memory usage
# Peak memory: [MB/GB]
# Average memory: [MB/GB]
```

### Expected Performance
- Expected load time: [e.g. <1ms]
- Expected throughput: [e.g. 1000+ GB/s]
- Expected memory usage: [e.g. <10MB]

### Performance Comparison
- Previous version performance: [if applicable]
- Traditional method performance: [if applicable]
- Performance difference: [e.g. 10x slower, 2x faster]

**Steps to Reproduce**
1. Download/package model: `...`
2. Run performance test: `...`
3. Measure results: `...`

**Configuration**
```yaml
# cnox.yaml configuration
audit_mode: false
log_level: "info"
default_backend: "auto"
performance_mode: true
max_model_size: 0
enable_caching: true
cache_dir: "/tmp/cnox-cache"
```

**Additional Context**
- Is this a regression from a previous version?
- Does this affect all models or specific ones?
- Are there specific conditions that trigger the issue?
- Any error messages or warnings?

**Files to Attach**
- CNOX file (if the issue is with a specific model)
- Configuration file (cnox.yaml)
- Performance logs or output
- System resource monitoring data

**Performance Optimization Suggestions**
If you have ideas for improving performance, please share:
- Potential bottlenecks identified
- Optimization strategies
- Alternative approaches considered

---

**Checklist:**
- [ ] I have measured performance with multiple runs
- [ ] I have compared against the example mobilenetv2.cnox file
- [ ] I have provided detailed timing information
- [ ] I have included system specifications
- [ ] I have tested with different configurations 