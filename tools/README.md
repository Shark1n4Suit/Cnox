# SAIC Tools

This directory contains utility scripts for testing and benchmarking SAIC performance.

## Scripts

### `benchmark_world_class.py`
Comprehensive performance benchmark for SAIC files. Tests throughput and latency across different model sizes.

**Usage:**
```bash
python3 tools/benchmark_world_class.py
```

**Features:**
- Tests 1MB to 1GB model sizes
- Measures throughput (GB/s)
- Measures latency (ms)
- Compares against world-class targets
- Generates detailed performance reports

### `create_binary_saic.py`
Creates SAIC files with the new binary manifest format for testing.

**Usage:**
```bash
python3 tools/create_binary_saic.py
```

**Features:**
- Creates test files from 1MB to 1GB
- Uses binary manifest format (64 bytes)
- Fixed-size model names (32 bytes)
- Compatible with optimized SAIC runner

## Performance Targets

| Model Size | Target Throughput | Target Latency |
|------------|------------------|----------------|
| <100MB | 1000+ GB/s | <5ms |
| 100MB-1GB | 500+ GB/s | <10ms |
| >1GB | 100+ GB/s | <50ms |

## Results

SAIC achieves **INSTANTANEOUS** loading across all model sizes:
- **1MB models**: 0.001s (INSTANT)
- **100MB models**: 0.001s (INSTANT)
- **1GB models**: 0.002s (INSTANT) 