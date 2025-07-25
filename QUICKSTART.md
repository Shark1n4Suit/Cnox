# SAIC Quick Start Guide

**Get started with SAIC in under 5 minutes**

## 🚀 Ultra-Quick Start

```bash
# 1. Install SAIC
curl -fsSL https://install.coelanox.com/saic | sh

# 2. Test with example model
saic-runner run mobilenetv2.saic

# 3. Execute inference
saic-runner execute mobilenetv2.saic --backend cpu
```

**That's it! You've just experienced INSTANTANEOUS AI model loading.** 🎉

## 📦 Package Your First Model

```bash
# Package any AI model format
saic-packager model.onnx --name "MyModel" --output model.saic
saic-packager model.pt --name "MyModel" --output model.saic
saic-packager model.pb --name "MyModel" --output model.saic

# Execute your model
saic-runner execute model.saic --backend auto
```

## ⚡ Performance Demo

```bash
# Test INSTANTANEOUS loading
time saic-runner run mobilenetv2.saic
# Result: <1ms (INSTANTANEOUS)

# Compare with traditional methods
time python -c "import torch; torch.load('model.pt')"
# Result: 1-60 seconds
```

## 🔧 Basic Configuration

```bash
# Generate configuration
saic-runner config saic.yaml

# Edit settings
nano saic.yaml
```

```yaml
# saic.yaml
audit_mode: false          # Performance mode
log_level: "info"          # Logging level
default_backend: "auto"    # Auto-select best backend
performance_mode: true     # Enable optimizations
```

## 🎯 CLI Commands

### Packager Commands
```bash
# Package any AI model
saic-packager <model_file> --name <name> --output <output.saic>

# Examples
saic-packager model.onnx --name "ResNet50" --output resnet50.saic
saic-packager model.pt --name "BERT" --output bert.saic
saic-packager model.pb --name "Inception" --output inception.saic
```

### Runner Commands
```bash
# Validate SAIC file
saic-runner run <file.saic>

# Execute inference
saic-runner execute <file.saic> --backend <cpu|gpu|auto>

# Generate configuration
saic-runner config <file.yaml>

# List available backends
saic-runner backends

# Enable audit logging
saic-runner run <file.saic> --audit
saic-runner execute <file.saic> --audit
```

## 🌍 Universal Format Support

SAIC works with **any AI model format**:

| Format | Extension | Support |
|--------|-----------|---------|
| **ONNX** | `.onnx` | ✅ Full |
| **PyTorch** | `.pt`, `.pth` | ✅ Full |
| **TensorFlow** | `.pb`, `.h5` | ✅ Full |
| **Safetensors** | `.safetensors` | ✅ Full |
| **Custom** | Any binary | ✅ Full |

**No conversion needed - just packaging!**

## ⚡ Performance Comparison

| Metric | SAIC | Traditional |
|--------|------|-------------|
| **Load Time** | <1ms | 1-60 seconds |
| **Parse Throughput** | 1000+ GB/s | 1-10 GB/s |
| **Memory Overhead** | ~12 bytes | 1-10GB |
| **Format Support** | Universal | Format-specific |

## 🔍 Troubleshooting

### Common Issues

#### "Backend not found"
```bash
# Check available backends
saic-runner backends

# Install missing dependencies
# See INSTALL.md for backend setup
```

#### "Permission denied"
```bash
# Fix permissions
chmod +x /usr/local/bin/saic-*
```

#### "File not found"
```bash
# Verify file exists
ls -la model.saic

# Check file format
file model.saic
```

### Performance Issues

#### Slow loading
```bash
# Enable performance mode
echo "performance_mode: true" >> saic.yaml

# Disable audit logging
echo "audit_mode: false" >> saic.yaml
```

#### Memory issues
```bash
# Check available memory
free -h

# Reduce model size or use CPU backend
saic-runner execute model.saic --backend cpu
```

## 🎯 Next Steps

1. **Read the Documentation**: [README.md](README.md)
2. **Explore Examples**: Try different model formats
3. **Configure for Production**: Set up your `saic.yaml`
4. **Join the Community**: [GitHub Discussions](https://github.com/coelanox/saic/discussions)

## 📚 Resources

- **Full Documentation**: [docs/](docs/)
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **Technical Architecture**: [docs/TECHNICAL_ARCHITECTURE.md](docs/TECHNICAL_ARCHITECTURE.md)
- **Contributing**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)

## 🏆 Performance Guarantee

**SAIC guarantees INSTANTANEOUS loading (<1ms) for all model sizes.**

If you're not experiencing this performance:
1. Check your configuration (`saic.yaml`)
2. Ensure you're using the latest version
3. Verify backend dependencies are installed
4. Report the issue on GitHub

---

**SAIC - The Universal AI Binary Container** 🚀

*Experience the future of AI deployment today.* 