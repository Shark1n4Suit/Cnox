# COELANOX SDK Examples

Working examples demonstrating SDK usage.

## 🎯 Available Examples

### **Translator Examples**
- `translate-tflite/` - TensorFlow Lite translation example
- `translate-onnx/` - ONNX translation example
- `translate-custom/` - Custom format translation
- `translate-batch/` - Batch translation processing

### **Backend Examples**
- `backend-cpu/` - CPU backend execution
- `backend-gpu/` - GPU backend execution
- `backend-custom/` - Custom hardware backend
- `backend-multi/` - Multi-backend execution

### **Complete Workflows**
- `pack-and-run/` - Package and run a model
- `optimize-model/` - Optimize a model for specific hardware
- `benchmark/` - Performance benchmarking

## 🚀 Running Examples

```bash
# Run a translator example
cargo run --example translate-tflite

# Run a backend example
cargo run --example backend-cpu

# Run a complete workflow
cargo run --example pack-and-run
```

## 📚 Learning Path

### **Beginner**
1. Start with `translate-tflite` to understand basic translation
2. Try `backend-cpu` to see basic execution
3. Explore `pack-and-run` for complete workflow

### **Intermediate**
1. Study `translate-custom` for building translators
2. Review `backend-custom` for building backends
3. Try `optimize-model` for performance tuning

### **Advanced**
1. Implement new translator in `translate-custom`
2. Build custom backend in `backend-custom`
3. Contribute back to SDK

## 🎯 Example Structure

Each example includes:

- **README.md** - Example documentation
- **src/main.rs** - Main example code
- **fixtures/** - Sample model files
- **tests/** - Example tests

## 📖 Documentation

- [Getting Started Guide](../docs/getting-started.md)
- [Translator API Reference](../docs/translator-api.md)
- [Backend API Reference](../docs/backend-api.md)

## 🤝 Contributing

Have a useful example? Submit it via pull request!

