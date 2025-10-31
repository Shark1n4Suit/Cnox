# COELANOX SDK Templates

Templates for quickly creating translators and backends.

## 🎯 Available Templates

### **Translator Templates**
- `translator-basic/` - Basic translator template
- `translator-tflite/` - TensorFlow Lite translator
- `translator-pytorch/` - PyTorch JIT translator
- `translator-onnx/` - ONNX translator

### **Backend Templates**
- `backend-basic/` - Basic CPU backend
- `backend-gpu/` - GPU backend template
- `backend-npu/` - NPU backend template
- `backend-custom/` - Custom hardware backend

## 📋 Usage

### **Generate a New Translator**

```bash
# Basic translator
cargo generate --path templates/translator-basic --name my-format-translator

# TensorFlow Lite translator
cargo generate --path templates/translator-tflite --name tflite-translator

# PyTorch translator
cargo generate --path templates/translator-pytorch --name pytorch-translator
```

### **Generate a New Backend**

```bash
# Basic backend
cargo generate --path templates/backend-basic --name my-hardware-backend

# GPU backend
cargo generate --path templates/backend-gpu --name gpu-backend

# NPU backend
cargo generate --path templates/backend-npu --name npu-backend
```

## 🏗️ Template Structure

Each template includes:

- **Cargo.toml** - Project configuration
- **src/lib.rs** - Plugin trait implementation
- **src/parser.rs** - Format parsing logic (translators)
- **src/executor.rs** - Hardware execution (backends)
- **tests/** - Test fixtures and test cases
- **README.md** - Template documentation
- **Examples/** - Working examples

## 🎯 Next Steps

After generating from a template:

1. Implement the specific trait methods
2. Add your format/hardware logic
3. Write tests
4. Document your implementation
5. Submit for review

## 📚 Documentation

- [Getting Started](../docs/getting-started.md)
- [Translator API Reference](../docs/translator-api.md)
- [Backend API Reference](../docs/backend-api.md)
- [Best Practices](../docs/best-practices.md)

## 🤝 Contributing

Feel free to create new templates for additional formats or hardware!

