# Development Environment Setup

This guide helps new contributors set up a complete development environment for the SAIC project, including Rust toolchain, backend dependencies, and project configuration.

## Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB+ recommended for GPU development)
- **Storage**: At least 10GB free space for dependencies and models
- **Network**: Stable internet connection for downloading dependencies

## Core Development Tools

### 1. Install Rust

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add Rust to your PATH (or restart your terminal)
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Recommended Rust Version**: 1.75.0 or later

### 2. Install Git

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# macOS (with Homebrew)
brew install git

# Verify installation
git --version
```

### 3. Install Build Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libssl-dev \
    libclang-dev libudev-dev libgcc-s1 libc6-dev
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Windows (WSL2)
```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential cmake pkg-config libssl-dev
```

## Backend Dependencies

### 1. OpenVINO Setup (CPU Backend)

Run the OpenVINO setup script:

```bash
# Clone the SAIC repository
git clone https://github.com/coelanox/saic.git
cd saic

# Make the setup script executable
chmod +x scripts/setup_openvino.sh

# Run the OpenVINO setup
./scripts/setup_openvino.sh
```

The script will:
- Download and install OpenVINO 2023.3
- Set up environment variables
- Install Python dependencies
- Configure the CPU backend

**Verification**:
```bash
# Check OpenVINO installation
python3 -c "import openvino; print(openvino.__version__)"

# Verify environment variables
echo $INTEL_OPENVINO_DIR
```

### 2. TensorRT Setup (GPU Backend)

Run the TensorRT setup script:

```bash
# Make the setup script executable
chmod +x scripts/setup_tensorrt.sh

# Run the TensorRT setup
./scripts/setup_tensorrt.sh
```

The script will:
- Install NVIDIA drivers (if needed)
- Download and install TensorRT 8.6
- Set up CUDA environment
- Configure the GPU backend

**Prerequisites for TensorRT**:
- NVIDIA GPU with compute capability 7.0+
- CUDA 11.8 or 12.0
- cuDNN 8.9

**Verification**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## Project Setup

### 1. Clone and Build

```bash
# Clone the repository
git clone https://github.com/coelanox/saic.git
cd saic

# Install dependencies and build
cargo build --release

# Run tests to verify setup
cargo test
```

### 2. Environment Configuration

Create a development configuration file:

```bash
# Create development config
cp config/development.yaml.example config/development.yaml

# Edit configuration for your environment
nano config/development.yaml
```

Example development configuration:

```yaml
# Development Configuration
development:
  debug: true
  log_level: "debug"
  
backends:
  cpu:
    enabled: true
    backend: "openvino"
    threads: 2
    
  gpu:
    enabled: true
    backend: "tensorrt"
    device_id: 0
    
storage:
  model_cache: "./models"
  temp_directory: "./temp"
  
logging:
  level: "debug"
  format: "text"
  output: "stdout"
```

### 3. Install Development Tools

```bash
# Install Rust development tools
cargo install cargo-watch
cargo install cargo-audit
cargo install cargo-tarpaulin

# Install code formatting and linting
rustup component add rustfmt
rustup component add clippy
```

## IDE Setup

### VS Code (Recommended)

1. **Install VS Code Extensions**:
   - Rust Analyzer
   - CodeLLDB
   - Better TOML
   - YAML

2. **Configure Rust Analyzer**:
   ```json
   {
     "rust-analyzer.server.extraEnv": {
       "INTEL_OPENVINO_DIR": "/opt/intel/openvino_2023.3.0"
     }
   }
   ```

### IntelliJ IDEA / CLion

1. **Install Rust Plugin**
2. **Configure Toolchain**: Point to your Rust installation
3. **Set Environment Variables**: Add OpenVINO and TensorRT paths

## Testing Your Setup

### 1. Run Unit Tests

```bash
# Run all tests
cargo test

# Run specific test modules
cargo test parser
cargo test validator
cargo test backends
```

### 2. Run Integration Tests

```bash
# Run integration tests
cargo test --test integration

# Run with specific backend
cargo test --test integration -- --backend cpu
```

### 3. Test Backend Functionality

```bash
# Test CPU backend
cargo run --bin saic-runner -- test-backend cpu

# Test GPU backend
cargo run --bin saic-runner -- test-backend gpu

# List available backends
cargo run --bin saic-runner -- list-backends
```

### 4. Test with Sample Models

```bash
# Download a test model
curl -O https://models.coelanox.com/examples/mobilenet.saic

# Test model execution
cargo run --bin saic-runner -- execute mobilenet.saic --backend cpu
```

## Development Workflow

### 1. Code Quality Checks

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Run security audit
cargo audit

# Run tests with coverage
cargo tarpaulin --out Html
```

### 2. Building Components

```bash
# Build packager
cargo build --bin saic-packager --release

# Build runner
cargo build --bin saic-runner --release

# Build all components
cargo build --release
```

### 3. Running Development Server

```bash
# Run in development mode
cargo run --bin saic-runner -- daemon --config config/development.yaml

# Run with hot reload (requires cargo-watch)
cargo watch -x 'run --bin saic-runner -- daemon'
```

## Troubleshooting

### Common Issues

1. **OpenVINO not found**
   ```bash
   # Check environment variables
   echo $INTEL_OPENVINO_DIR
   
   # Re-run setup script
   ./scripts/setup_openvino.sh
   ```

2. **TensorRT not found**
   ```bash
   # Check CUDA installation
   nvcc --version
   
   # Check TensorRT paths
   echo $TENSORRT_ROOT
   
   # Re-run setup script
   ./scripts/setup_tensorrt.sh
   ```

3. **Build errors**
   ```bash
   # Clean and rebuild
   cargo clean
   cargo build
   
   # Check Rust version
   rustc --version
   ```

4. **Permission errors**
   ```bash
   # Fix script permissions
   chmod +x scripts/*.sh
   
   # Run with sudo if needed
   sudo ./scripts/setup_openvino.sh
   ```

### Performance Optimization

1. **Enable parallel compilation**:
   ```bash
   # Set number of parallel jobs
   export CARGO_BUILD_JOBS=8
   ```

2. **Use release builds for testing**:
   ```bash
   # Build in release mode
   cargo build --release
   ```

3. **Optimize for your CPU**:
   ```bash
   # Set target CPU
   export RUSTFLAGS="-C target-cpu=native"
   ```

## Next Steps

After setting up your development environment:

1. **Read the Architecture Documentation**: `docs/architecture/README.md`
2. **Review Contributing Guidelines**: `docs/CONTRIBUTING.md`
3. **Join the Community**: Participate in discussions and code reviews
4. **Start Contributing**: Pick up issues labeled "good first issue"

---

**Need Help?** If you encounter issues during setup, please:
1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed error information
4. Join our community discussions 