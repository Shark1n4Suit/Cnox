# SAIC Installation Guide

**Fast, Secure, Small** - The next-generation AI deployment standard

## Quick Install

### Linux/macOS
```bash
# Download and install SAIC
curl -fsSL https://install.coelanox.com/saic | sh

# Verify installation
saic-runner --version
saic-packager --version
```

### Windows
```powershell
# Download SAIC for Windows
Invoke-WebRequest -Uri "https://install.coelanox.com/saic-windows" -OutFile "saic-install.ps1"
.\saic-install.ps1

# Verify installation
saic-runner --version
saic-packager --version
```

## Manual Installation

### Prerequisites

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential pkg-config libssl-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install openssl-devel

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install rust pkg-config openssl

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Windows
```powershell
# Install Rust (if not already installed)
winget install Rust.Rust
# or visit https://rustup.rs/

# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/coelanox/saic.git
cd saic

# Build the packager
cd packager
cargo build --release

# Build the runner
cd ../runner
cargo build --release

# Install to system (optional)
sudo cp target/release/saic-packager /usr/local/bin/
sudo cp target/release/saic-runner /usr/local/bin/
```

## Backend Dependencies

### CPU Backend (OpenVINO)
```bash
# Linux
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list
sudo apt update
sudo apt install intel-openvino-dev-ubuntu20-2023.1.0

# macOS
brew install openvino

# Windows
# Download from https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
```

### GPU Backend (TensorRT)
```bash
# Linux
# Download from https://developer.nvidia.com/tensorrt
# Follow NVIDIA's installation guide for your specific CUDA version

# Windows
# Download from https://developer.nvidia.com/tensorrt
# Follow NVIDIA's installation guide
```

## Configuration

### Create Configuration File
```bash
# Generate sample configuration
saic-runner config saic.yaml

# Edit configuration
nano saic.yaml
```

### Sample Configuration
```yaml
# SAIC Configuration File
audit_mode: false
log_level: "info"
default_backend: "auto"
performance_mode: true
max_model_size: 0
enable_caching: true
cache_dir: "/tmp/saic-cache"
```

## Verification

### Test Installation
```bash
# Test packager
saic-packager --help

# Test runner
saic-runner --help

# Test with example model
saic-runner run mobilenetv2.saic

# Test execution
saic-runner execute mobilenetv2.saic --backend cpu
```

### Performance Test
```bash
# Quick performance validation
time saic-runner run mobilenetv2.saic
# Should complete in <1ms
```

## Troubleshooting

### Common Issues

#### Build Errors
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

#### Backend Not Found
```bash
# Check backend availability
saic-runner backends

# Install missing dependencies
# See Backend Dependencies section above
```

#### Permission Errors
```bash
# Fix permissions
chmod +x /usr/local/bin/saic-*
```

### Getting Help

- **Documentation**: https://docs.coelanox.com/saic
- **Issues**: https://github.com/coelanox/saic/issues
- **Discussions**: https://github.com/coelanox/saic/discussions

## Next Steps

1. **Read the Documentation**: Start with the [README.md](README.md)
2. **Try the Examples**: Use the provided `mobilenetv2.saic` file
3. **Package Your Models**: Use `saic-packager` to create your own SAIC files
4. **Join the Community**: Contribute to the project and share your experience

---

**SAIC - The Universal AI Binary Container** 🚀 