#!/bin/bash

# SAIC Build Script
# Builds both packager and runner with all features

set -e

echo "🚀 Building SAIC (Secure AI Container)"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Rust is installed
check_rust() {
    print_status "Checking Rust installation..."
    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed. Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
        print_success "Rust installed successfully"
    else
        print_success "Rust is already installed: $(rustc --version)"
    fi
}

# Build packager
build_packager() {
    print_status "Building SAIC Packager..."
    cd packager
    
    # Clean previous builds
    cargo clean
    
    # Build with release optimizations
    cargo build --release
    
    if [ $? -eq 0 ]; then
        print_success "Packager built successfully"
        ls -la target/release/saic-packager
    else
        print_error "Packager build failed"
        exit 1
    fi
    
    cd ..
}

# Build runner
build_runner() {
    print_status "Building SAIC Runner..."
    cd runner
    
    # Clean previous builds
    cargo clean
    
    # Build with different feature combinations
    print_status "Building with CPU backend..."
    cargo build --release --features cpu-backend
    
    print_status "Building with GPU backend..."
    cargo build --release --features gpu-backend
    
    print_status "Building with full inference..."
    cargo build --release --features full-inference
    
    if [ $? -eq 0 ]; then
        print_success "Runner built successfully"
        ls -la target/release/saic-runner
    else
        print_error "Runner build failed"
        exit 1
    fi
    
    cd ..
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    cd packager
    cargo test
    cd ..
    
    cd runner
    cargo test
    cd ..
    
    print_success "All tests passed"
}

# Create release directory
create_release() {
    print_status "Creating release directory..."
    
    RELEASE_DIR="release"
    mkdir -p $RELEASE_DIR
    
    # Copy binaries
    cp packager/target/release/saic-packager $RELEASE_DIR/
    cp runner/target/release/saic-runner $RELEASE_DIR/
    
    # Copy documentation
    cp README.md $RELEASE_DIR/
    cp QUICKSTART.md $RELEASE_DIR/
    cp INSTALL.md $RELEASE_DIR/
    cp CHANGELOG.md $RELEASE_DIR/
    cp ROADMAP.md $RELEASE_DIR/
    cp FOUNDER_NOTE.md $RELEASE_DIR/
    
    # Copy example files
    cp mobilenetv2.saic $RELEASE_DIR/
    cp saic.yaml $RELEASE_DIR/
    
    # Create installation script
    cat > $RELEASE_DIR/install.sh << 'EOF'
#!/bin/bash

# SAIC Installation Script
echo "Installing SAIC..."

# Create bin directory if it doesn't exist
sudo mkdir -p /usr/local/bin

# Copy binaries
sudo cp saic-packager /usr/local/bin/
sudo cp saic-runner /usr/local/bin/

# Make executable
sudo chmod +x /usr/local/bin/saic-packager
sudo chmod +x /usr/local/bin/saic-runner

echo "SAIC installed successfully!"
echo "Try: saic-runner --help"
EOF
    
    chmod +x $RELEASE_DIR/install.sh
    
    print_success "Release directory created: $RELEASE_DIR"
    ls -la $RELEASE_DIR/
}

# Main build process
main() {
    echo "Starting SAIC build process..."
    
    # Check and install Rust if needed
    check_rust
    
    # Build components
    build_packager
    build_runner
    
    # Run tests
    run_tests
    
    # Create release
    create_release
    
    print_success "SAIC build completed successfully!"
    print_status "Release files are in the 'release' directory"
    print_status "Run './release/install.sh' to install SAIC"
}

# Run main function
main "$@" 