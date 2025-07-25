#!/bin/bash

# Simple SAIC Build Test Script
# Tests basic build functionality

echo "🧪 Testing SAIC Build System"
echo "============================"

# Check if we're in the right directory
if [ ! -f "packager/Cargo.toml" ] || [ ! -f "runner/Cargo.toml" ]; then
    echo "❌ Error: Not in SAIC project directory"
    echo "Please run this script from the SAIC project root"
    exit 1
fi

# Check if Rust is available
if ! command -v rustc &> /dev/null; then
    echo "❌ Error: Rust is not installed"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "✅ Rust found: $(rustc --version)"

# Test packager build
echo "🔨 Testing packager build..."
cd packager
if cargo check; then
    echo "✅ Packager check passed"
else
    echo "❌ Packager check failed"
    exit 1
fi
cd ..

# Test runner build
echo "🔨 Testing runner build..."
cd runner
if cargo check --features cpu-backend; then
    echo "✅ Runner check passed (CPU backend)"
else
    echo "❌ Runner check failed (CPU backend)"
    exit 1
fi
cd ..

echo "🎉 All build tests passed!"
echo "SAIC is ready for compilation" 