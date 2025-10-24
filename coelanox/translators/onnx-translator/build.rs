fn main() {
    // For now, we'll use a pre-generated approach to avoid protoc dependency
    // In production, you would use prost-build with protoc installed
    println!("cargo:rerun-if-changed=schemas/onnx.proto");
}