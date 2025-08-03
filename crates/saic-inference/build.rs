use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to rerun this script if the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // Generate OpenVINO bindings if the feature is enabled
    #[cfg(feature = "openvino")]
    {
        generate_openvino_bindings();
    }

    // Generate TensorRT bindings if the feature is enabled
    #[cfg(feature = "tensorrt")]
    {
        generate_tensorrt_bindings();
    }
}

#[cfg(feature = "openvino")]
fn generate_openvino_bindings() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Try to find OpenVINO installation
    let openvino_path = find_openvino_path();
    
    // Check if OpenVINO headers exist
    let header_path = format!("{}/include/openvino/c/ov_core.h", openvino_path);
    if !std::path::Path::new(&header_path).exists() {
        // Create stub bindings if headers not found
        create_stub_openvino_bindings(&out_path);
        return;
    }
    
    // Create bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", openvino_path))
        .clang_arg(format!("-I{}/include/openvino", openvino_path))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate OpenVINO bindings");

    // Write bindings to file
    bindings
        .write_to_file(out_path.join("openvino_bindings.rs"))
        .expect("Couldn't write OpenVINO bindings");
}

#[cfg(feature = "tensorrt")]
fn generate_tensorrt_bindings() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Try to find TensorRT installation
    let tensorrt_path = find_tensorrt_path();
    
    // Check if TensorRT headers exist
    let header_path = format!("{}/include/NvInfer.h", tensorrt_path);
    if !std::path::Path::new(&header_path).exists() {
        // Create stub bindings if headers not found
        create_stub_tensorrt_bindings(&out_path);
        return;
    }
    
    // Create bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", tensorrt_path))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate TensorRT bindings");

    // Write bindings to file
    bindings
        .write_to_file(out_path.join("tensorrt_bindings.rs"))
        .expect("Couldn't write TensorRT bindings");
}

fn create_stub_openvino_bindings(out_path: &PathBuf) {
    let stub_content = r#"
// Stub OpenVINO bindings - headers not found
pub type ov_core_t = *mut std::ffi::c_void;
pub type ov_model_t = *mut std::ffi::c_void;
pub type ov_compiled_model_t = *mut std::ffi::c_void;
pub type ov_infer_request_t = *mut std::ffi::c_void;
pub type ov_tensor_t = *mut std::ffi::c_void;
pub type ov_element_type_e = u32;
pub type ov_status_e = u32;
pub type ov_shape_t = *mut std::ffi::c_void;

// Stub function declarations
#[link(name = "openvino_c")]
extern "C" {
    pub fn ov_core_create(core: *mut *mut ov_core_t) -> ov_status_e;
    pub fn ov_core_free(core: *mut ov_core_t) -> ov_status_e;
    pub fn ov_core_read_model_from_memory(
        core: *mut ov_core_t,
        model: *const i8,
        size: usize,
        model_ptr: *mut *mut ov_model_t
    ) -> ov_status_e;
    pub fn ov_core_compile_model(
        core: *mut ov_core_t,
        model: *mut ov_model_t,
        device: *const i8,
        compiled_model: *mut *mut ov_compiled_model_t
    ) -> ov_status_e;
    pub fn ov_model_free(model: *mut ov_model_t) -> ov_status_e;
    pub fn ov_compiled_model_free(compiled_model: *mut ov_compiled_model_t) -> ov_status_e;
    pub fn ov_compiled_model_create_infer_request(
        compiled_model: *mut ov_compiled_model_t,
        infer_request: *mut *mut ov_infer_request_t
    ) -> ov_status_e;
    pub fn ov_infer_request_free(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub fn ov_infer_request_get_input_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        index: usize,
        tensor: *mut *mut ov_tensor_t
    ) -> ov_status_e;
    pub fn ov_infer_request_get_output_tensor_by_index(
        infer_request: *mut ov_infer_request_t,
        index: usize,
        tensor: *mut *mut ov_tensor_t
    ) -> ov_status_e;
    pub fn ov_infer_request_infer(infer_request: *mut ov_infer_request_t) -> ov_status_e;
    pub fn ov_tensor_data(tensor: *mut ov_tensor_t, data: *mut *mut std::ffi::c_void) -> ov_status_e;
    pub fn ov_tensor_get_shape(tensor: *mut ov_tensor_t, shape: *mut ov_shape_t) -> ov_status_e;
}
"#;
    
    std::fs::write(out_path.join("openvino_bindings.rs"), stub_content)
        .expect("Couldn't write stub OpenVINO bindings");
}

fn create_stub_tensorrt_bindings(out_path: &PathBuf) {
    let stub_content = r#"
// Stub TensorRT bindings - headers not found
pub type nvinfer1_ICudaEngine = *mut std::ffi::c_void;
pub type nvinfer1_IExecutionContext = *mut std::ffi::c_void;
pub type nvinfer1_IRuntime = *mut std::ffi::c_void;
"#;
    
    std::fs::write(out_path.join("tensorrt_bindings.rs"), stub_content)
        .expect("Couldn't write stub TensorRT bindings");
}

fn find_openvino_path() -> String {
    // Try common OpenVINO installation paths
    let possible_paths = vec![
        "/usr/local/openvino",
        "/opt/intel/openvino",
        "/usr/include/openvino",
        "/usr/local/include/openvino",
    ];

    for path in possible_paths {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }

    // Fallback to system include path
    "/usr/include".to_string()
}

fn find_tensorrt_path() -> String {
    // Try common TensorRT installation paths
    let possible_paths = vec![
        "/usr/local/tensorrt",
        "/opt/tensorrt",
        "/usr/include/tensorrt",
        "/usr/local/include/tensorrt",
    ];

    for path in possible_paths {
        if std::path::Path::new(path).exists() {
            return path.to_string();
        }
    }

    // Fallback to system include path
    "/usr/include".to_string()
} 