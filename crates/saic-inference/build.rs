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
    
    // Check if OpenVINO library exists
    let library_paths = vec![
        "/usr/lib/x86_64-linux-gnu/libopenvino_c.so",
        "/usr/local/lib/libopenvino_c.so",
        "/opt/intel/openvino/lib/libopenvino_c.so",
    ];
    
    let library_exists = library_paths.iter().any(|path| std::path::Path::new(path).exists());
    
    if !library_exists {
        // Create stub bindings if library not found
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
// Stub OpenVINO bindings - library not found
pub type ov_core_t = *mut std::ffi::c_void;
pub type ov_model_t = *mut std::ffi::c_void;
pub type ov_compiled_model_t = *mut std::ffi::c_void;
pub type ov_infer_request_t = *mut std::ffi::c_void;
pub type ov_tensor_t = *mut std::ffi::c_void;
pub type ov_element_type_e = u32;
pub type ov_status_e = u32;
pub type ov_shape_t = *mut std::ffi::c_void;

// Constants
pub const ov_status_e_OK: ov_status_e = 0;
pub const ov_element_type_e_F32: ov_element_type_e = 1;

// Stub function implementations (return success)
#[no_mangle]
pub extern "C" fn ov_core_create(_core: *mut *mut ov_core_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_core_free(_core: *mut ov_core_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_core_read_model_from_memory(
    _core: *mut ov_core_t,
    _model: *const i8,
    _size: usize,
    _model_ptr: *mut *mut ov_model_t
) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_core_compile_model(
    _core: *mut ov_core_t,
    _model: *mut ov_model_t,
    _device: *const i8,
    _compiled_model: *mut *mut ov_compiled_model_t
) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_model_free(_model: *mut ov_model_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_compiled_model_free(_compiled_model: *mut ov_compiled_model_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_compiled_model_create_infer_request(
    _compiled_model: *mut ov_compiled_model_t,
    _infer_request: *mut *mut ov_infer_request_t
) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_infer_request_free(_infer_request: *mut ov_infer_request_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_infer_request_get_input_tensor_by_index(
    _infer_request: *mut ov_infer_request_t,
    _index: usize,
    _tensor: *mut *mut ov_tensor_t
) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_infer_request_get_output_tensor_by_index(
    _infer_request: *mut ov_infer_request_t,
    _index: usize,
    _tensor: *mut *mut ov_tensor_t
) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_infer_request_infer(_infer_request: *mut ov_infer_request_t) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_tensor_data(_tensor: *mut ov_tensor_t, _data: *mut *mut std::ffi::c_void) -> ov_status_e {
    ov_status_e_OK
}

#[no_mangle]
pub extern "C" fn ov_tensor_get_shape(_tensor: *mut ov_tensor_t, _shape: *mut ov_shape_t) -> ov_status_e {
    ov_status_e_OK
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