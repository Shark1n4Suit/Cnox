//! OpenVINO FFI bindings

#[cfg(feature = "openvino")]
include!(concat!(env!("OUT_DIR"), "/openvino_bindings.rs"));

#[cfg(not(feature = "openvino"))]
// Stub function declarations for when OpenVINO is not available
extern "C" {
    pub fn ov_core_create(_core: *mut *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_core_free(_core: *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_core_read_model_from_memory(
        _core: *mut std::ffi::c_void,
        _model: *const i8,
        _size: usize,
        _model_ptr: *mut *mut std::ffi::c_void
    ) -> u32 { 0 }
    pub fn ov_core_compile_model(
        _core: *mut std::ffi::c_void,
        _model: *mut std::ffi::c_void,
        _device: *const i8,
        _compiled_model: *mut *mut std::ffi::c_void
    ) -> u32 { 0 }
    pub fn ov_model_free(_model: *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_compiled_model_free(_compiled_model: *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_compiled_model_create_infer_request(
        _compiled_model: *mut std::ffi::c_void,
        _infer_request: *mut *mut std::ffi::c_void
    ) -> u32 { 0 }
    pub fn ov_infer_request_free(_infer_request: *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_infer_request_get_input_tensor_by_index(
        _infer_request: *mut std::ffi::c_void,
        _index: usize,
        _tensor: *mut *mut std::ffi::c_void
    ) -> u32 { 0 }
    pub fn ov_infer_request_get_output_tensor_by_index(
        _infer_request: *mut std::ffi::c_void,
        _index: usize,
        _tensor: *mut *mut std::ffi::c_void
    ) -> u32 { 0 }
    pub fn ov_infer_request_infer(_infer_request: *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_tensor_data(_tensor: *mut std::ffi::c_void, _data: *mut *mut std::ffi::c_void) -> u32 { 0 }
    pub fn ov_tensor_get_shape(_tensor: *mut std::ffi::c_void, _shape: *mut std::ffi::c_void) -> u32 { 0 }
}

// Type aliases for when OpenVINO feature is disabled
#[cfg(not(feature = "openvino"))]
pub type ov_core_t = *mut std::ffi::c_void;
#[cfg(not(feature = "openvino"))]
pub type ov_model_t = *mut std::ffi::c_void;
#[cfg(not(feature = "openvino"))]
pub type ov_compiled_model_t = *mut std::ffi::c_void;
#[cfg(not(feature = "openvino"))]
pub type ov_infer_request_t = *mut std::ffi::c_void;
#[cfg(not(feature = "openvino"))]
pub type ov_tensor_t = *mut std::ffi::c_void;
#[cfg(not(feature = "openvino"))]
pub type ov_element_type_e = u32;
#[cfg(not(feature = "openvino"))]
pub type ov_status_e = u32;
#[cfg(not(feature = "openvino"))]
pub type ov_shape_t = *mut std::ffi::c_void; 