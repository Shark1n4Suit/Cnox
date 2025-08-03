//! OpenVINO FFI bindings

#[cfg(feature = "openvino")]
include!(concat!(env!("OUT_DIR"), "/openvino_bindings.rs"));

#[cfg(not(feature = "openvino"))]
// Stub types when OpenVINO feature is disabled
pub type ov_core_t = *mut std::ffi::c_void;
pub type ov_model_t = *mut std::ffi::c_void;
pub type ov_compiled_model_t = *mut std::ffi::c_void;
pub type ov_infer_request_t = *mut std::ffi::c_void;
pub type ov_tensor_t = *mut std::ffi::c_void;
pub type ov_element_type_e = u32;
pub type ov_status_e = u32;
pub type ov_shape_t = *mut std::ffi::c_void;

// Stub function declarations for when OpenVINO is not available
#[cfg(not(feature = "openvino"))]
extern "C" {
    pub fn ov_core_create(_core: *mut *mut ov_core_t) -> ov_status_e { 0 }
    pub fn ov_core_free(_core: *mut ov_core_t) -> ov_status_e { 0 }
    pub fn ov_core_read_model_from_memory(
        _core: *mut ov_core_t,
        _model: *const i8,
        _size: usize,
        _model_ptr: *mut *mut ov_model_t
    ) -> ov_status_e { 0 }
    pub fn ov_core_compile_model(
        _core: *mut ov_core_t,
        _model: *mut ov_model_t,
        _device: *const i8,
        _compiled_model: *mut *mut ov_compiled_model_t
    ) -> ov_status_e { 0 }
    pub fn ov_model_free(_model: *mut ov_model_t) -> ov_status_e { 0 }
    pub fn ov_compiled_model_free(_compiled_model: *mut ov_compiled_model_t) -> ov_status_e { 0 }
    pub fn ov_compiled_model_create_infer_request(
        _compiled_model: *mut ov_compiled_model_t,
        _infer_request: *mut *mut ov_infer_request_t
    ) -> ov_status_e { 0 }
    pub fn ov_infer_request_free(_infer_request: *mut ov_infer_request_t) -> ov_status_e { 0 }
    pub fn ov_infer_request_get_input_tensor_by_index(
        _infer_request: *mut ov_infer_request_t,
        _index: usize,
        _tensor: *mut *mut ov_tensor_t
    ) -> ov_status_e { 0 }
    pub fn ov_infer_request_get_output_tensor_by_index(
        _infer_request: *mut ov_infer_request_t,
        _index: usize,
        _tensor: *mut *mut ov_tensor_t
    ) -> ov_status_e { 0 }
    pub fn ov_infer_request_infer(_infer_request: *mut ov_infer_request_t) -> ov_status_e { 0 }
    pub fn ov_tensor_data(_tensor: *mut ov_tensor_t, _data: *mut *mut std::ffi::c_void) -> ov_status_e { 0 }
    pub fn ov_tensor_get_shape(_tensor: *mut ov_tensor_t, _shape: *mut ov_shape_t) -> ov_status_e { 0 }
} 