// OpenVINO C API headers
#ifdef OPENVINO_FEATURE
#include <openvino/c/ov_core.h>
#include <openvino/c/ov_model.h>
#include <openvino/c/ov_compiled_model.h>
#include <openvino/c/ov_infer_request.h>
#include <openvino/c/ov_tensor.h>
#include <openvino/c/ov_common.h>
#endif

// TensorRT headers
#ifdef TENSORRT_FEATURE
#include <NvInfer.h>
#endif 