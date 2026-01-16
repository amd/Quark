// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_ONNX
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_ONNX

#include <vector>

#include "onnxruntime_cxx_api.h"
#include "op_fuser/fusion_rt.hpp"

namespace ryzenai::onnx_utils {

const char* type_to_str(ONNXTensorElementDataType type);

::Tensor getInputTensor(
  const Ort::KernelContext& ctx, int index, const std::vector<int64_t>& shape
);
::Tensor getInputTensor(const Ort::KernelContext& ctx, int index);
::Tensor getOutputTensor(
  const Ort::KernelContext& ctx, int index, const std::vector<int64_t>& shape
);

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_ONNX
