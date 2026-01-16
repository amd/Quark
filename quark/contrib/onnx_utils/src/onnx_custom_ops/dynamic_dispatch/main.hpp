// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_MAIN
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_MAIN

#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

std::vector<const OrtCustomOp*> create_dynamic_dispatch_ops(
  const Ort::ConstSessionOptions& options
);

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_MAIN
