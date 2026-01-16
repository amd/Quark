// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

std::vector<const OrtCustomOp*> create_corelib_ops(
  const Ort::ConstSessionOptions& options
);

}  // namespace ryzenai::onnx_utils
