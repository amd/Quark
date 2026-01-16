// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MAIN
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MAIN

#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

std::vector<const OrtCustomOp*> create_hybrid_llm_ops(
  const Ort::ConstSessionOptions& options
);

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MAIN
