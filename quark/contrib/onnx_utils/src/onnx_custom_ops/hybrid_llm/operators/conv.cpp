// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "conv.hpp"

using namespace ryzenai::onnx_utils;

ConvKernel::ConvKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_conv_ = std::make_unique<AMDConvKernel>(info, session_configs);
#else
  throw std::runtime_error{"NPU must be enabled to use Conv"};
#endif
}

ConvKernel::~ConvKernel() {}

void ConvKernel::Compute(OrtKernelContext* context) {
  if (getBackend(Ort::KernelContext(context)) != Backend::Npu) {
    throw std::runtime_error{"NPU backend must be used for Conv"};
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_conv_->Compute(context);
#endif
}
