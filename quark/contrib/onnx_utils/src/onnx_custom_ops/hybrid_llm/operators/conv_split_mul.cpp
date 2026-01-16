// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "conv_split_mul.hpp"

using namespace ryzenai::onnx_utils;

ConvSplitMulKernel::ConvSplitMulKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_kernel_ = std::make_unique<AMDConvSplitMulKernel>(info, session_configs);
#else
  throw std::runtime_error{"NPU must be enabled to use Conv"};
#endif
}

ConvSplitMulKernel::~ConvSplitMulKernel() {}

void ConvSplitMulKernel::Compute(OrtKernelContext* context) {
  if (getBackend(Ort::KernelContext(context)) != Backend::Npu) {
    throw std::runtime_error{"NPU backend must be used for ConvSplitMul"};
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_kernel_->Compute(context);
#endif
}
