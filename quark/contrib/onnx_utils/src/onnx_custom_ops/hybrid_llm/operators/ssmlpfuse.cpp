// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "ssmlpfuse.hpp"

#include "../npu/ssmlpfuse.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

SSMlpFuseKernel::SSMlpFuseKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_ssmlpfuse_ = std::make_unique<AMDSSMlpFuseKernel>(info, session_configs);
#else
  throw std::runtime_error{"NPU must be enabled to use SSMlpFuse"};
#endif
}

SSMlpFuseKernel::~SSMlpFuseKernel() {}

void SSMlpFuseKernel::Compute(OrtKernelContext* context) {
  if (getBackend(Ort::KernelContext(context)) != Backend::Npu) {
    throw std::runtime_error{"NPU backend must be used for SSMlpFuse"};
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  npu_ssmlpfuse_->Compute(context);
#endif
}

}  // namespace ryzenai::onnx_utils
