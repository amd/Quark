// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

class SkipSimplifiedLayerNormKernel : public HybridKernel {
 public:
  SkipSimplifiedLayerNormKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SkipSimplifiedLayerNormKernel();

  void Compute(OrtKernelContext* context);

 private:
  float epsilon_ = 0.0f;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // std::unique_ptr<AMD npu Kernel> npu_instance_;
#endif
};

static const char kSkipSimplifiedLayerNorm[] =
  "SkipSimplifiedLayerNormalization";

struct SkipSimplifiedLayerNorm
  : Operator<SkipSimplifiedLayerNormKernel, kSkipSimplifiedLayerNorm> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils
