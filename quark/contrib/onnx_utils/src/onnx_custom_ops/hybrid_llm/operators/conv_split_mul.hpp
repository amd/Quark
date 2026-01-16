// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "opUtils.h"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/conv_split_mul.hpp"
#endif

namespace ryzenai::onnx_utils {
class ConvSplitMulKernel : public HybridKernel {
 public:
  ConvSplitMulKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~ConvSplitMulKernel();

  void Compute(OrtKernelContext* context);

 private:
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDConvSplitMulKernel> npu_kernel_;
#endif
};

static const char kConvSplitMul[] = "ConvSplitMul";

struct ConvSplitMul : HybridOperator<ConvSplitMulKernel, kConvSplitMul> {
  explicit ConvSplitMul(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<ConvSplitMulKernel, kConvSplitMul>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const { return {}; }
};

}  // namespace ryzenai::onnx_utils
