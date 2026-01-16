// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "opUtils.h"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/conv.hpp"
#endif

namespace ryzenai::onnx_utils {
class ConvKernel : public HybridKernel {
 public:
  ConvKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~ConvKernel();

  void Compute(OrtKernelContext* context);

 private:
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDConvKernel> npu_conv_;
#endif
};

static const char kConv[] = "Conv";

struct Conv : HybridOperator<ConvKernel, kConv> {
  explicit Conv(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<ConvKernel, kConv>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const { return {}; }
};

}  // namespace ryzenai::onnx_utils
