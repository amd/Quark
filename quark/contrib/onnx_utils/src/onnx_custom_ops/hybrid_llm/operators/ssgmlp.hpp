// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/ssgmlp.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace ryzenai::onnx_utils {

class SSGMlpKernel : public HybridKernel {
 public:
  SSGMlpKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SSGMlpKernel();

  void Compute(OrtKernelContext* context);

 private:
  float epsilon_ = 0.0f;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDSSGMLPKernel> npu_instance_;
#endif
};

static const char kSSGMlp[] = "SSGMLP";

struct SSGMlp : HybridOperator<SSGMlpKernel, kSSGMlp> {
  explicit SSGMlp(const Ort::ConstSessionOptions& session_options)
    : HybridOperator(session_options, GetSessionConfigKeys()) {}
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    // if gate/up is fused, up becomes optional
    if (index == 16)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    else
      return INPUT_OUTPUT_REQUIRED;
  }
  virtual size_t GetInputTypeCount() const noexcept { return 18; }

  std::unordered_set<std::string> GetSessionConfigKeys() const { return {}; }
};

}  // namespace ryzenai::onnx_utils
