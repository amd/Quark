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
#include "../npu/ssmlp.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace ryzenai::onnx_utils {

class SSMlpKernel : public HybridKernel {
 public:
  SSMlpKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SSMlpKernel();

  void Compute(OrtKernelContext* context);

 private:
  float epsilon_ = 0.0f;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDSSMLPKernel> npu_instance_;
#endif
};

static const char kSSMlp[] = "SSMLP";

struct SSMlp : HybridOperator<SSMlpKernel, kSSMlp> {
  explicit SSMlp(const Ort::ConstSessionOptions& session_options)
    : HybridOperator(session_options, GetSessionConfigKeys()) {}
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    // if gate/up is fused, up becomes optional
    if (index == 14)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    else
      return INPUT_OUTPUT_REQUIRED;
  }
  virtual size_t GetInputTypeCount() const noexcept { return 16; }

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"hybrid_opt_execution_mode"};
  }
};

}  // namespace ryzenai::onnx_utils
