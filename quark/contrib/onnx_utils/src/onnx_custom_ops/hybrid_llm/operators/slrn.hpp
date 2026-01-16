// Copyright (c) 2025 Advanced Micro Devices, Inc.

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
#include "../npu/slrn.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace ryzenai::onnx_utils {

class SimplifiedLayerNormKernel : public HybridKernel {
 public:
  SimplifiedLayerNormKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SimplifiedLayerNormKernel();

  void Compute(OrtKernelContext* context);
  void run_cpu_reshape(
    OrtKernelContext* context, Ort::ConstValue& input, OrtValue* shape_tensor
  );

 private:
  float epsilon_ = 0.0f;
  const OrtApi* api_;
  Ort::Op reshape1_{nullptr};
  Ort::Op reshape2_{nullptr};
  OrtValue* shape_in_tensor_ = nullptr;
  OrtValue* shape_out_tensor_ = nullptr;
  std::vector<int64_t> shape_in_;
  std::vector<int64_t> shape_out_;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDSLRNKernel> npu_instance_;
#endif
};

static const char kSimplifiedLayerNorm[] = "SLRN";

struct SimplifiedLayerNorm
  : HybridOperator<SimplifiedLayerNormKernel, kSimplifiedLayerNorm> {
  explicit SimplifiedLayerNorm(const Ort::ConstSessionOptions& session_options)
    : HybridOperator(session_options, GetSessionConfigKeys()) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"hybrid_opt_execution_mode"};
  }
};

}  // namespace ryzenai::onnx_utils
