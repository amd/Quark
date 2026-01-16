// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_CAST
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_CAST

#include "opUtils.h"
#include "operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace ryzenai::onnx_utils {

class CastKernel {
 public:
  CastKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~CastKernel();

  void Compute(OrtKernelContext* context);

 private:
  OrtApi ort_{};
  std::string node_name_;
  int64_t to_attr_;
  int64_t saturate_attr_;
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif
};

static const char kCast[] = "Cast";

struct Cast : Operator<CastKernel, kCast> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_Cast
