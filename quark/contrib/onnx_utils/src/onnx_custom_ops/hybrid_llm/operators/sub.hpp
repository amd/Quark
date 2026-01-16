// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_Sub
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_Sub

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "opUtils.h"
#include "operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

class SubKernel : public HybridKernel {
 public:
  SubKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SubKernel();

  void Compute(OrtKernelContext* context);

 private:
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // NPU implementation if available
#endif
};

static const char kSub[] = "Sub";

struct Sub : Operator<SubKernel, kSub> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_Sub
