// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_REDUCE_SUM
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_REDUCE_SUM

#include "hybrid_kernel.hpp"
#include "opUtils.h"
#include "operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

class ReduceSumKernel : public HybridKernel {
 public:
  ReduceSumKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~ReduceSumKernel();

  void Compute(OrtKernelContext* context);

 private:
  int64_t keepdims_attr_ = 1;
  int64_t noop_with_empty_axes_attr_ = 0;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // std::unique_ptr<AMD npu Kernel> npu_instance_;
#endif
};

static const char kReduceSum[] = "ReduceSum";

struct ReduceSum : Operator<ReduceSumKernel, kReduceSum> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_REDUCE_SUM
