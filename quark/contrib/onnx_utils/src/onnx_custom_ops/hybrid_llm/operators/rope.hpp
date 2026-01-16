// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_ROPE
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_ROPE

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

class RotaryEmbeddingKernel : public HybridKernel {
 public:
  RotaryEmbeddingKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~RotaryEmbeddingKernel();

  void Compute(OrtKernelContext* context);

 private:
  bool interleaved_;
  // int64_t batch_size_;
  // int64_t sequence_length_;
  // int64_t num_heads_;
  // int64_t head_size_;
};

static const char kRotaryEmbedding[] = "RotaryEmbedding";

struct RotaryEmbedding : Operator<RotaryEmbeddingKernel, kRotaryEmbedding> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_ROPE
