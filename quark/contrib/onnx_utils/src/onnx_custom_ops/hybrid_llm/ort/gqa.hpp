// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <tuple>

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtGQA : public OrtOperator {
 public:
  void construct(
    const Ort::ConstKernelInfo& info, int64_t do_rotary, float scale,
    int64_t rotary_interleaved, int64_t num_heads, int64_t kv_num_heads,
    int64_t rotary_embedding_dim, float softcap, int64_t local_window_size,
    bool has_head_sink
  );
  void execute(
    float* output_data, float* present_k_data, float* present_v_data,
    float* q_data, float* k_data, float* v_data, float* past_k_data,
    float* past_v_data, const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& kv_shape,
    const std::vector<int64_t>& past_k_shape,
    const std::vector<int64_t>& past_v_shape,
    const std::vector<int64_t>& present_k_shape,
    const std::vector<int64_t>& present_v_shape, int64_t seq_len,
    const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
    std::vector<float>& head_sink, OrtKernelContext* context
  );

 private:
  bool has_head_sink_ = false;
};

}  // namespace ryzenai::onnx_utils
