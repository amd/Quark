// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <tuple>

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtRotaryEmbedding : public OrtOperator {
 public:
  void construct(
    const Ort::ConstKernelInfo& info, float scale, int64_t rotary_interleaved,
    int64_t num_heads, int64_t rotary_embedding_dim
  );
  void execute(
    float* output_data, float* input_data, int64_t* pos_ids_data,
    const OrtValue* cos_cache, const OrtValue* sin_cache,
    const std::vector<int64_t>& input_shape, OrtKernelContext* context,
    int rewind_pos
  );
};

}  // namespace ryzenai::onnx_utils
