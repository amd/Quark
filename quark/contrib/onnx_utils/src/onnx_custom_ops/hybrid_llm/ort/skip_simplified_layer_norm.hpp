// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtSkipSimplifiedLayerNorm : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info);
  void execute(
    float* output_1_data, float* output_2_data, float* input_a_data,
    float* input_b_data, const std::vector<int64_t>& input_shape,
    float* weights_data, const std::vector<int64_t>& weights_shape,
    OrtKernelContext* context
  );
};

}  // namespace ryzenai::onnx_utils
