// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <vector>

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtMatMul : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info);
  void execute(
    OrtKernelContext* context, float* activation_ptr,
    std::vector<int64_t> act_dim, float* weights_ptr,
    std::vector<int64_t> wts_dim, float* output_ptr,
    std::vector<int64_t> out_dim
  );
};

}  // namespace ryzenai::onnx_utils
