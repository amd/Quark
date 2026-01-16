// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::RyzenMM {
struct Allocator;
}

namespace ryzenai::onnx_utils {

class OrtSimplifiedLayerNorm : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info);
  template <typename Wts, typename Out>
  void execute(
    Out* output_data, Ort::BFloat16_t* input_data,
    const std::vector<int64_t>& input_shape, Wts* weights_data,
    const std::vector<int64_t>& weights_shape,
    const RyzenMM::Allocator& allocator, OrtKernelContext* context
  );
};

}  // namespace ryzenai::onnx_utils
