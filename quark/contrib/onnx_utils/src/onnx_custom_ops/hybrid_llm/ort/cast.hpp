// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

template <typename kFrom, typename kTo>
class OrtCast : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info);

  void execute(
    kTo* output_data, kFrom* input_data,
    const std::vector<int64_t>& input_shape, OrtKernelContext* context
  );
  void execute(
    kTo* output_data, const Ort::ConstValue& input_data,
    OrtKernelContext* context
  );
};

}  // namespace ryzenai::onnx_utils
