// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

template <typename T>
class OrtTranspose : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info, std::vector<int64_t> perm);
  void execute(
    T* output_data, T* input_data, const std::vector<int64_t>& input_shape,
    OrtKernelContext* context
  );

 private:
  std::vector<int64_t> perm_;
};

}  // namespace ryzenai::onnx_utils
