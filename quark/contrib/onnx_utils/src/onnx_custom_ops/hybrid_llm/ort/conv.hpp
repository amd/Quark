// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <ryzenai/ryzen_mm.h>

#include "cast.hpp"
#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtConv : public OrtOperator {
 public:
  void construct(
    const Ort::ConstKernelInfo& info, int64_t group,
    std::vector<int64_t> kernel_shape, std::vector<int64_t> pads
  );
  void execute(
    Ort::ConstValue inp, Ort::ConstValue wts, Ort::UnownedValue out,
    OrtKernelContext* context
  );

 private:
  int64_t group_ = 2048;
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
};

}  // namespace ryzenai::onnx_utils
