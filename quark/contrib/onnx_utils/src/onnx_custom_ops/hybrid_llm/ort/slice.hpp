// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <ryzenai/ryzen_mm.h>

#include "cast.hpp"
#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtSlice : public OrtOperator {
 public:
  void construct(const Ort::ConstKernelInfo& info);
  void execute(
    Ort::ConstValue inp, Ort::ConstValue starts, Ort::ConstValue ends,
    Ort::ConstValue axes, Ort::UnownedValue out, OrtKernelContext* context
  );
};

}  // namespace ryzenai::onnx_utils
