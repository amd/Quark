// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <ryzenai/ryzen_mm.h>

#include "cast.hpp"
#include "onnxruntime_cxx_api.h"
#include "ort_operator.hpp"

namespace ryzenai::onnx_utils {

class OrtSplit : public OrtOperator {
 public:
  void construct(
    const Ort::ConstKernelInfo& info, int64_t axis, int64_t num_outputs
  );
  void execute(
    const Ort::ConstValue& inp, const Ort::ConstValue& split_sizes,
    Ort::UnownedValue& out0, Ort::UnownedValue& out1, Ort::UnownedValue& out2,
    OrtKernelContext* context
  );
};

}  // namespace ryzenai::onnx_utils
