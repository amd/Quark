// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {
struct CoreLibKernel {
  CoreLibKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info, const std::string& name
  );
  ~CoreLibKernel() noexcept;

  void Compute(OrtKernelContext* context);

 private:
  static const OrtCustomOp* FindCustomOpByName(const std::string& name);

 private:
  const OrtCustomOp* const customOp_ = nullptr;
  void* const kernel_ = nullptr;
};
}  // namespace ryzenai::onnx_utils
