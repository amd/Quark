// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "corelib_kernel.hpp"

#include <ryzenai/corelib.h>

#include "../../corelib/corelib.hpp"  // for create_corelib_ops

namespace ryzenai::onnx_utils {

CoreLibKernel::CoreLibKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info, const std::string& name
)
  : customOp_(FindCustomOpByName(name)),
    kernel_(customOp_->CreateKernel(customOp_, &ort_api, info)) {}

CoreLibKernel::~CoreLibKernel() noexcept {
  if (customOp_ && kernel_) customOp_->KernelDestroy(kernel_);
}

void CoreLibKernel::Compute(OrtKernelContext* context) {
  customOp_->KernelCompute(kernel_, context);
}

const OrtCustomOp* CoreLibKernel::FindCustomOpByName(const std::string& name) {
  for (const OrtCustomOp* op : create_corelib_ops({}))
    if (name == op->GetName(op)) return op;

  throw std::runtime_error{"CoreLib Custom Op " + name + " not found"};
}
}  // namespace ryzenai::onnx_utils
