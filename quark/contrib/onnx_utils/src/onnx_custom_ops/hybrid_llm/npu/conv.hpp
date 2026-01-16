// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <ops/liquidai/conv1d/conv1d.hpp>

#include "../ort/conv.hpp"
#include "../ort/transpose.hpp"
#include "npu_op.hpp"

namespace ryzenai::onnx_utils {
class AMDConvKernel : public NpuOp {
 public:
  AMDConvKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDConvKernel();

  void Compute(OrtKernelContext* context);

  void initializeKernels() override {}
  void UpdateSharedBuffer(size_t kernel_size) override {}

 private:
  RyzenMM::NPUAllocator<'NCNV'> allocator_;
  std::unique_ptr<
    ryzenai::liquidai::conv1d<std::uint16_t, std::uint16_t, std::uint16_t>>
    dd_op_;
  int64_t kw_{3};
  RyzenMM::BufferRef wts_bf16_;
  OrtTranspose<float> transpose_;
  OrtConv cpu_op_;
  SharedBuffer::Client shared_buffer_token_;
};
}  // namespace ryzenai::onnx_utils
