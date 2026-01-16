// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <ops/liquidai/conv1d/conv1d.hpp>

#include "../ort/concat.hpp"
#include "../ort/conv.hpp"
#include "../ort/slice.hpp"
#include "../ort/split.hpp"
#include "../ort/transpose.hpp"
#include "npu_op.hpp"
#include "ops/elwmul/elwmul.hpp"

namespace ryzenai::onnx_utils {
class AMDConvSplitMulKernel : public NpuOp {
 public:
  AMDConvSplitMulKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDConvSplitMulKernel();

  void Compute(OrtKernelContext* context);

  void initializeKernels() override {}
  void UpdateSharedBuffer(size_t kernel_size) override {}
  void ConvNpu(
    uint16_t* inp_bf16_ptr, uint16_t* wts_bf16_ptr, uint16_t* out_bf16_ptr,
    std::vector<int64_t> inp_dims, std::vector<int64_t> wts_dims,
    std::vector<int64_t> out_dims, bool init_weights
  );
  void ConvCpu(
    Ort::MemoryInfo const& cpu_mem_info, uint16_t* inp_bf16_ptr,
    uint16_t* wts_bf16_ptr, uint16_t* conv_out_ptr,
    std::vector<int64_t> inp_dims, std::vector<int64_t> wts_dims,
    std::vector<int64_t> out_dims, OrtKernelContext* context
  );
  void initBufBos(size_t seq_len);

 private:
  struct State {
    int instances__ = 0;
    int seq_len_ = 0;
    int run_instances__ = 0;
    std::vector<int64_t> concat_axis_;
    std::vector<int64_t> split_axis_;
    int64_t kw_{3};
    RyzenMM::BufferRef wts_bf16_;
    OrtTranspose<float> ort_trans_;
    OrtConv ort_conv_;
    // RyzenMM::NPUAllocator<'CSM'> allocator_;
    // SharedBuffer::Client shared_buffer_;
    std::unique_ptr<
      ryzenai::liquidai::conv1d<std::uint16_t, std::uint16_t, std::uint16_t>>
      dd_conv_;
    std::shared_ptr<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>> elwmul_{
      nullptr
    };
    xrt::bo mul0_bo0_, mul0_bo1_, mul0_bo2_;
    xrt::bo mul1_bo0_, mul1_bo1_;
    xrt::bo conv_in0_bo_, conv_wts_bo_, conv_out_bo_;
    OrtConcat ort_concat_;
    OrtSplit ort_split_;
    OrtSlice ort_slice_;
    uint16_t* split0_ptr_;
    uint16_t* split1_ptr_;
    uint16_t* split2_ptr_;
    uint16_t* conv_in_ptr_;
    float* transpose_ptr_;
    uint8_t* total_buf_{nullptr};
    size_t split0_sz, split1_sz, split2_sz;
    bool inline_concat_en_ = true;
  };

  SessionState<State> ss_;
};
}  // namespace ryzenai::onnx_utils
