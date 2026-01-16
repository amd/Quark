// Copyright (c) 2025 Advanced Micro Devices, Inc.

#include "conv.hpp"

#include "ops/ops_common/dtype_utils.h"
#include "profiling/profiling.hpp"

constexpr const auto use_cpu_conv = false;

using namespace ryzenai::onnx_utils;

AMDConvKernel::AMDConvKernel(
  const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : NpuOp(info, session_configs) {
  dd_op_ = std::make_unique<
    ryzenai::liquidai::conv1d<std::uint16_t, std::uint16_t, std::uint16_t>>(
    "bfloat16", "bfloat16", "bfloat16", false,
    std::map<std::string, std::any>{{"pdi_name", std::string("DPU_7")}}
  );

  Ort::ConstKernelInfo info2(info);

  const auto group = info2.GetAttribute<int64_t>("group");
  auto kernel_shape = info2.GetAttributes<int64_t>("kernel_shape");
  auto pads = info2.GetAttributes<int64_t>("pads");

  kw_ = kernel_shape[0];
  transpose_.construct(info2, {0, 2, 1});
  cpu_op_.construct(info2, group, std::move(kernel_shape), std::move(pads));

  shared_buffer_ = SharedBuffer::Client("Conv", session_id_, name());
  shared_buffer_token_ =
    SharedBuffer::Client("Conv_Token", session_id_, name());
}

AMDConvKernel::~AMDConvKernel() {}

void AMDConvKernel::Compute(OrtKernelContext* context) {
  static const auto cpu_mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::KernelContext ctx(context);
  const auto inp = ctx.GetInput(0);
  const auto inp_dims = inp.GetTensorTypeAndShapeInfo().GetShape();
  const auto out_dims =
    std::vector<int64_t>{inp_dims[0], inp_dims[1] - 3, inp_dims[2]};
  auto out = ctx.GetOutput(0, out_dims);
  const auto wts = ctx.GetInput(1);
  const auto wts_dims = wts.GetTensorTypeAndShapeInfo().GetShape();
  const auto seq_length = out_dims[1];

  /*
  printf("inp - %llu, inp_dims = {%lld, %lld, %lld}\n",
  inp.GetTensorSizeInBytes(), inp_dims[0], inp_dims[1], inp_dims[2]);
  printf("out - %llu, out_dims = {%lld, %lld, %lld}\n",
  out.GetTensorSizeInBytes(), out_dims[0], out_dims[1], out_dims[2]);
  printf("wts - %llu, wts_dims = {%lld, %lld, %lld}\n",
  wts.GetTensorSizeInBytes(), wts_dims[0], wts_dims[1], wts_dims[2]);*/

  // NOTE: NPU Conv 128+256 kernels produce repeated garbage.
  // TODO: enable that kernels and fix.
  if (use_cpu_conv || seq_length <= 256) {
    PROFILING_START(token_share_buf_update)
    const int64_t conv_out_dims[3] = {
      out_dims[0], out_dims[2], out_dims[1] + 1
    };
    const int64_t conv_inp_dims[3] = {inp_dims[0], inp_dims[2], inp_dims[1]};
    const auto conv_inp_len = inp.GetTensorSizeInBytes();
    const auto conv_out_len =
      (conv_out_dims[0] * conv_out_dims[1] * conv_out_dims[2]) * sizeof(float);
    const auto transpose_buffer_len = (std::max)(conv_inp_len, conv_out_len);

    shared_buffer_token_.Update({
      {"transpose", transpose_buffer_len},
      {"conv_out", conv_out_len},
    });
    PROFILING_END(token_share_buf_update, false, name().c_str())

    PROFILING_START(cpu_trans_before)
    auto transpose_buffer_ptr =
      (float*)shared_buffer_token_.Get("transpose").ptr;
    auto conv_out_buffer_ptr = (float*)shared_buffer_token_.Get("conv_out").ptr;

    RecordDuration(Metric::Transpose, [&]() {
      transpose_.execute(
        transpose_buffer_ptr, (float*)inp.GetTensorData<float>(),
        {inp_dims[0], inp_dims[1], inp_dims[2]}, context
      );
    });
    PROFILING_END(cpu_trans_before, false, name().c_str())

    PROFILING_START(cpu_conv)
    RecordDuration(Metric::KernelExecution, [&]() {
      cpu_op_.execute(
        Ort::Value::CreateTensor<float>(
          cpu_mem_info, transpose_buffer_ptr, transpose_buffer_len,
          conv_inp_dims, 3
        )
          .GetConst(),
        wts,
        Ort::Value::CreateTensor<float>(
          cpu_mem_info, conv_out_buffer_ptr, conv_out_len, conv_out_dims, 3
        )
          .GetUnowned(),
        context
      );
    });
    PROFILING_END(cpu_conv, false, name().c_str())

    PROFILING_START(cpu_trans_after)
    RecordDuration(Metric::Transpose, [&]() {
      transpose_.execute(
        transpose_buffer_ptr, conv_out_buffer_ptr,
        {conv_out_dims[0], conv_out_dims[1], conv_out_dims[2]}, context
      );
    });
    PROFILING_END(cpu_trans_after, false, name().c_str())

#if 0  // another slice implementation for not transposed output
    {  // generic Slice(1, max(INT64), 2) that basically just removes the first
       // column
      const auto batch_size = out_dims[0];
      const auto rows = out_dims[1];
      const auto cols = conv_out_dims[2];
      const auto cols_out = out_dims[2];

      const auto src = conv_out_buffer_ptr;
      auto dst = out.GetTensorMutableData<float>();

      for (int64_t batch = 0; batch < batch_size; ++batch) {
        for (int64_t row = 0; row < rows; ++row) {
          const auto src_base = (batch * rows + row) * cols;
          const auto dst_base = (batch * rows + row) * cols_out;

          MemCpy(
            dst + dst_base, src + src_base + 1,
            static_cast<size_t>(cols_out) * sizeof(float)
          );
        }
      }
    }
#endif

    PROFILING_START(cp_cput_out)
    // kinda Slice out first row
    MemCpy(
      out.GetTensorMutableData<float>(), transpose_buffer_ptr + out_dims[2],
      size_t(out_dims[1]) * size_t(out_dims[2]) * sizeof(float)
    );
    PROFILING_END(cp_cput_out, false, name().c_str())

    return;
  }

  const auto init_weights = !wts_bf16_;

  PROFILING_START(init_wts)
  if (init_weights) {
    wts_bf16_ =
      allocator_.AllocateBuffer(wts.GetTensorSizeInBytes() / sizeof(uint16_t));

    RecordDuration(Metric::Casting, [&]() {
      ryzenai::float_buffer_to_bfloat16(
        wts.GetTensorData<float>(),
        wts.GetTensorTypeAndShapeInfo().GetElementCount(),
        wts_bf16_.Data<uint16_t>()
      );
    });
  }
  PROFILING_END(init_wts, false, name().c_str())

  PROFILING_START(share_buf_update)
  const auto inp_bf16_len = inp.GetTensorSizeInBytes() / sizeof(uint16_t);
  const auto out_bf16_len = out.GetTensorSizeInBytes() / sizeof(uint16_t);

  shared_buffer_.Update(
    {{"inp_bf16", inp_bf16_len}, {"out_bf16", out_bf16_len}}
  );
  PROFILING_END(share_buf_update, false, name().c_str())

  auto inp_bf16_ptr = shared_buffer_.Get("inp_bf16").ptr;
  auto out_bf16_ptr = shared_buffer_.Get("out_bf16").ptr;

  PROFILING_START(in_fp_to_bf16)
  RecordDuration(Metric::Casting, [&]() {
    ryzenai::float_buffer_to_bfloat16(
      inp.GetTensorData<float>(),
      inp.GetTensorTypeAndShapeInfo().GetElementCount(), (uint16_t*)inp_bf16_ptr
    );
  });
  PROFILING_END(in_fp_to_bf16, false, name().c_str())

  PROFILING_START(dd_op_exec)
  auto wts_tensors = std::vector<Tensor>{
    {wts_bf16_.Data(),
     {size_t(wts_dims[0]), size_t(wts_dims[1]), size_t(wts_dims[2])},
     "bfloat16"}
  };
  auto inp_tensors = std::vector<Tensor>{
    {inp_bf16_ptr, {size_t(inp_dims[1]), size_t(inp_dims[2])}, "bfloat16"},
    wts_tensors[0]
  };
  auto out_tensors = std::vector<Tensor>{
    {out_bf16_ptr, {size_t(out_dims[1]), size_t(out_dims[2])}, "bfloat16"}
  };

  dd_op_->set_tensor_shape(inp_tensors, out_tensors);

  if (init_weights) dd_op_->initialize_const_params(wts_tensors, {});

  RecordDuration(Metric::KernelExecution, [&]() {
    dd_op_->execute(inp_tensors, out_tensors);
  });
  PROFILING_END(dd_op_exec, false, name().c_str())

  PROFILING_START(out_bf16_to_fp)
  RecordDuration(Metric::Casting, [&]() {
    ryzenai::bfloat16_buffer_to_float(
      (const uint16_t*)out_bf16_ptr, out_bf16_len / sizeof(uint16_t),
      out.GetTensorMutableData<float>()
    );
  });
  PROFILING_END(out_bf16_to_fp, false, name().c_str())
}
