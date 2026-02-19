// Copyright (c) 2025 Advanced Micro Devices, Inc.

#include "conv_split_mul.hpp"

#include <cassert>
#include <vector>

#include "common.hpp"
#include "mladf_version.hpp"
#include "npu_utils.hpp"
#include "ops/ops_common/dtype_utils.h"

constexpr const auto use_cpu_conv = false;

namespace ryzenai::onnx_utils {

namespace {

template <typename T, typename U>
void transpose021WithCast(const T* inp, U* out, int d0, int d1, int d2) {
  for (int i = 0; i < d0; i++) {
    for (int j = 0; j < d1; j++) {
      for (int k = 0; k < d2; k++) {
        int in_idx = i * (d1 * d2) + j * d2 + k;
        int out_idx = i * (d2 * d1) + k * d1 + j;

        if constexpr (std::is_same_v<T, uint16_t> && std::is_same_v<U, float>) {
          out[out_idx] = bfloat16_to_float_single(inp[in_idx]);
        } else if constexpr (std::is_same_v<T, float> &&
                             std::is_same_v<U, uint16_t>) {
          out[out_idx] = float_to_bfloat16(inp[in_idx]);
        } else {
          static_assert(
            !sizeof(T), "Unsupported types to transpose021WithCast"
          );
        }
      }
    }
  }
}

}  // namespace

AMDConvSplitMulKernel::AMDConvSplitMulKernel(
  const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : NpuOp(info, session_configs), ss_(session_configs) {
  if (ss_->instances__ == 0) {
    Ort::ConstKernelInfo info2(info);
    auto header = initializeNpuOp("ConvSplitMul", session_configs, info2);

    std::map<std::string, std::any> attr{
      {"pdi_name", std::string("DPU_7")},
    };
    ss_->dd_conv_ = std::make_unique<
      ryzenai::liquidai::conv1d<std::uint16_t, std::uint16_t, std::uint16_t>>(
      "bfloat16", "bfloat16", "bfloat16", false, attr
    );

    const auto group = info2.GetAttribute<int64_t>("group");
    auto kernel_shape = info2.GetAttributes<int64_t>("kernel_shape");
    std::vector<int64_t> pads;
    try {
      pads = info2.GetAttributes<int64_t>("pads");
    } catch (...) {
      // some models (like LFM2.5-1.2B-Thinking-ONNX) dont have it
    }

    ss_->kw_ = kernel_shape[0];
    ss_->ort_trans_.construct(info2, {0, 2, 1});
    ss_->ort_conv_.construct(
      info2, group, std::move(kernel_shape), std::move(pads)
    );
    ss_->ort_slice_.construct(info2);

    // const auto concat_axis = info2.GetAttributes<int64_t>("concat_axis");
    // const auto split_axis = info2.GetAttributes<int64_t>("split_axis");
    // std::cout << "222 ss_->concat_axis_:" << concat_axis << "
    // ss_->split_axis_:" << split_axis[0] << std::endl;
    // assert(ss_->concat_axis_ == 1);
    // assert(ss_->split_axis_ == 2);
    ss_->ort_split_.construct(info2, 2, 3);  // split_axis[0]);
    ss_->ort_concat_.construct(info2, 1);

    auto attr_mul = attr;
    attr_mul.insert(
      {{"skip_create_input", 1},
       {"skip_create_output", 1},
       {"op_version", mladfVersion().str()}}
    );
    // ElwMul
    ss_->elwmul_ =
      std::make_shared<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_mul
      );

    // ss_->shared_buffer_ = SharedBuffer::Client("CSM", session_id_, name());
    const int max_seq_len =
      maxSeqLength();  // max input seq len the LFM2 model supported
    initBufBos(max_seq_len);
  }
  ss_->instances__++;
}

AMDConvSplitMulKernel::~AMDConvSplitMulKernel() {
  ss_->instances__--;
  if (ss_->instances__ == 0) {
    // free some mem
    ss_->elwmul_.reset();
  }
  if (ss_->total_buf_ != nullptr) {
#ifdef _WIN32
    _aligned_free(ss_->total_buf_);
#else
    free(ss_->total_buf_);
#endif
  }
}

void AMDConvSplitMulKernel::ConvNpu(
  uint16_t* inp_bf16_ptr, uint16_t* wts_bf16_ptr, uint16_t* out_bf16_ptr,
  std::vector<int64_t> inp_dims, std::vector<int64_t> wts_dims,
  std::vector<int64_t> out_dims, bool init_weights
) {
  auto wts_tensors = std::vector<Tensor>{
    {wts_bf16_ptr,
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

  ss_->dd_conv_->set_tensor_shape(inp_tensors, out_tensors);
  if (init_weights) ss_->dd_conv_->initialize_const_params(wts_tensors, {});

  ss_->dd_conv_->execute(inp_tensors, out_tensors);
}

// input  : [N, W, C]
// weight : [M, C/G, kw]
// bias   : [M] (optional)
// output : [N, W_out, M]
void AMDConvSplitMulKernel::Conv1DSim(
  const uint16_t* input, const uint16_t* weight, const uint16_t* bias,
  uint16_t* output, int N, int W, int C, int M, int kw, int groups, int stride,
  int padding, int dilation
) {
  assert(C % groups == 0);
  assert(M % groups == 0);

  const int Cg = C / groups;
  const int Mg = M / groups;

  const int W_out = (W + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;

  auto in_idx = [&](int n, int w, int c) { return n * W * C + w * C + c; };

  auto wt_idx = [&](int m, int c, int k) {
    // weight layout: [M, Cg, kw]
    return m * Cg * kw + c * kw + k;
  };

  auto out_idx = [&](int n, int w, int m) {
    if (w > 0) {  // slice out first row.
      return n * (W_out - 1) * M + (w - 1) * M + m;
    } else {
      return n * (W_out - 1) * M + w * M + m;
    }
  };

  for (int n = 0; n < N; ++n) {
    for (int w_out = 0; w_out < W_out; ++w_out) {
      for (int g = 0; g < groups; ++g) {
        const int c_base = g * Cg;
        const int m_base = g * Mg;

        for (int m = 0; m < Mg; ++m) {
          const int m_idx = m_base + m;
          float sum = bias ? bias[m_idx] : 0.0f;

          for (int k = 0; k < kw; ++k) {
            const int w_in = w_out * stride + k * dilation - padding;
            if (w_in < 0 || w_in >= W) continue;

            for (int c = 0; c < Cg; ++c) {
              sum += bfloat16_to_float(input[in_idx(n, w_in, c_base + c)]) *
                     bfloat16_to_float(weight[wt_idx(m_idx, c, k)]);
            }
          }
          output[out_idx(n, w_out, m_idx)] = float_to_bfloat16(sum);
        }
      }
    }
  }
}

void AMDConvSplitMulKernel::ConvCpu(
  Ort::MemoryInfo const& cpu_mem_info, uint16_t* inp_bf16_ptr,
  uint16_t* wts_bf16_ptr, uint16_t* conv_out_ptr, std::vector<int64_t> inp_dims,
  std::vector<int64_t> wts_dims, std::vector<int64_t> out_dims,
  OrtKernelContext* context
) {
  PROFILING_START(token_share_buf_update)
  const int64_t conv_out_dims[3] = {
    out_dims[0], out_dims[2], (out_dims[1] + 1)
  };
  const int64_t conv_inp_dims[3] = {inp_dims[0], inp_dims[2], inp_dims[1]};
  const int64_t conv_wts_dims[3] = {wts_dims[0], wts_dims[1], wts_dims[2]};
  const auto conv_inp_sz =
    inp_dims[0] * inp_dims[1] * inp_dims[2] * sizeof(float);
  const auto conv_out_sz =
    conv_out_dims[0] * conv_out_dims[1] * conv_out_dims[2] * sizeof(float);
  const auto transpose_sz = (std::max)(conv_inp_sz, conv_out_sz);
  const auto wts_elems = wts_dims[0] * wts_dims[1] * wts_dims[2];
  const auto wts_sz = wts_elems * sizeof(float);

  PROFILING_START(cpu_trans_before)
  /*std::vector<float> tmp_float;
  int elem = inp_dims[0]*inp_dims[1]*inp_dims[2];
  tmp_float.resize(elem);
  ryzenai::bfloat16_buffer_to_float(
      (const uint16_t*)inp_bf16_ptr, elem,
      tmp_float.data()
    );
  ss_->ort_trans_.execute(
    (float*)(ss_->transpose_ptr_), tmp_float.data(),
    {inp_dims[0], inp_dims[1], inp_dims[2]}, context
  );*/
  transpose021WithCast<uint16_t, float>(
    inp_bf16_ptr, ss_->transpose_ptr_, inp_dims[0], inp_dims[1], inp_dims[2]
  );
  PROFILING_END(cpu_trans_before, false, name().c_str())

  PROFILING_START(cast_wts)
  // reuse partial split2_ptr buffer for wts.
  float* wts_fp_ptr =
    (float*)ss_->split2_ptr_ + alignTo4096(4 * out_dims[2] * sizeof(float));
  ryzenai::bfloat16_buffer_to_float(wts_bf16_ptr, wts_elems, wts_fp_ptr);
  PROFILING_END(cast_wts, false, name().c_str())

  std::vector<float> tmp_out_float;
  int elem = conv_out_dims[0] * conv_out_dims[1] * conv_out_dims[2];
  tmp_out_float.resize(elem);

  PROFILING_START(cpu_conv)
  ss_->ort_conv_.execute(
    Ort::Value::CreateTensor<float>(
      cpu_mem_info, ss_->transpose_ptr_, transpose_sz, conv_inp_dims, 3
    )
      .GetConst(),  // in
    Ort::Value::CreateTensor<float>(
      cpu_mem_info, wts_fp_ptr, wts_sz, conv_wts_dims, 3
    )
      .GetConst(),  // wts
    Ort::Value::CreateTensor<float>(
      cpu_mem_info, (float*)tmp_out_float.data(), conv_out_sz, conv_out_dims, 3
    )
      .GetUnowned(),  // out
    context
  );
  PROFILING_END(cpu_conv, false, name().c_str())

  PROFILING_START(cpu_trans_after)
  ss_->ort_trans_.execute(
    (float*)ss_->transpose_ptr_, tmp_out_float.data(),
    {conv_out_dims[0], conv_out_dims[1], conv_out_dims[2]}, context
  );
  ryzenai::float_buffer_to_bfloat16(
    (float*)ss_->transpose_ptr_ + out_dims[2],  // Slice out first row
    size_t(out_dims[1]) * size_t(out_dims[2]), (uint16_t*)conv_out_ptr
  );
  PROFILING_END(cpu_trans_after, false, name().c_str())
  return;
}

void AMDConvSplitMulKernel::initBufBos(size_t seq_len) {
  constexpr int conv_cache_seq_len = 3;
  const std::vector<size_t> split_sizes_val = {2048, 2048, 2048};
  const int concat_in0_elem = conv_cache_seq_len * split_sizes_val[0];
  // const auto split_axis_elemcnt = split_sizes_val[0] + split_sizes_val[1] +
  // split_sizes_val[2];
  ss_->split0_sz = seq_len * split_sizes_val[0] * sizeof(uint16_t);
  ss_->split1_sz = seq_len * split_sizes_val[1] * sizeof(uint16_t);
  ss_->split2_sz = seq_len * split_sizes_val[2] * sizeof(uint16_t);
  size_t concat_in0_sz = 3 * split_sizes_val[0] * sizeof(uint16_t);
  const auto conv_in0_len = ss_->split0_sz + concat_in0_sz;

#if 0
  // Notice: 3k crash with the shared buffer approach, using aligned alloc
  // 1. make sure "split0" and "split2" buffer back to back,
  // and reused for ConvCpu float output.
  // 2. "transpose" buffer used by float transpose output and bf16 Conv output.
  ss_->shared_buffer_.Update({
    {"split1", alignTo4096(ss_->split1_sz)},
    {"conv_in", alignTo4096(conv_in0_len)},
    {"split0", alignTo4096(ss_->split0_sz)},
    {"split2", alignTo4096(ss_->split2_sz)},
    {"transpose", alignTo4096(conv_in0_len * 2)}, // float
  });

  ss_->split0_ptr_ =
      (uint16_t*)ss_->shared_buffer_.Get("split0").ptr;
  ss_->split1_ptr_ =
      (uint16_t*)ss_->shared_buffer_.Get("split1").ptr;
  ss_->split2_ptr_ =
      (uint16_t*)ss_->shared_buffer_.Get("split2").ptr;
  ss_->conv_in_ptr_ =
      (uint16_t*)ss_->shared_buffer_.Get("conv_in").ptr;
  ss_->transpose_ptr_ =
      (float*)ss_->shared_buffer_.Get("transpose").ptr;
#else
  auto total_mem_size =
    alignTo4096(ss_->split0_sz) + alignTo4096(ss_->split1_sz) +
    alignTo4096(ss_->split2_sz) + alignTo4096(conv_in0_len) +
    alignTo4096(conv_in0_len * 2);
#ifdef _WIN32
  ss_->total_buf_ = (uint8_t*)_aligned_malloc(total_mem_size, 4096);
#else
  ss_->total_buf_ = (uint8_t*)aligned_alloc(4096, total_mem_size);
#endif
  int offset = 0;

  ss_->split1_ptr_ = (uint16_t*)(ss_->total_buf_ + offset);
  offset += alignTo4096(ss_->split1_sz);

  ss_->conv_in_ptr_ = (uint16_t*)(ss_->total_buf_ + offset);
  offset += alignTo4096(conv_in0_len);

  ss_->split0_ptr_ = (uint16_t*)ss_->total_buf_ + offset;
  offset += alignTo4096(ss_->split0_sz);

  ss_->split2_ptr_ = (uint16_t*)(ss_->total_buf_ + offset);
  offset += alignTo4096(ss_->split2_sz);

  ss_->transpose_ptr_ = (float*)(ss_->total_buf_ + offset);
  offset += alignTo4096(conv_in0_len * 2);
#endif
  ss_->mul0_bo0_ = ss_->elwmul_->bind_bo(ss_->split0_ptr_, ss_->split0_sz);
  ss_->mul0_bo1_ = ss_->elwmul_->bind_bo(ss_->split2_ptr_, ss_->split1_sz);
  ss_->mul0_bo0_.sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split0_sz, 0);
  ss_->mul0_bo1_.sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split1_sz, 0);
  if (ss_->inline_concat_en_) {
    ss_->mul0_bo2_ = ss_->elwmul_->bind_bo(
      (void*)(ss_->conv_in_ptr_ + concat_in0_elem), ss_->split1_sz
    );
    ss_->mul0_bo2_.sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split1_sz, 0);
  }

  ss_->mul1_bo0_ = ss_->elwmul_->bind_bo(ss_->split1_ptr_, ss_->split1_sz);
  ss_->mul1_bo1_ = ss_->elwmul_->bind_bo(ss_->transpose_ptr_, ss_->split1_sz);
  ss_->mul1_bo0_.sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split1_sz, 0);
  ss_->mul1_bo1_.sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split1_sz, 0);
}

void split_cpu(
  uint16_t* inp, uint16_t* out0, uint16_t* out1, uint16_t* out2, size_t B,
  size_t M, size_t K, size_t k_step
) {
  constexpr int out_num = 3;
  auto step = out_num * k_step;
  auto size = k_step * sizeof(uint16_t);
  // for (auto i = 0; i < B; i++) {
  for (auto j = 0; j < M; j++) {
    memcpy(out0 + j * k_step, inp + j * K, size);
    memcpy(out1 + j * k_step, inp + j * K + k_step, size);
    memcpy(out2 + j * k_step, inp + j * K + 2 * k_step, size);
  }
  //}
}

void concat_cpu(
  uint16_t* in1, uint16_t* in2, uint16_t* out, size_t B, size_t M1, size_t M2,
  size_t K
) {
  auto elem1 = K * M1;
  auto elem2 = K * M2;
  auto size1 = elem1 * sizeof(uint16_t);
  auto size2 = elem2 * sizeof(uint16_t);
  auto out_elem = elem1 + elem2;
  for (auto i = 0; i < B; i++) {
    memcpy(out + i * out_elem, in1, size1);
    memcpy(out + i * out_elem + elem1, in2, size2);
  }
}

static void mul_cpu(
  uint16_t* a, uint16_t* b, uint16_t* out, size_t M, size_t K
) {
  std::uint32_t val_int;
  float val_float;
  float total_sz = M * K;
  for (int i = 0; i < total_sz; i++) {
    // std::cout << "M:" << M << " K:" << K << " i:" << i << std::endl;
    val_float = bfloat_to_float_2(a[i]) * bfloat_to_float_2(b[i]);
    val_int = *reinterpret_cast<std::uint32_t*>(&val_float);
    out[i] = (val_int >> 16) & 0xFFFF;
  }
}

void AMDConvSplitMulKernel::Compute(OrtKernelContext* context) {
  static const auto cpu_mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::KernelContext ctx(context);

  const auto concat_in0 = ctx.GetInput(0);
  const auto concat_in0_shape =
    concat_in0.GetTensorTypeAndShapeInfo().GetShape();
  const auto concat_in0_elem =
    concat_in0.GetTensorTypeAndShapeInfo().GetElementCount();
  const auto concat_in0_sz = concat_in0_elem * sizeof(uint16_t);
  const auto split_in0 = ctx.GetInput(1);
  const auto split_in0_shape = split_in0.GetTensorTypeAndShapeInfo().GetShape();
  const int split_sz =
    split_in0.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(uint16_t);
  const int64_t B = split_in0_shape[0];
  const int64_t seq_len = split_in0_shape[1];

  const auto split_sizes = ctx.GetInput(2);
  auto split_sizes_val = split_sizes.GetTensorData<int64_t>();

  const auto out0_dims = std::vector<int64_t>{B, seq_len, split_sizes_val[1]};
  auto output_tensor0 = ctx.GetOutput(0, out0_dims);
  uint16_t* out_ptr0 =
    (uint16_t*)output_tensor0.GetTensorMutableData<Ort::BFloat16_t>();

  const auto split_axis_elemcnt =
    split_sizes_val[0] + split_sizes_val[1] + split_sizes_val[2];
  ss_->split0_sz = split_sz * split_sizes_val[0] / split_axis_elemcnt;
  ss_->split1_sz = split_sz * split_sizes_val[1] / split_axis_elemcnt;
  ss_->split2_sz = split_sz * split_sizes_val[2] / split_axis_elemcnt;
  const auto conv_in0_len = ss_->split0_sz + concat_in0_sz;

  // split
  const int64_t split_in_dim[3] = {B, seq_len, split_in0_shape[2]};
  const int64_t split0_dim[3] = {B, seq_len, split_sizes_val[0]};
  const int64_t split1_dim[3] = {B, seq_len, split_sizes_val[1]};
  const int64_t split2_dim[3] = {B, seq_len, split_sizes_val[2]};

  bool split_cpu_sim_en = true;
  Ort::Value ort_split0 = Ort::Value::CreateTensor<uint16_t>(  //crash?
            cpu_mem_info, ss_->split0_ptr_, ss_->split0_sz, split0_dim, 3
          );

  auto* split_in_ptr = (uint16_t*)(split_in0.GetTensorData<Ort::BFloat16_t>());
  if (split_cpu_sim_en) {
    split_cpu(
      split_in_ptr, ss_->split0_ptr_, ss_->split1_ptr_, ss_->split2_ptr_, B,
      seq_len, split_in0_shape[2], split_sizes_val[0]
    );
  } else {  // TODO: ort crash to be debug.
    Ort::Value ort_split1 = Ort::Value::CreateTensor<uint16_t>(
      cpu_mem_info, ss_->split1_ptr_, ss_->split1_sz, split1_dim, 3
    );
    Ort::Value ort_split2 = Ort::Value::CreateTensor<uint16_t>(
      cpu_mem_info, ss_->split2_ptr_, ss_->split2_sz, split2_dim, 3
    );
    auto ort_split_in0 = Ort::Value::CreateTensor<uint16_t>(
                           cpu_mem_info, split_in_ptr, split_sz, split_in_dim, 3
    )
                           .GetConst();
    auto out0 = ort_split0.GetUnowned();
    auto out1 = ort_split1.GetUnowned();
    auto out2 = ort_split2.GetUnowned();
    ss_->ort_split_.execute(
      ort_split_in0, split_sizes, out0, out1, out2, context
    );
  }

  // mul 0
  bool mul_cpu_en = false;
  if (mul_cpu_en) {
    mul_cpu(
      ss_->split0_ptr_, ss_->split2_ptr_,
      ss_->inline_concat_en_ ? (ss_->conv_in_ptr_ + concat_in0_elem)
                             : ss_->split0_ptr_,
      (size_t)split0_dim[1], (size_t)split0_dim[2]
    );
  } else {
    ss_->elwmul_->set_kernel_shape(
      {(size_t)split0_dim[1], (size_t)split0_dim[2]}
    );
    std::vector<xrt::bo> mul_in_bos = {ss_->mul0_bo0_, ss_->mul0_bo1_};
    std::vector<xrt::bo> mul_out_bos = {
      ss_->inline_concat_en_ ? ss_->mul0_bo2_ : ss_->mul0_bo0_
    };
    mul_in_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split0_sz, 0);
    mul_in_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split1_sz, 0);
    ss_->elwmul_->execute(mul_in_bos, mul_out_bos, true);
    mul_out_bos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE, ss_->split0_sz, 0);
  }

  // concat
  bool concat_cpu_sim_en = true;
  const int64_t concat_out_dim[3] = {
    concat_in0_shape[0], concat_in0_shape[1] + seq_len, concat_in0_shape[2]
  };
  Ort::Value ort_concat_out = Ort::Value::CreateTensor<uint16_t>(
    cpu_mem_info, ss_->conv_in_ptr_, (ss_->split0_sz + concat_in0_sz),
    concat_out_dim, 3
  );
  auto* concat_in0_ptr =
    (uint16_t*)(concat_in0.GetTensorData<Ort::BFloat16_t>());
  if (ss_->inline_concat_en_) {
    memcpy(ss_->conv_in_ptr_, concat_in0_ptr, concat_in0_sz);
  } else if (concat_cpu_sim_en) {
    concat_cpu(
      concat_in0_ptr, ss_->split0_ptr_, ss_->conv_in_ptr_, B,
      concat_in0_shape[1], seq_len, concat_in0_shape[2]
    );
  } else {  // TODO: ort crash to be debug.
    const int64_t concat_in0_dim[3] = {
      concat_in0_shape[0], concat_in0_shape[1], concat_in0_shape[2]
    };
    auto ort_concat_in0 =
      Ort::Value::CreateTensor<uint16_t>(
        cpu_mem_info, concat_in0_ptr, concat_in0_sz, concat_in0_dim, 3
      )
        .GetConst();
    auto out = ort_concat_out.GetUnowned();
    ss_->ort_concat_.execute(
      ort_concat_in0, ort_split0.GetConst(), out, context
    );
  }

  // conv
  bool conv_cpu_en = false;
  const auto conv_wts = ctx.GetInput(3);
  const auto conv_wts_shape = conv_wts.GetTensorTypeAndShapeInfo().GetShape();
  uint16_t* wts_ptr = (uint16_t*)conv_wts.GetTensorData<Ort::BFloat16_t>();
  std::vector<int64_t> conv_inp_dims = {
    concat_out_dim[0], concat_out_dim[1], concat_out_dim[2]
  };
  std::vector<int64_t> conv_wts_dims = {
    conv_wts_shape[0], conv_wts_shape[1], conv_wts_shape[2]
  };
  std::vector<int64_t> conv_out_dims = {
    concat_out_dim[0], seq_len, concat_out_dim[2]
  };

  if (conv_cpu_en || seq_len <= 256) {
    bool cpu_sim_en = false;
    if (cpu_sim_en) {
      Conv1DSim(
        ss_->conv_in_ptr_, wts_ptr, nullptr, (uint16_t*)ss_->transpose_ptr_,
        conv_inp_dims[0], conv_inp_dims[1], conv_inp_dims[2], conv_wts_dims[0],
        conv_wts_dims[2], 2048
      );
    } else {
      ConvCpu(
        cpu_mem_info, ss_->conv_in_ptr_, wts_ptr,
        (uint16_t*)ss_->transpose_ptr_, conv_inp_dims, conv_wts_dims,
        conv_out_dims, context
      );
    }
  } else {
    ConvNpu(
      ss_->conv_in_ptr_, wts_ptr, (uint16_t*)ss_->transpose_ptr_, conv_inp_dims,
      conv_wts_dims, conv_out_dims, true
    );
  }

  // ort slice & cpu pad
  auto out_dims = concat_in0_shape;
  out_dims[1] = 3 + 1;
  auto output_tensor = ctx.GetOutput(1, out_dims);
  auto out = output_tensor.GetTensorMutableData<Ort::BFloat16_t>();
  auto output_sz = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount() *
                   sizeof(uint16_t);
  auto out_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  const int64_t out_dim[3] = {out_shape[0], out_shape[1] - 1, out_shape[2]};

  memset(
    out + 3 * concat_in0_shape[concat_in0_shape.size() - 1], 0,
    concat_in0_shape[concat_in0_shape.size() - 1] * sizeof(uint16_t)
  );
  auto ort_out = Ort::Value::CreateTensor<uint16_t>(
    cpu_mem_info, (uint16_t*)out, output_sz, out_dim, 3
  );
  ss_->ort_slice_.execute(
    ort_concat_out.GetConst(), ctx.GetInput(4), ctx.GetInput(5),
    ctx.GetInput(6), ort_out.GetUnowned(), context
  );

  // mul1
  auto* mul1_in0_ptr = ss_->mul1_bo0_.map<uint16_t*>();
  auto* mul1_out_ptr = mul1_in0_ptr;
  if (mul_cpu_en) {
    mul_cpu(
      mul1_in0_ptr, ss_->mul1_bo1_.map<uint16_t*>(), mul1_out_ptr,
      (size_t)split2_dim[1], (size_t)split2_dim[2]
    );
  } else {
    ss_->elwmul_->set_kernel_shape(
      {(size_t)split2_dim[1], (size_t)split2_dim[2]}
    );
    std::vector<xrt::bo> mul1_in_bos = {ss_->mul1_bo0_, ss_->mul1_bo1_};
    std::vector<xrt::bo> mul1_out_bos = {ss_->mul1_bo0_};
    mul1_in_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split0_sz, 0);
    mul1_in_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE, ss_->split2_sz, 0);
    ss_->elwmul_->execute(mul1_in_bos, mul1_out_bos, true);
    mul1_out_bos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE, ss_->split1_sz, 0);
  }
  std::memcpy((void*)out_ptr0, mul1_out_ptr, ss_->split1_sz);

  ss_->run_instances__++;
}

}  // namespace ryzenai::onnx_utils
