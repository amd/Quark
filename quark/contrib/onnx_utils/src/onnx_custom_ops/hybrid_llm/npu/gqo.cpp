// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "gqo.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

#include "common.hpp"
#include "external_data.hpp"
#include "jit_node_impl.hpp"
#include "lora.hpp"
#include "ops/ops_common/matmul_matrix_public.hpp"
#include "ort.hpp"
#include "profiling/profiling.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#include "ryzenai/onnx_utils/string.hpp"

namespace fs = std::filesystem;

namespace {

// uncomment to save the KV cache to files
// #define DUMP_KV_CACHE
// update the name to save the KV cache from a particular GQO node
constexpr auto kDumpGqoName = "gqo_6_0_0";

template <typename T>
void saveKvCache(
  const std::string& label, const std::string& dtype, const T* data, size_t size
) {
  std::cout << label << " (" << dtype << "): ";
  for (int i = 0; i < 10; ++i) {
    std::cout << data[i].ToFloat() << " ";
  }
  std::cout << std::endl;
  ryzenai::onnx_utils::writeToFile<T, float>(
    "dump_prefill_" + label + "_" + dtype + ".txt", data, size, false
  );
}

std::string shape2str(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

constexpr auto kPastKeyIdx = 3;
constexpr auto kCosIdx = 7;
constexpr auto kSinIdx = 8;
constexpr auto kMmWtsIdx = 12;
constexpr auto kMmScaleIdx = 13;
constexpr auto kMmZpIdx = 14;

}  // namespace

namespace ryzenai::onnx_utils {

static void convert_buffer_bfloat16_to_float16(
  std::uint16_t* dest_ptr, const std::uint16_t* src_ptr, const int num_elems
) {
  // need to convert from bfloat16 to float16
  // use intrinsic to convert bfloat16 to float32 then back down to float16

  constexpr int kBlockSize = 16;
  const auto num_blocks = num_elems / kBlockSize;
  constexpr int kRoundingMode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    const std::uint16_t* curr_src_ptr = &src_ptr[kBlockSize * block_idx];

    std::uint16_t* curr_dest_ptr = &dest_ptr[kBlockSize * block_idx];

    // read 16 bfloat16 - 8 in each 128-bit
    // [src_0, src_1 | src_2, src_3]
    __m256i src_packed16_i =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr));

    // cast from bloat16 to float32
    __m512 src = _mm512_cvtpbh_ps((__m256bh)src_packed16_i);

    // cast from float32 to float16
    __m256i dst_i = _mm512_cvtps_ph(src, kRoundingMode);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(curr_dest_ptr), dst_i);
  }

  const auto offset = kBlockSize * num_blocks;

  if (offset != num_elems) {
    throw std::runtime_error("implement bfloat to float16 leftover conversion");
  }
}

static void convert_buffer_float16_to_bfloat16(
  std::uint16_t* dest_ptr, const std::uint16_t* src_ptr, const int num_elems
) {
  // need to convert from float16 to bfloat16
  // use intrinsic to convert float16 to float32 then back down to bfloat16

  constexpr int kBlockSize = 16;
  const auto num_blocks = num_elems / kBlockSize;

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    const std::uint16_t* curr_src_ptr = &src_ptr[kBlockSize * block_idx];

    std::uint16_t* curr_dest_ptr = &dest_ptr[kBlockSize * block_idx];

    // read 16 float16
    __m256h src_packed16_i = _mm256_loadu_ph(curr_src_ptr);

    // cast from float16 to float32
    __m512 src = _mm512_cvtph_ps((__m256i)src_packed16_i);

    // cast from float32 to bfloat16
    __m256bh dst_bf16 = _mm512_cvtneps_pbh(src);

    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(curr_dest_ptr), (__m256i)dst_bf16
    );
  }

  const auto offset = kBlockSize * num_blocks;

  if (offset != num_elems) {
    throw std::runtime_error("implement bfloat to float16 leftover conversion");
  }
}

template class JitNode<AMDGQOKernel>;
using NPUTensor = ::Tensor;

int64_t GqaKernelInfo::tryPadSeq(int64_t S, size_t max_seq_len) const {
  if (S < 0 || S > max_seq_len) {
    std::string msg = "S = " + std::to_string(S) +
                      ", max_seq_len = " + std::to_string(max_seq_len) +
                      " is unsupported";
    throw std::runtime_error(msg);
  }

  auto it = supported_seqlen_.lower_bound(S);
  return (it != supported_seqlen_.end()) ? *it : max_seq_len;
}

// check if the seq_len is supported by aie
bool GqaKernelInfo::isSeqSupported(int seq_len) const {
  return (supported_seqlen_.find(seq_len) != supported_seqlen_.end());
}

// check if the num_head is supported by aie
bool GqaKernelInfo::isNumHeadsSupported(int num_head) const {
  return (kSupportedNumHeads.find(num_head) != kSupportedNumHeads.end());
}

void GqaKernelInfo::setVersion(const MladfVersion& version) {
  switch (version) {
    case MladfVersion::v1:
      supported_seqlen_ = kSupportedSeqLenV1;
      break;
    case MladfVersion::v2:
      supported_seqlen_ = kSupportedSeqLenV2;
      break;
    default:
      throw std::invalid_argument(
        "Unsupported MLADF version: " + version.str()
      );
  }
}

AttnMaskLut::AttnMaskLut() {
  auto seqs = {128,  256,  512,  640,  1024, 1152, 1280, 1536, 1664,
               1792, 2048, 2176, 2304, 2560, 2816, 3072, 4096};
  int32_t idx = 0;
  for (const auto& s : seqs) {
    auto lut = ryzenai::RyzenMM::NPUAllocator<'ATM1'>().AllocateBuffer(
      sizeof(uint16_t) * s * s
    );
    fill(lut.Data<uint16_t>(), s, -1);
    attn_masks_lut_.try_emplace(s, std::move(lut));
  }
}

bool AttnMaskLut::has(int32_t seq_len) const {
  return attn_masks_lut_.count(seq_len) != 0;
}

auto AttnMaskLut::get(int32_t seq_len) const {
  if (!has(seq_len)) {
    throw std::invalid_argument(
      "There is not LUT for " + std::to_string(seq_len)
    );
  }
  return attn_masks_lut_.at(seq_len);
}

void AttnMaskLut::fill(
  uint16_t* attn_mask, int seq_len, int64_t local_window_size
) {
  std::memset(attn_mask, 0, seq_len * seq_len * sizeof(uint16_t));
  const uint16_t neg_inf_ui16 =
    float_to_bfloat16(-std::numeric_limits<float>::infinity());

  // by default generate causal mask where only lower triangle
  // in matrix (including diagonal) is zero
  for (int i = 0; i < seq_len; i++) {
    uint16_t* start_ptr = attn_mask + i * seq_len + (i + 1);
    size_t count = seq_len - (i + 1);
#if defined(_MSC_VER)
    __stosw(start_ptr, neg_inf_ui16, count);
#else
    std::fill_n(start_ptr, count, neg_inf_ui16);
#endif
  }

  // for banded/sliding window attention
  // local window size includes current token, so it will process
  // local_window_size - 1  tokens to the left/before it
  const bool use_local_window =
    (local_window_size != -1) && (local_window_size < seq_len);

  if (use_local_window) {
    for (int i = local_window_size; i < seq_len; i++) {
      uint16_t* start_ptr = attn_mask + i * seq_len;
      size_t count = i - local_window_size + 1;
#if defined(_MSC_VER)
      __stosw(start_ptr, neg_inf_ui16, count);
#else
      std::fill_n(start_ptr, count, neg_inf_ui16);
#endif
    }
  }
}

// kernel expands cos as cos,cos,cos_diff,cos_diff, so K dim should be
// 4*shape_cs_1, sin is the same as cos.
static const int ROPE_WTS_EXPAND_FACTOR = 4;

RyzenMM::BufferRef AttnMaskProvider::get(
  int32_t S, int32_t T, int32_t T_pad, int64_t local_window_size
) {
  //      T-S
  //    |-----+----S-----|---------|
  //    |      \      *  |         |
  //    S   0    `\      | S       |
  //    |\          `\   |         |
  //    |*`\           `\|         |
  // T  |---+-local_win--+         | T_pad
  //    |                |         |
  //    |       *        | T-S     |
  //    |------- T-------|         |
  //    |                     *    |
  //    |                          |
  //    |                          |
  //    |--------------------------|
  //              T_pad

  // S: current seq_len
  // T: total seq_len including current prompt
  // T_pad: pad T to some kernel size
  // local_window_size: -1 if global attention (default), otherwise banded
  //                     attention

  auto size = T_pad * T_pad;
  auto bf16_attention_mask =
    ryzenai::RyzenMM::NPUAllocator<'ATM2'>().AllocateBuffer(
      size * sizeof(uint16_t)
    );
  uint16_t* uptr = bf16_attention_mask.Data<uint16_t>();
  const uint16_t neg_inf_ui16 = float_to_bfloat16(-3.389e38f);
#if defined(_MSC_VER)
  __stosw(uptr, neg_inf_ui16, size);
#else
  std::fill_n(uptr, size, neg_inf_ui16);
#endif

  bool use_local_window = (local_window_size != -1) && (local_window_size < T);
  int left_window_size = use_local_window ? (local_window_size - 1) : T;

  for (int i = 0; i < S; i++) {
    uint16_t* row_ptr = uptr + i * T_pad;
    int pos_id = T - S + i;
    int start_pos_id = std::max(0, pos_id - left_window_size);
    uint16_t* start_ptr = row_ptr + start_pos_id;
    size_t count = pos_id - start_pos_id + 1;
    memset(start_ptr, 0, count * sizeof(uint16_t));
  }

  return bf16_attention_mask;
}

RyzenMM::BufferRef AttnMaskProvider::get(int32_t S, int64_t local_window_size) {
  // if S is in aie support list, try to get LUT first.
  auto is_supported = use_lut(S);
  bool use_window_size = local_window_size != -1 && (local_window_size < S);

  if (is_supported && !use_window_size) {
    return attn_mask_lut_->get(S);
  } else {
    //  otherwise construct the LUT on the fly.
    //  Todo(ltp): consider case when b != 1;

    auto size = 1 * 1 * S * S;  // B * 1 * S * S
    auto bf16_attention_mask =
      ryzenai::RyzenMM::NPUAllocator<'ATM2'>().AllocateBuffer(
        size * sizeof(uint16_t)
      );

    AttnMaskLut::fill(
      bf16_attention_mask.Data<uint16_t>(), S, local_window_size
    );
    return bf16_attention_mask;
  }
}

bool AttnMaskProvider::use_lut(int32_t seq_len) const {
  return aie_kernel_info_->isSeqSupported(seq_len) &&
         attn_mask_lut_->has(seq_len);
}

void getHeadSink(
  std::vector<float>& head_sink, Ort::ConstValue& head_sink_tensor
) {
  const auto dtype =
    head_sink_tensor.GetTensorTypeAndShapeInfo().GetElementType();
  const auto shape = head_sink_tensor.GetTensorTypeAndShapeInfo().GetShape();

  const auto num_heads = shape.at(0);

  const Ort::Float16_t* head_sink_fp16 =
    head_sink_tensor.GetTensorData<Ort::Float16_t>();

  const Ort::BFloat16_t* head_sink_bf16 =
    head_sink_tensor.GetTensorData<Ort::BFloat16_t>();

  const float* head_sink_fp32 = head_sink_tensor.GetTensorData<float>();

  head_sink.resize(num_heads);

  for (auto head_idx = 0; head_idx < num_heads; head_idx++) {
    float val = 0.0f;

    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == dtype) {
      val = head_sink_fp32[head_idx];
    } else if (ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 == dtype) {
      val = head_sink_bf16[head_idx].ToFloat();
    } else {
      val = head_sink_fp16[head_idx].ToFloat();
    }

    head_sink.at(head_idx) = val;
  }
}

std::pair<RyzenMM::BufferRef, std::vector<int64_t>> getRopeCache(
  RyzenMM::Allocator* allocator, Ort::ConstValue& cos_tensor,
  Ort::ConstValue& sin_tensor, bool rotary_interleaved,
  const MladfVersion& mladf_version
) {
  auto cos_shape = cos_tensor.GetTensorTypeAndShapeInfo().GetShape();
  const auto cache_dtype =
    cos_tensor.GetTensorTypeAndShapeInfo().GetElementType();
  int shape_cs_0 = cos_shape[0];
  int shape_cs_1 = cos_shape[1];

  const Ort::Float16_t* cos_data_fp16 =
    cos_tensor.GetTensorData<Ort::Float16_t>();
  const Ort::Float16_t* sin_data_fp16 =
    sin_tensor.GetTensorData<Ort::Float16_t>();

  const Ort::BFloat16_t* cos_data_bf16 =
    cos_tensor.GetTensorData<Ort::BFloat16_t>();
  const Ort::BFloat16_t* sin_data_bf16 =
    sin_tensor.GetTensorData<Ort::BFloat16_t>();

  const float* cos_data_fp32 = cos_tensor.GetTensorData<float>();
  const float* sin_data_fp32 = sin_tensor.GetTensorData<float>();

  auto convert_cos_sin_data = [&](int i, int j) {
    float cos_val = 0.0f;
    float sin_val = 0.0f;

    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == cache_dtype) {
      cos_val = cos_data_fp32[i * shape_cs_1 + j];
      sin_val = sin_data_fp32[i * shape_cs_1 + j];
    } else if (ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 == cache_dtype) {
      cos_val = cos_data_bf16[i * shape_cs_1 + j].ToFloat();
      sin_val = sin_data_bf16[i * shape_cs_1 + j].ToFloat();
    } else {
      cos_val = cos_data_fp16[i * shape_cs_1 + j].ToFloat();
      sin_val = sin_data_fp16[i * shape_cs_1 + j].ToFloat();
    }

    return std::make_pair(cos_val, sin_val);
  };

  auto max_seq_length = shape_cs_0;
  // cs_1 = shape_cs_1;
  // kernel expands cos as cos,cos,cos_diff,cos_diff, so K dim should be
  // 4*shape_cs_1, sin is the same as cos.

  auto cos_sin_cache = allocator->AllocateBuffer(
    2 * max_seq_length * (shape_cs_1 * ROPE_WTS_EXPAND_FACTOR) *
    sizeof(uint16_t)
  );
  auto cos_element_num = max_seq_length * shape_cs_1;

  auto cos_cache_dest = reinterpret_cast<uint16_t*>(cos_sin_cache.Data());
  auto sin_cache_dest = reinterpret_cast<uint16_t*>(cos_sin_cache.Data()) +
                        ROPE_WTS_EXPAND_FACTOR * cos_element_num;

  int M_stride = shape_cs_1 * ROPE_WTS_EXPAND_FACTOR;
  int diff_offset = 2 * shape_cs_1;
  // duplicate second dimension for cos and sin cache
  for (int i = 0; i < shape_cs_0; ++i) {
    auto offset = i * M_stride;
    for (int j = 0; j < shape_cs_1; ++j) {
      const auto [cos_val, sin_val] = convert_cos_sin_data(i, j);

      std::uint16_t cos_val_bfloat16 = float_to_bfloat16_2(cos_val);
      std::uint16_t sin_val_bfloat16 = float_to_bfloat16_2(sin_val);
      float trig_cos = bfloat16_to_float_single(cos_val_bfloat16);
      float trig_sin = bfloat16_to_float_single(sin_val_bfloat16);
      float loss_cos = cos_val - trig_cos;
      float loss_sin = sin_val - trig_sin;
      std::uint16_t cos_val_bfloat16_low = float_to_bfloat16_2(loss_cos);
      std::uint16_t sin_val_bfloat16_low = float_to_bfloat16_2(loss_sin);
      // if 0 keep as is, otherwise flip sign bit
      std::uint16_t neg_sin_val_bfloat16 =
        (sin_val_bfloat16 == 0) ? (0) : (sin_val_bfloat16 ^ (1U << 15));
      std::uint16_t neg_sin_val_bfloat16_low =
        (sin_val_bfloat16 == 0) ? (0) : (sin_val_bfloat16_low ^ (1U << 15));

      // Duplicate the values for second half
      if (mladf_version == MladfVersion::aie4_v1) {
        cos_cache_dest[offset + (2 * j)] =
          uint16_t(*(uint32_t*)&cos_val) & 0xFFFF;
        cos_cache_dest[offset + (2 * j + 1)] =
          uint16_t(((*(uint32_t*)&cos_val) & 0xFFFF0000) >> 16);
        cos_cache_dest[offset + diff_offset + (2 * j)] =
          cos_cache_dest[offset + (2 * j)];
        cos_cache_dest[offset + diff_offset + (2 * j + 1)] =
          cos_cache_dest[offset + (2 * j + 1)];

        sin_cache_dest[offset + (2 * j)] =
          uint16_t(*(uint32_t*)&sin_val) & 0xFFFF;
        sin_cache_dest[offset + (2 * j + 1)] =
          uint16_t(((*(uint32_t*)&sin_val) & 0xFFFF0000) >> 16);
        sin_cache_dest[offset + diff_offset + (2 * j)] =
          sin_cache_dest[offset + (2 * j)];
        sin_cache_dest[offset + diff_offset + (2 * j + 1)] =
          sin_cache_dest[offset + (2 * j + 1)];
      } else {
        if (!rotary_interleaved) {
          cos_cache_dest[offset + j] = cos_val_bfloat16;
          cos_cache_dest[offset + shape_cs_1 + j] = cos_val_bfloat16;
          cos_cache_dest[offset + diff_offset + j] = cos_val_bfloat16_low;
          cos_cache_dest[offset + diff_offset + shape_cs_1 + j] =
            cos_val_bfloat16_low;

          sin_cache_dest[offset + j] = sin_val_bfloat16;
          sin_cache_dest[offset + shape_cs_1 + j] = sin_val_bfloat16;
          sin_cache_dest[offset + diff_offset + j] = sin_val_bfloat16_low;
          sin_cache_dest[offset + diff_offset + shape_cs_1 + j] =
            sin_val_bfloat16_low;
        } else {
          cos_cache_dest[offset + (2 * j)] = cos_val_bfloat16;
          cos_cache_dest[offset + (2 * j + 1)] = cos_val_bfloat16;
          cos_cache_dest[offset + diff_offset + (2 * j)] = cos_val_bfloat16_low;
          cos_cache_dest[offset + diff_offset + (2 * j + 1)] =
            cos_val_bfloat16_low;

          sin_cache_dest[offset + (2 * j)] = neg_sin_val_bfloat16;
          sin_cache_dest[offset + (2 * j + 1)] = sin_val_bfloat16;
          sin_cache_dest[offset + diff_offset + (2 * j)] =
            neg_sin_val_bfloat16_low;
          sin_cache_dest[offset + diff_offset + (2 * j + 1)] =
            sin_val_bfloat16_low;
        }
      }
    }
  }

  return {cos_sin_cache, cos_shape};
}

// pad qkv to qkv_padded with 0
void pad_qkv(
  const ryzenai::ExecutionProviderExtensions& epx, const uint16_t* t,
  uint16_t* t_padded, int64_t seqlen, int64_t seqlen_padded, int64_t num_heads,
  int64_t head_size
) {
  std::memset(
    t_padded, 0, num_heads * seqlen_padded * head_size * sizeof(uint16_t)
  );
  for (int64_t n = 0; n < num_heads; n++) {
    epx.MemCpy(
      t_padded + n * seqlen_padded * head_size, t + n * seqlen * head_size,
      seqlen * head_size * sizeof(uint16_t)
    );
  }
}

/* pad q even in aie_rope mode.
   Because there is the case, that q is padded to 128,
   but k is concatted with content in kvcache then padded to bigger such as 1024
   . So the padded result of q in aie_rope can't be used directly in bmm1, it
   must be repadded as 1024.
 */
void pad_q(
  const ryzenai::ExecutionProviderExtensions& epx, const uint16_t* src,
  uint16_t* dst, int64_t seqlen, int64_t S_pad1, int64_t S_pad2,
  int64_t num_heads, int64_t head_size
) {
  std::memset(dst, 0, num_heads * S_pad2 * head_size * sizeof(uint16_t));
  for (int64_t n = 0; n < num_heads; n++) {
    epx.MemCpy(
      dst + n * S_pad2 * head_size, src + n * S_pad1 * head_size,
      seqlen * head_size * sizeof(uint16_t)
    );
  }
}

/// @brief  pad k/v from [B, N_kv, S, H] to [B, N_q, S, H]
/// @param dst padded k/v
/// @param src k/v to pad
void pad_group_kv(
  ryzenai::ExecutionProviderExtensions& epx, uint16_t* dst, uint16_t* src,
  int q_num_head, int kv_num_head, int seq_len, int head_size
) {
  int copy_size = seq_len * head_size;
  int num_group = q_num_head / kv_num_head;
  for (int n = 0; n < kv_num_head; n++) {
    for (int g = 0; g < num_group; g++) {
      epx.MemCpy(
        dst + (n * num_group + g) * copy_size, src + n * copy_size,
        copy_size * sizeof(uint16_t)
      );
    }
  }
}

void pad_group_kv_BSNH(
  ryzenai::ExecutionProviderExtensions& epx, uint16_t* dst, uint16_t* src,
  int q_num_head, int kv_num_head, int seq_len, int head_size
) {
  int copy_size = head_size;
  int num_group = q_num_head / kv_num_head;
  for (int s = 0; s < seq_len; s++) {
    for (int n = 0; n < kv_num_head; n++) {
      for (int g = 0; g < num_group; g++) {
        epx.MemCpy(
          dst + s * q_num_head * head_size + (n * num_group + g) * copy_size,
          src + s * kv_num_head * head_size + n * copy_size,
          copy_size * sizeof(uint16_t)
        );
      }
    }
  }
}
void try_pad_qkv(
  ryzenai::ExecutionProviderExtensions& epx, uint16_t* dst, uint16_t* src,
  int64_t N, int64_t S, int64_t S_pad, int64_t H
) {
  if (S != S_pad) {
    std::memset(dst, 0, N * S_pad * H * sizeof(uint16_t));
    for (int64_t n = 0; n < N; n++) {
      epx.MemCpy(
        dst + n * S_pad * H, src + n * S * H, S * H * sizeof(uint16_t)
      );
    }
  } else {
    epx.MemCpy(dst, src, N * S * H * sizeof(uint16_t));
  }
}
void AMDGQOKernel::set_params_bmm(size_t index) {
  std::map<std::string, std::any> attr = {
    {"skip_create_input_a", 1},
    {"skip_create_input_b", 1},
    {"skip_create_output", 1}
  };

  if (index == 0) {
    std::vector<size_t> a_shape_1 = {
      (size_t)num_heads_, maxSeqLength(), (size_t)head_size_
    };
    std::vector<size_t> w_shape_1 = {
      (size_t)kv_num_heads_, (size_t)head_size_, maxSeqLength()
    };
    ss_->bmm1_->set_params("BMM", a_shape_1, w_shape_1, attr);
  } else {
    std::vector<size_t> a_shape_2 = {
      (size_t)num_heads_, maxSeqLength(), maxSeqLength()
    };
    std::vector<size_t> w_shape_2 = {
      (size_t)kv_num_heads_, maxSeqLength(), (size_t)head_size_
    };

    ss_->bmm2_->set_params("BMM", a_shape_2, w_shape_2, attr);
  }
}

void AMDGQOKernel::initializeKernels() {
  if (use_aie_rope_) {
    if (!ss_->rope_) {
      std::string transpose_type = "input";
      auto rope_attr = getCommonAttrs();
      rope_attr["transpose"] = transpose_type;
      if (rotary_interleaved_) {
        std::string modelname = "CHATGLM";
        rope_attr["model_name"] = modelname;
      }
      rope_attr["head_dim"] = static_cast<int>(head_size_);

      ss_->rope_ =
        std::make_unique<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, rope_attr
        );
    }
  }

  bool reinit = false;

  if (!use_flash_mha_) {
    if (!ss_->bmm1_) {
      auto attr = getCommonAttrs();
      attr["head_dim"] = static_cast<int>(head_size_);

      ss_->bmm1_ = std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", "bfloat16", "bfloat16", true, true, attr
      );

      reinit = true;

      constexpr size_t kBMM1Idx = 0;

      set_params_bmm(kBMM1Idx);

      ss_->bmm1_->debug(false);

      ss_->bmm1_inputs_ = ss_->bmm1_->get_inputs();
      ss_->bmm1_outputs_ = ss_->bmm1_->get_outputs();
    }

    if (!ss_->bmm2_) {
      auto attr = getCommonAttrs();
      attr["head_dim"] = static_cast<int>(head_size_);

      ss_->bmm2_ = std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", "bfloat16", "bfloat16", true, false, attr
      );

      reinit = true;

      constexpr size_t kBMM2Idx = 1;

      set_params_bmm(kBMM2Idx);

      ss_->bmm2_->debug(false);

      ss_->bmm2_inputs_ = ss_->bmm2_->get_inputs();
      ss_->bmm2_outputs_ = ss_->bmm2_->get_outputs();
    }

    // this stage either indicates BMM1 and/or BMM2 got reset
    if (reinit) {
      ss_->bmm2_inputs_[0] = ss_->bmm1_outputs_[0];
    }

    if (!ss_->softmax_) {
      auto attr_softmax = getCommonAttrs();
      attr_softmax.insert(
        {{"skip_create_input", 1},
         {"skip_create_output", 1},
         {"headsize", static_cast<int>(head_size_)}}
      );

      ss_->softmax_ =
        std::make_unique<ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, attr_softmax
        );

      reinit = true;

      ss_->softmax_->debug(false);

      // clang-format off
      /*
      skip creating input BO for first input of bmm2 and reuse output BO of bmm1
      this works due to softmax being able to operate in place otherwise output of
      softmax would need to go to bmm2_in[0]
      bmm1_in[0] ---> bmm1 -> bmm1_out[0] -> softmax -> bmm1_out[0] --> bmm2 -> bmm2_out[0]
      bmm1_in[1] --/          mask -------/             bmm2_in[1]----/
      */
      // clang-format on

      ss_->softmax_mask_ = ss_->softmax_->get_inputs()[1];
    }

    if (reinit) {
      ss_->seq_len_ = 0;
      ss_->rewind_pos_ = -1;
      ss_->curr_local_window_size_ = -1;
    }
  } else {
    if (!flash_mha_) {
      auto attr = getCommonAttrs();
      attr["disable_gm"] = true;
      // normal flash mha use DPU_3, sliding window mha use DPU_8
      std::string pdi_name = local_window_size_ > 0 ? "DPU_8" : "DPU_3";
      attr["pdi_name"] = pdi_name;
      if (local_window_size_ > 0) {
        attr["window_size"] = local_window_size_;
      }
      flash_mha_ =
        std::make_unique<ryzenai::dynamic_dispatch::transformer::flash_mha<
          uint16_t, uint16_t, uint16_t>>(
          "bfloat16", "bfloat16", "bfloat16", true, attr
        );
      ss_->seq_len_ = 0;
      ss_->rewind_pos_ = -1;
      ss_->curr_local_window_size_ = -1;
    }
  }
}

void AMDGQOKernel::initializeMatMulNBitsKernel() {
  if (!ss_->gemm_) {
    auto mm_attrs = getCommonAttrs();
    updateAttrsForSharedWeights(mm_attrs);

    ss_->gemm_ = std::make_unique<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, mm_attrs
    );
  }
}

void AMDGQOKernel::initializeMatMulNBits(
  const std::vector<int8_t>& b, const std::vector<int8_t>& zeros,
  const std::vector<float>& scales, const std::vector<float>& bias
) {
  initializeMatMulNBitsKernel();

  std::vector<size_t> b_shape_dd = {
    static_cast<size_t>(matmulnbits_attrs_.k),
    static_cast<size_t>(matmulnbits_attrs_.n)
  };

  Tensor weight_tensor = {const_cast<int8_t*>(b.data()), b_shape_dd, "int4"};
  Tensor bias_tensor = {
    const_cast<float*>(bias.data()),
    {(size_t)matmulnbits_attrs_.block_size, 0},
    "float"
  };
  Tensor scales_tensor = {
    const_cast<float*>(scales.data()),
    {(size_t)matmulnbits_attrs_.block_size, 0},
    "float"
  };
  Tensor zeros_tensor = {const_cast<int8_t*>(zeros.data()), b_shape_dd, "int4"};
  std::vector<Tensor> constant_tensors = {
    weight_tensor, bias_tensor, scales_tensor, zeros_tensor
  };
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["group_size"] = static_cast<int>(matmulnbits_attrs_.block_size);
  attrs["max_m"] = maxSeqLength();
  attrs["skip_create_input"] = 1;
  attrs["skip_create_output"] = 1;
  updateAttrsForSharedWeights(attrs);
  ss_->gemm_->initialize_const_params(constant_tensors, attrs);
}

void AMDGQOKernel::parseUnpackedWeights(const Ort::ConstKernelInfo& info) {
  // Get weights
  auto is_constant = 0;
  auto weights_tensor = info.GetTensorConstantInput(kMmWtsIdx, &is_constant);

  // Get scales
  is_constant = 0;
  auto scales_tensor = info.GetTensorConstantInput(kMmScaleIdx, &is_constant);

  // Get zero-points
  is_constant = 0;
  auto zeros_tensor = info.GetTensorConstantInput(kMmZpIdx, &is_constant);
  bool is_asymmetric = is_constant;

  auto k = matmulnbits_attrs_.k;
  auto n = matmulnbits_attrs_.n;
  auto bits = matmulnbits_attrs_.bits;
  auto block_size = matmulnbits_attrs_.block_size;

  std::vector<float> scales(k * n / block_size);
  std::vector<int8_t> b(k * n, 0);
  size_t kblks = k / block_size;
  // fill this with zeros for Symmetric quantization
  size_t zp_shape = (n * std::floor((float)((kblks + 1) * bits) / 8.0f));
  // std::vector<int8_t> zeros(k_k * k_n / k_block_size, 0);
  std::vector<int8_t> zeros(zp_shape * 2, 0);
  std::vector<float> bias(n, 0);  // fill with zeros

  // Original weights are in NxK/2 packed as uint8
  // Convert to KXN uint8
  // mladf version 2 only support uint input, so no need to correct data to
  // int
  uint8_t wts_type_off = 8;
  const uint8_t* wts = weights_tensor.GetTensorData<uint8_t>();
  for (int64_t i = 0; i < k; i += 2) {
    for (int64_t j = 0; j < n; j++) {
      auto srcv = wts[j * k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - wts_type_off;
      auto src1 = ((srcv & 0xf0) >> 4) - wts_type_off;
      b[i * n + j] = static_cast<int8_t>(src0);
      b[(i + 1) * n + j] = static_cast<int8_t>(src1);
    }
  }

  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BlockSize)xN shape
  const auto* scl = scales_tensor.GetTensorData<Ort::Float16_t>();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < kblks; j++) {
      scales[j * n + i] = scl[i * kblks + j].ToFloat();
    }
  }

  // fill this with zeros for Symmetric quantization
  if (is_asymmetric) {
    const uint8_t* zero_pt = zeros_tensor.GetTensorData<uint8_t>();
    int kblks_pad = 2 * zp_shape / n;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < kblks_pad; j = j + 2) {
        // auto zpv = zero_pt[(i * (kblks / 2)) + (j / 2)];
        auto zpv = zero_pt[(i * kblks_pad) / 2 + j / 2];
        zeros[j * n + i] = (zpv & 0xf) - wts_type_off;
        zeros[(j + 1) * n + i] = ((zpv & 0xf0) >> 4) - wts_type_off;
      }
    }
  }

  // fill this with zeros for MatMul without bias
  // TODO(varunsh): if bias is needed, then get it from the input
  // auto biased = false;
  // if (biased) {
  //   const float* m_bias_ptr = m_bias_.GetTensorData<float>();
  //   MemCpy(bias.data(), m_bias_ptr, sizeof(float) * n);
  // }

  initializeMatMulNBits(b, zeros, scales, bias);
}

void AMDGQOKernel::initializeMatMulNBits(
  const uint8_t* preformat_consts, size_t size, bool use_shared_buffer
) {
  initializeMatMulNBitsKernel();

  std::vector<size_t> consts_shape_dd = {size};

  Tensor packed_const_tensor = {
    const_cast<uint8_t*>(preformat_consts), consts_shape_dd, "uint8"
  };
  Tensor weight_tensor = {
    nullptr,
    {static_cast<size_t>(matmulnbits_attrs_.k),
     static_cast<size_t>(matmulnbits_attrs_.n)},
    "int4"
  };
  std::vector<Tensor> constant_tensors = {weight_tensor, packed_const_tensor};
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["group_size"] = static_cast<int>(matmulnbits_attrs_.block_size);
  attrs["max_m"] = maxSeqLength();
  attrs["num_preformat_tensors"] = 1;
  attrs["tensor_size"] = use_shared_buffer ? static_cast<int>(size) : 0;
  attrs["use_host_buffer"] = 1;
  attrs["skip_create_input"] = 1;
  attrs["skip_create_output"] = 1;
  updateAttrsForSharedWeights(attrs);
  ss_->gemm_->initialize_const_params(constant_tensors, attrs);
}

const std::vector<size_t>& AMDGQOKernel::setFlashMhaGranularity() const {
  const auto& version = mladfVersion();
  switch (version) {
    case MladfVersion::v2:
      return kFlashMhaGranularityAie2_;
    case MladfVersion::aie4_v1:
      return kFlashMhaGranularityAie4_;
    default:
      throw std::runtime_error("Unsupported MLADF version: " + version.str());
  }
}

AMDGQOKernel::AMDGQOKernel(
  const OrtKernelInfo* k_info, const OrtApi& api,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : JitNode(this, session_configs),
    NpuOp(k_info, session_configs),
    ss_(session_configs) {
#ifdef NPU_GQO_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};
  auto num_inputs = info.GetInputCount();
  auto header = initializeNpuOp(op_type_, session_configs, info);
  // Get attrs
  do_rotary_ = getAttribute<int64_t>(info, "do_rotary");
  kv_num_heads_ = getAttribute<int64_t>(info, "kv_num_heads");
  local_window_size_ =
    getAttribute<int64_t>(info, "local_window_size", -1) > maxSeqLength()
      ? -1
      : getAttribute<int64_t>(info, "local_window_size", -1);

  num_heads_ = getAttribute<int64_t>(info, "num_heads");
  rotary_interleaved_ = getAttribute<int64_t>(info, "rotary_interleaved");
  scale_ = getAttribute<float>(info, "scale");
  auto softcap = getAttribute<float>(info, "softcap", 0.0f);
  rotary_embedding_dim_ =
    getAttribute<int64_t>(info, "rotary_embedding_dim", 0);
  auto default_head_size = info.GetInputTypeInfo(kPastKeyIdx)
                             .GetTensorTypeAndShapeInfo()
                             .GetShape()
                             .back();
  head_size_ = getAttribute<int64_t>(info, "head_size", default_head_size);
  if (rotary_embedding_dim_ == 0) rotary_embedding_dim_ = head_size_;
  input_cast_indices_ = getAttributes<int64_t>(
    info, "hybrid_llm_cast_input", std::vector<int64_t>({0})
  );
  output_cast_indices_ = getAttributes<int64_t>(
    info, "hybrid_llm_cast_output", std::vector<int64_t>({0, 1, 2})
  );
  shared_weights_.setWeightsCount(1);
  shared_weights_.setWeightKey(0, info, "wts_hash");

  setMladfVersion(info);
  mha_aie_kernel_info_.setVersion(mladfVersion());
  rope_cache_name_ = info.GetInputName(kCosIdx);

  auto is_const_cache =
    setUseAieRope(info, rotary_embedding_dim_, session_configs);
  setUseAieGqa(session_configs);
  setUseFlashMHA(session_configs);
  // note: order is important here
  // if flash mha is set will disable context chunk
  setEnableContextChunk(session_configs);
  setContextChunkSize(session_configs);

  initializeSharedBuffers(ss_->bo_data_);

  // optional input head_sink
  has_head_sink_ = (kHeadSinkIdx < num_inputs);

  if (has_head_sink_) {
    int is_head_sink_const = 0;
    auto head_sink_tensor =
      info.GetTensorConstantInput(kHeadSinkIdx, &is_head_sink_const);

    if (is_head_sink_const) {
      getHeadSink(head_sink_, head_sink_tensor);
    } else {
      head_sink_.assign(num_heads_, 0.0f);
      has_head_sink_ = false;
    }
    if (local_window_size_ > 0) {
      sink_data_ = allocator_.AllocateBuffer(num_heads_ * sizeof(uint16_t));
      if (has_head_sink_) {
        ryzenai::float_buffer_to_bfloat16(
          head_sink_.data(), num_heads_, (uint16_t*)sink_data_.Data()
        );

      } else {
        // if op doesn't has sink, NPU flatmha needs 0xFF7F for const buffer
        // initialization
        std::fill(
          (uint16_t*)sink_data_.Data(),
          ((uint16_t*)sink_data_.Data()) + num_heads_, uint16_t(0xFF7F)
        );
      }
    }
  }

  // transpose [0, 1, 2, 3] to [0, 2, 1, 3]
  ort_transpose_.construct(info, {0, 2, 1, 3});

  ort_cast_fp16_to_bf16_.construct(info);
  ort_cast_fp16_to_fp32_.construct(info);
  ort_cast_bf16_to_fp16_.construct(info);
  ort_cast_bf16_to_fp32_.construct(info);

  ort_rope_q_.construct(
    info, scale_, rotary_interleaved_, num_heads_, rotary_embedding_dim_
  );

  ort_rope_k_.construct(
    info, scale_, rotary_interleaved_, kv_num_heads_, rotary_embedding_dim_
  );

  ort_gqa_.construct(
    info, do_rotary_, scale_, rotary_interleaved_, num_heads_, kv_num_heads_,
    rotary_embedding_dim_, softcap, local_window_size_, has_head_sink_
  );

  initializeKernels();

  // initialize the atten_provider
  atten_mask_provider_ = std::make_unique<AttnMaskProvider>(
    &mha_aie_kernel_info_, ss_->attn_mask_lut_weak_
  );

  // requires setUseAieRope to have been called
  if (use_aie_rope_ && is_const_cache && !ss_->cos_sin_cache_) {
    std::tie(ss_->cos_sin_cache_, ss_->cos_shape_) = getRopeCache(
      &allocator_, const_cos_, const_sin_, rotary_interleaved_, mladfVersion()
    );
  }

  //  Get weights
  // int is_constant = 0;

  //  Extracting the attributes for MatMul Nbits
  matmulnbits_attrs_.n = info.GetAttribute<int64_t>("o_proj_N");
  matmulnbits_attrs_.k = info.GetAttribute<int64_t>("o_proj_K");
  matmulnbits_attrs_.bits = info.GetAttribute<int64_t>("o_proj_bits");
  matmulnbits_attrs_.block_size =
    info.GetAttribute<int64_t>("o_proj_block_size");

  const auto last_input_name = info.GetInputName(info.GetInputCount() - 1);
  const bool packed_consts = endsWith(last_input_name, "packed");

  if (!packed_consts) {
    parseUnpackedWeights(info);
  } else {
    int is_constant = 0;
    m_packed_consts_ =
      info.GetTensorConstantInput(info.GetInputCount() - 1, &is_constant);

    // no JIT
    if (m_packed_consts_.GetTensorTypeAndShapeInfo().GetElementCount() > 1) {
      const uint8_t* value = m_packed_consts_.GetTensorData<uint8_t>();
      initializeMatMulNBits(
        value, m_packed_consts_.GetTensorTypeAndShapeInfo().GetElementCount()
      );
    } else {  // with JIT
      initializeMatMulNBitsKernel();
      if (!shared_weights_.isEnabled()) {
        jit_tensor_info_ = getExternalTensorInfo(header.get(), name(), -1);
        ss_->jit_max_bo_size_ = proto::getNpuMaxSize(header.get(), op_type_);
        loadFirstData();
      }
    }
    const auto& flash_mha_granularity = setFlashMhaGranularity();
    q_seq_len_ = getNPUKernelGranularity(
      getInitPromptSize(session_configs),
      use_flash_mha_ ? flash_mha_granularity
                     : std::vector<size_t>{1024, 2048, 3072, 4096}
    );
    npu_kernel_size_q_ = q_seq_len_;
    UpdateSharedBuffer(q_seq_len_);
  }

  initializeLora();

  // Register for LoRA loading via LoraCompute/LoraComputeFromBuffer
  Lora::addLoraOp(this);

  cnt_ = ss_->instances_++;

#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point config_end = Clock::now();
  const Duration config_duration = config_end - config_start;
  measurements_.at(EventID::CONFIG_ID) += config_duration;
#endif
}

void AMDGQOKernel::initializeLora() {
  if (Lora::isEnabled()) {
    if (!ss_->gemm_) {
      throw std::runtime_error(
        "Gemm is not initialized before loading LoRA data"
      );
    }
    lora_buffers_.addBo(
      matmulnbits_attrs_.k, matmulnbits_attrs_.n, ss_->gemm_.get(), allocator_
    );
  }
}

void AMDGQOKernel::loadLoraData() {
  ExternalTensorInfo jit_tensor_info =
    getExternalTensorInfo(Lora::prefillHeader(), name(), -1);
  if (jit_tensor_info.size > 0) {
    Lora::loadBinData(
      lora_buffers_.data(0), jit_tensor_info.size, jit_tensor_info.offset
    );
  } else {
    // Need to clean up the previous lora buffers even if no new lora data is
    // loaded
    lora_buffers_.zeroesAllLoraData();
  }
  lora_buffers_.syncLora();
}

void AMDGQOKernel::LoadLora() {
  const auto& lora_name = Lora::getLoraName();
  if (lora_name != lora_buffers_.getLoraName()) {
    lora_buffers_.setLora(lora_name);
    if (lora_name != "base") {
      loadLoraData();
    } else {
      lora_buffers_.zeroesLoraData(0);
      lora_buffers_.syncLora();
    }
  }
}

void AMDGQOKernel::readDataImpl(int idx) {
  if (shared_weights_.isEnabled()) return;
  updateJitBuffer(
    dynamicJitFactor(), ss_->bo_data_[idx], jit_tensor_info_.size,
    ss_->jit_max_bo_size_
  );
  loadBin(
    ss_->bo_data_[idx].Data(), externalData().string(), jit_tensor_info_.size,
    jit_tensor_info_.offset
  );
}

void AMDGQOKernel::loadDataImpl(int idx) {
  auto value = (uint8_t*)ss_->bo_data_[idx].Data();

  // if we're reading everything, we don't need to use a shared buffer
  const bool use_shared_buffer = isJitEnabled();

  initializeMatMulNBits(value, ss_->bo_data_[idx].Size(), use_shared_buffer);
}

AMDGQOKernel::~AMDGQOKernel() {
  if (trig_max_len_)
#ifdef _WIN32
    _aligned_free(trig_max_len_);
#else
    free(trig_max_len_);
  ss_->gemm_.reset();
  ryzenai::dynamic_dispatch::xrt_context::destroy_ctx_map();
#endif

#ifdef NPU_GQO_PROFILE_EN
  std::ostringstream os;

  int event_id = 0;

  for (const auto& measurement : measurements_) {
    os << name() << ",NPUGQO," << event_id << ","
       << MillisecondsFp{measurement}.count() << "\n";
    event_id++;
  }

  std::cout << os.str() << std::flush;
#endif

  ss_->instances_--;
  if (ss_->instances_ == 0) {
    for (auto i = 0; i < readAhead(); ++i) {
      ss_->bo_data_[i] = {};
      shared_buffer_.Reset();
    }
    ss_->cos_sin_cache_ = {};
  }
}

uint16_t* AMDGQOKernel::rope_aie_execute(
  const uint16_t* input, const int N, const int S, const int H, const int pad_S,
  bool b_isK, int rewind_pos
) {
  std::vector<size_t> in_shape{
    (size_t)N, (size_t)pad_S, (size_t)H, (size_t)rotary_embedding_dim_
  };
  ss_->rope_->set_params("rope", in_shape);
  ss_->rope_inbos_ = ss_->rope_->get_inputs();
  ss_->rope_outbos_ = ss_->rope_->get_outputs();
  // uint16_t* rope_out = rope_outbos_[0].map<uint16_t*>();
  uint16_t* a_bo_map = ss_->rope_inbos_[0].map<uint16_t*>();
  MemCpy((void*)a_bo_map, (void*)input, N * S * H * sizeof(uint16_t));

  const size_t in_bo_size = N * S * H * sizeof(uint16_t);
  const size_t in_bo_offset = 0;

  RecordDuration(Metric::XRTBOSync, [&]() {
    // partial sync makes llama3-8B get incorrect result if prompt=1800, so
    // sync whole bo.
    ss_->rope_inbos_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  });

  uint16_t* b_bo_map = ss_->rope_inbos_[1].map<uint16_t*>();

  auto trig_max_len_offset =
    ss_->cos_shape_[0] * ss_->cos_shape_[1] * ROPE_WTS_EXPAND_FACTOR;

  int b_rewindpos = rewind_pos * ss_->cos_shape_[1] * ROPE_WTS_EXPAND_FACTOR;
  if (S == pad_S) {
    auto b_bo_map_offset = S * ss_->cos_shape_[1] * ROPE_WTS_EXPAND_FACTOR;
    MemCpy(
      (void*)b_bo_map,
      (void*)(reinterpret_cast<uint16_t*>(ss_->cos_sin_cache_.Data()) +
              b_rewindpos),
      b_bo_map_offset * sizeof(uint16_t)
    );
    MemCpy(
      (void*)(b_bo_map + b_bo_map_offset),
      (void*)(reinterpret_cast<uint16_t*>(ss_->cos_sin_cache_.Data()) +
              b_rewindpos + trig_max_len_offset),
      b_bo_map_offset * sizeof(uint16_t)
    );

    const size_t rope_cache_bo_size = 2 * b_bo_map_offset * sizeof(uint16_t);
    const size_t rope_cache_bo_offset = 0;

    RecordDuration(Metric::XRTBOSync, [&]() {
      ss_->rope_inbos_[1].sync(
        XCL_BO_SYNC_BO_TO_DEVICE, rope_cache_bo_size, rope_cache_bo_offset
      );
    });
  } else {
    auto b_bo_size = S * ss_->cos_shape_[1] * ROPE_WTS_EXPAND_FACTOR;
    auto b_bo_map_offset = pad_S * ss_->cos_shape_[1] * ROPE_WTS_EXPAND_FACTOR;
    MemCpy(
      (void*)b_bo_map,
      (void*)(reinterpret_cast<uint16_t*>(ss_->cos_sin_cache_.Data()) +
              b_rewindpos),
      b_bo_size * sizeof(uint16_t)
    );
    MemCpy(
      (void*)(b_bo_map + b_bo_map_offset),
      (void*)(reinterpret_cast<uint16_t*>(ss_->cos_sin_cache_.Data()) +
              b_rewindpos + trig_max_len_offset),
      b_bo_size * sizeof(uint16_t)
    );

    const size_t rope_cache_bo_size = 2 * b_bo_map_offset * sizeof(uint16_t);
    const size_t rope_cache_bo_offset = 0;

    RecordDuration(Metric::XRTBOSync, [&]() {
      ss_->rope_inbos_[1].sync(
        XCL_BO_SYNC_BO_TO_DEVICE, rope_cache_bo_size, rope_cache_bo_offset
      );
    });
  }

  if (b_isK) {
    ss_->rope_outbos_[0] = ss_->bmm1_inputs_[1];
    RecordDuration(Metric::XRTBOSync, [&]() {
      ss_->rope_outbos_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    });
    tryContinueOnException([&]() {
      RecordDuration(Metric::KernelExecution, [&]() {
        ss_->rope_->execute(
          ss_->rope_inbos_, ss_->rope_outbos_, !continueOnException()
        );
      });
    });
    RecordDuration(Metric::XRTBOSync, [&]() {
      ss_->rope_outbos_[0].sync(
        XCL_BO_SYNC_BO_FROM_DEVICE, in_bo_size, in_bo_offset
      );
    });
  } else {  // this is Q
    ss_->rope_outbos_[0] = ss_->bmm1_inputs_[0];
    RecordDuration(Metric::XRTBOSync, [&]() {
      ss_->rope_outbos_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    });
    conditionalTry( // async with rope k output bo sync and copy.
      [&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->rope_->execute(ss_->rope_inbos_, ss_->rope_outbos_, true);
        });
      },  // false->true! Q need sync now because it need re_pad!
      continueOnException(), name()
    );
    if (rewind_pos) {
      RecordDuration(Metric::XRTBOSync, [&]() {
        ss_->rope_outbos_[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, in_bo_size, in_bo_offset
        );
      });
    }
  }
  uint16_t* rope_out = ss_->rope_outbos_[0].map<uint16_t*>();

  return rope_out;
}

void execute_mha(
  ryzenai::dynamic_dispatch::transformer::flash_mha<
    uint16_t, uint16_t, uint16_t>* flash_mha,
  xrt::bo& in, std::vector<size_t> kernel_shape, size_t S_q, size_t S_k,
  size_t buffer_s, std::vector<xrt::bo>& bmm2_outputs, uint16_t* xCasted,
  uint16_t* yCasted, uint16_t* zCasted, std::vector<size_t> qkv_out_size_,
  bool wait, bool continue_on_exception, const std::string& name,
  bool use_aie_rope, bool en_cpy, int rewind_pos, xrt::bo& sink_bo
) {
  // Get XRT Buffers
  auto x_elements = qkv_out_size_.at(0);
  auto y_elements = qkv_out_size_.at(1);
  auto z_elements = qkv_out_size_.at(2);
  if (!use_aie_rope || rewind_pos || (buffer_s != kernel_shape.at(2)) ||
      en_cpy) {
    uint16_t* a_bo_map = in.map<uint16_t*>();  // Q
    memset(a_bo_map, 0, in.size());
    memcpy((void*)a_bo_map, (void*)xCasted, x_elements);

    uint16_t* b_bo_map =  // K
      in.map<uint16_t*>() + x_elements / sizeof(uint16_t);
    memcpy((void*)b_bo_map, (void*)yCasted, y_elements);

    uint16_t* c_bo_map =  // V
      in.map<uint16_t*>() + (x_elements + y_elements) / sizeof(uint16_t);
    memcpy((void*)c_bo_map, (void*)zCasted, z_elements);
    // Sync data
    in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // Execute QKT MatMul
  // std::vector<xrt::bo> inputs = {in};
  std::vector<NPUBufferSpan> inputs = {{in, 0, in.size()}};
  if (sink_bo) {
    NPUBufferSpan sink_tensor = {sink_bo, 0, sink_bo.size()};
    inputs.push_back(sink_tensor);
  }
  std::vector<NPUBufferSpan> outputs = {
    {bmm2_outputs[0], 0, qkv_out_size_.at(3)}
  };
  if (wait)
    conditionalTry(
      [&]() { flash_mha->run(inputs, outputs); }, continue_on_exception, name
    );
  else {
    std::map<std::string, std::any> attr = {};
    auto run = flash_mha->create_run(inputs, outputs, attr);
    run.value().start();
  }
}

void execute_bmm1_npu(
  ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm1_inputs,
  std::vector<xrt::bo>& bmm1_outputs, uint16_t* xCasted, size_t x_elements,
  uint16_t* yCasted, size_t y_elements, bool wait, bool continue_on_exception,
  const std::string& name, bool use_aie_rope, int rewind_pos
) {
  // Get XRT Buffers
  if (!use_aie_rope || rewind_pos) {
    uint16_t* a_bo_map = bmm1_inputs[0].map<uint16_t*>();  // Q
    epx.MemCpy((void*)a_bo_map, (void*)xCasted, x_elements * sizeof(uint16_t));
    uint16_t* b_bo_map = bmm1_inputs[1].map<uint16_t*>();
    epx.MemCpy((void*)b_bo_map, (void*)yCasted, y_elements * sizeof(uint16_t));

    const size_t bmm1_in0_size = x_elements * sizeof(uint16_t);
    const size_t bmm1_in1_size = y_elements * sizeof(uint16_t);
    const size_t bmm1_offset = 0;

    // Sync data
    epx.RecordDuration(Metric::XRTBOSync, [&]() {
      bmm1_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, bmm1_in0_size, bmm1_offset);
      bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE, bmm1_in1_size, bmm1_offset);
    });
    // bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // Execute QKT MatMul
  conditionalTry(
    [&]() {
      epx.RecordDuration(Metric::KernelExecution, [&]() {
        bmm->execute(bmm1_inputs, bmm1_outputs, wait);
      });
    },
    continue_on_exception, name
  );
}

/**
 * This is a CPU implementation of BMM1, copied from
 * tests/cpp/include/bmm_helpers.hpp in DynamicDispatch.
 */
template <typename Tx, typename Tw, typename Ty>
void cpu_bmmb1(Tx X, Tw W, Ty Y, bool trans) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      float acc = 0.0;
      for (int k = 0; k < X.num_cols; ++k) {
        float fx = bfloat16_to_float_single(X.at(r, k));
        float fw = 0.0;
        if (trans) {
          fw = bfloat16_to_float_single(W.at(c, k));
        } else {
          fw = bfloat16_to_float_single(W.at(k, c));
        }
        acc += fx * fw;
      }
      Y.at(r, c) = float_to_bfloat16(acc);
    }
  }
}

void execute_bmm1_cpu(
  uint16_t* x, uint16_t* y, uint16_t* out, int b0, int b1, int m, int k, int n
) {
  using matmul_matrix::RowMajorMatrix;
  for (int i = 0; i < b0; i++) {
    RowMajorMatrix<uint16_t> XX(m, k, x + i * m * k);
    int dv = b0 / b1;
    RowMajorMatrix<uint16_t> WW(n, k, y + (i / dv) * k * n);
    RowMajorMatrix<uint16_t> cpu_YY(m, n, out + i * m * n);
    cpu_bmmb1<
      RowMajorMatrix<uint16_t>, RowMajorMatrix<uint16_t>,
      RowMajorMatrix<uint16_t>>(XX, WW, cpu_YY, true);
  }
}

void execute_bmm1(
  bool use_cpu, ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm1_inputs,
  std::vector<xrt::bo>& bmm1_outputs, uint16_t* xCasted, uint16_t* yCasted,
  int batch_size, int num_heads, int seq_query_len, int head_size,
  int kv_num_heads, int seq_key_len, bool wait, bool continue_on_exception,
  const std::string& name, bool use_aie_rope, int rewind_pos
) {
  if (use_cpu) {
    auto* out = bmm1_outputs[0].map<uint16_t*>();
    execute_bmm1_cpu(
      xCasted, yCasted, out, num_heads, kv_num_heads, seq_query_len, head_size,
      seq_key_len
    );
    epx.RecordDuration(Metric::XRTBOSync, [&]() {
      bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    });
  } else {
    auto x_elements = batch_size * num_heads * seq_query_len * head_size;
    auto y_elements = batch_size * kv_num_heads * seq_key_len * head_size;
    execute_bmm1_npu(
      bmm, epx, bmm1_inputs, bmm1_outputs, xCasted, x_elements, yCasted,
      y_elements, wait, continue_on_exception, name, use_aie_rope, rewind_pos
    );
  }
}

void execute_bmm2_npu(
  ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm2_inputs,
  std::vector<xrt::bo>& bmm2_outputs, uint16_t* yCasted, size_t y_elements,
  bool wait, bool continue_on_exception, const std::string& name
) {
  // Get SMV MatMul buffers
  uint16_t* value_bo_map = bmm2_inputs[1].map<uint16_t*>();
  epx.MemCpy(
    (void*)value_bo_map, (void*)yCasted, y_elements * sizeof(uint16_t)
  );

  // Sync
  const size_t bmm2_in1_size = y_elements * sizeof(std::uint16_t);
  const size_t bmm2_in1_offset = 0;
  epx.RecordDuration(Metric::XRTBOSync, [&]() {
    bmm2_inputs[1].sync(
      XCL_BO_SYNC_BO_TO_DEVICE, bmm2_in1_size, bmm2_in1_offset
    );
  });

  // Execute SMV MatMul
  conditionalTry(
    [&]() {
      epx.RecordDuration(Metric::KernelExecution, [&]() {
        bmm->execute(bmm2_inputs, bmm2_outputs, !continue_on_exception);
      });
    },
    continue_on_exception, name
  );
}

/**
 * This is a CPU implementation of BMM2, copied from
 * tests/cpp/include/bmm_helpers.hpp in DynamicDispatch.
 */
template <typename Tx, typename Tw, typename Ty>
void cpu_bmmb2(Tx X, Tw W, Ty Y, int B0, int B1) {
  // golden computation including transpose of heads in ofm
  int M = Y.num_rows / B0;
  int K = X.num_cols;
  int N = Y.num_cols;
  int dv = B0 / B1;

  for (int b = 0; b < B0; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0;
        for (int k = 0; k < K; ++k) {
          float fx = bfloat16_to_float_single(X.at(b * M + m, k));
          float fw = bfloat16_to_float_single(W.at((b / dv) * K + k, n));
          acc += fx * fw;
        }
        Y.at(m * B0 + b, n) = float_to_bfloat16(acc);
      }
    }
  }
}

void execute_bmm2_cpu(
  uint16_t* x, uint16_t* y, uint16_t* out, int b0, int b1, int m, int k, int n
) {
  using matmul_matrix::RowMajorMatrix;
  RowMajorMatrix<uint16_t> XX(b0 * m, k, x);
  RowMajorMatrix<uint16_t> WW(b1 * k, n, y);
  RowMajorMatrix<uint16_t> cpu_YY(m * b0, n, out);
  cpu_bmmb2(XX, WW, cpu_YY, b0, b1);
}

void execute_bmm2(
  bool use_cpu, ryzenai::bmm<uint16_t, uint16_t, uint16_t>* bmm,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm2_inputs,
  std::vector<xrt::bo>& bmm2_outputs, uint16_t* yCasted, int batch_size,
  int num_heads, int seq_query_len, int head_size, int kv_num_heads,
  int seq_key_len, bool wait, bool continue_on_exception,
  const std::string& name
) {
  if (use_cpu) {
    epx.RecordDuration(Metric::XRTBOSync, [&]() {
      bmm2_inputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      bmm2_inputs[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    });
    auto* xCasted = bmm2_inputs[0].map<uint16_t*>();
    auto* out = bmm2_outputs[0].map<uint16_t*>();
    execute_bmm2_cpu(
      xCasted, yCasted, out, num_heads, kv_num_heads, seq_query_len,
      seq_key_len, head_size
    );
    epx.RecordDuration(Metric::XRTBOSync, [&]() {
      bmm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    });
  } else {
    auto y_elements = batch_size * kv_num_heads * seq_key_len * head_size;
    execute_bmm2_npu(
      bmm, epx, bmm2_inputs, bmm2_outputs, yCasted, y_elements, wait,
      continue_on_exception, name
    );
  }
}

void execute_softmax_npu(
  ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>* softmax,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm1_outputs,
  xrt::bo& softmax_mask, std::vector<xrt::bo>& bmm2_inputs, uint16_t* mCasted,
  int seq_query_len, int seq_key_len, bool wait, bool continue_on_exception,
  const std::string& name, bool sync_mask
) {
  if (sync_mask) {
    uint16_t* mask_bo_map = softmax_mask.map<uint16_t*>();
    epx.MemCpy(
      (void*)mask_bo_map, (void*)mCasted,
      seq_query_len * seq_key_len * sizeof(uint16_t)
    );
    // Sync
    const size_t mask_size =
      seq_query_len * seq_key_len * sizeof(std::uint16_t);
    const size_t mask_offset = 0;
    epx.RecordDuration(Metric::XRTBOSync, [&]() {
      softmax_mask.sync(XCL_BO_SYNC_BO_TO_DEVICE, mask_size, mask_offset);
    });
  }

  // bmm2_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<xrt::bo> inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> outputs = {bmm2_inputs[0]};

  // Execute Softmax
  conditionalTry(
    [&]() {
      epx.RecordDuration(Metric::KernelExecution, [&]() {
        softmax->execute(inputs, outputs, wait);
      });
    },
    continue_on_exception, name
  );
}

void execute_softmax_cpu(
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm1_outputs,
  uint16_t* mCasted, std::vector<xrt::bo>& bmm2_inputs, float scale,
  int seq_query_len, int num_heads, int64_t local_window_size
) {
  // std::vector<size_t> softmax_shape{(size_t)N, (size_t)S_q, (size_t)S_q};
  //  std::cout << "executing softmax on cpu N = " << N << ", S_q = " << S_q
  //  << ", S_k = " << S_k << "\n";
  std::uint16_t* in_ptr = bmm1_outputs[0].map<std::uint16_t*>();
  std::uint16_t* out_ptr = bmm2_inputs[0].map<std::uint16_t*>();
  std::uint16_t* mask_ptr = mCasted;

  epx.RecordDuration(Metric::XRTBOSync, [&]() {
    bmm1_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  });

  const float d_k_scale = scale;

  size_t dim_2_stride = seq_query_len * seq_query_len;
  size_t dim_1_stride = seq_query_len;

  for (int i = 0; i < num_heads; i++) {
    for (int j = 0; j < seq_query_len; j++) {
      // this is inner most dim, which we will apply softmax on

      // step 1 scale and apply mask
      //        keep track of max

      float curr_max_val = std::numeric_limits<float>::lowest();

      std::vector<float> intermediate_buffer(seq_query_len, 0.0f);
      for (int k = 0; k <= j; k++) {
        int index = i * dim_2_stride + j * dim_1_stride + k;
        std::uint32_t val_int = in_ptr[index] << 16;

        int mask_index = j * dim_1_stride + k;
        std::uint32_t mask_int = mask_ptr[mask_index] << 16;

        float val = *reinterpret_cast<float*>(&val_int);
        float mask_val = *reinterpret_cast<float*>(&mask_int);
        val = d_k_scale * val + mask_val;

        if (val > curr_max_val) {
          curr_max_val = val;
        }

        intermediate_buffer.at(k) = val;
      }

      // step 2 calculate softmax denominator
      //        exponentiate values with re-bias using max

      float sum_exp = 0.0f;
      for (int k = 0; k <= j; k++) {
        intermediate_buffer.at(k) -= curr_max_val;
        intermediate_buffer.at(k) = std::expf(intermediate_buffer.at(k));
        sum_exp += intermediate_buffer.at(k);
      }

      float recip_sum_exp = 0;

      bool stable = (sum_exp > 5E-6);

      if (stable) {
        __m128 x = _mm_set_ss(sum_exp);
        x = _mm_rcp_ss(x);

        _mm_store_ss(&recip_sum_exp, x);
      } else {
        __m128 x = _mm_set_ss(j + 1.0f);
        x = _mm_rcp_ss(x);

        _mm_store_ss(&recip_sum_exp, x);
      }

      // step 3 apply normalization
      for (int k = 0; k <= j; k++) {
        int index = i * dim_2_stride + j * dim_1_stride + k;

        float val = intermediate_buffer.at(k);

        if (stable) {
          val = val * recip_sum_exp;
        } else {
          val = recip_sum_exp;
        }

        std::uint32_t val_int = *reinterpret_cast<std::uint32_t*>(&val);
        out_ptr[index] = (val_int >> 16) & 0xFFFF;
      }

      // zero out trailing part since its causal mask
      for (int k = j + 1; k < seq_query_len; k++) {
        int index = i * dim_2_stride + j * dim_1_stride + k;

        out_ptr[index] = 0;
      }

      // zero out beginning of local_window_size is not -1
      if (local_window_size != -1 && (local_window_size < j + 1)) {
        for (int k = 0; k < j - local_window_size + 1; k++) {
          int index = i * dim_2_stride + j * dim_1_stride + k;

          out_ptr[index] = 0;
        }
      }
    }
  }

  epx.RecordDuration(Metric::XRTBOSync, [&]() {
    bmm2_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  });
}

void execute_softmax(
  bool use_cpu, ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>* softmax,
  ryzenai::ExecutionProviderExtensions& epx, std::vector<xrt::bo>& bmm1_outputs,
  xrt::bo& softmax_mask, std::vector<xrt::bo>& bmm2_inputs, uint16_t* mCasted,
  int seq_query_len, int seq_key_len, float scale, int num_heads, bool wait,
  bool continue_on_exception, const std::string& name, bool sync_mask,
  int64_t local_window_size
) {
  if (use_cpu) {
    // auto* out = bmm1_outputs[0].map<uint16_t*>();
    execute_softmax_cpu(
      epx, bmm1_outputs, mCasted, bmm2_inputs, scale, seq_query_len, num_heads,
      local_window_size
    );
  } else {
    execute_softmax_npu(
      softmax, epx, bmm1_outputs, softmax_mask, bmm2_inputs, mCasted,
      seq_query_len, seq_key_len, wait, continue_on_exception, name, sync_mask
    );
  }
}

void AMDGQOKernel::executeMatMulNBitsAie(
  const uint16_t* input_data, uint16_t* out, std::vector<int64_t> input_shape,
  std::pair<size_t, size_t> wts_shape, int grp_size, int run_cnt
) {
  int cnt_wts = run_cnt;

  if (wait_for_data_) {
    if (isJitEnabled()) {
      cnt_wts = 0;
    }

    auto ready = weightsReady();
    if (ready < 0) {
      std::cerr << "Disable NPU JIT with `hybrid_opt_npu_read_ahead='-1'` in "
                   "session options\n";
      throw std::invalid_argument("Weights not loaded in " + name());
    }

    wait_for_data_ = false;
  }
  // Ryzen-AI implementation
  int M = input_shape[0] * input_shape[1];
  std::vector<size_t> a_shape = {static_cast<size_t>(M), wts_shape.first};

  std::vector<size_t> c_shape = {static_cast<size_t>(M), wts_shape.second};
  std::vector<size_t> wts_shape_dd = {wts_shape.first, wts_shape.second};
  ss_->gemm_->set_shape(a_shape, wts_shape_dd, grp_size);
  Tensor output_tensor = {out, c_shape, "bfloat16"};
  std::vector<Tensor> output_tensors = {output_tensor};
  if (auto rebind = shared_buffer_.Validate(
        "bmm2_out", ss_->gemm_->get_inputs(kGemmBOsSelector)[0]
      )) {
    ss_->gemm_->create_bo(rebind->ptr, rebind->len, 0, kGemmBOsSelector);
  }

  if (auto rebind = shared_buffer_.Validate(
        "gemm_out", ss_->gemm_->get_outputs(kGemmBOsSelector)[0]
      )) {
    ss_->gemm_->create_bo(rebind->ptr, rebind->len, 1, kGemmBOsSelector);
  }

  if (has_scratch_buffer_) {
    if (auto rebind = shared_buffer_.Validate(
          "scratch", ss_->gemm_last_scratch_ptr_, ss_->gemm_last_scratch_len_
        )) {
      ss_->gemm_->create_bo(rebind->ptr, rebind->len, 2, kGemmBOsSelector);

      ss_->gemm_last_scratch_ptr_ = rebind->ptr;
      ss_->gemm_last_scratch_len_ = rebind->len;
    }
  }

  if (M > 1) {
    // Exec
    // DD Tensors

    Tensor input_tensor = {(int16_t*)input_data, a_shape, "bfloat16"};
    std::vector<Tensor> input_tensors = {input_tensor};
    if (Lora::isEnabled()) {
      std::vector<size_t> shape = {lora_buffers_.getBoSize(0)};
      Tensor lora_tensor = {(int8_t*)lora_buffers_.data(0), shape, "int8"};
      input_tensors.push_back(lora_tensor);
    }
    tryContinueOnException([&]() {
      RecordDuration(Metric::KernelExecution, [&]() {
        ss_->gemm_->execute_internal(input_tensors, output_tensors, cnt_wts);
      });
    });
  } else {
    Tensor input_tensor = {(int16_t*)input_data, a_shape, "bfloat16"};
    std::vector<Tensor> input_tensors = {input_tensor};
    if (Lora::isEnabled()) {
      std::vector<size_t> shape = {lora_buffers_.getBoSize(0)};
      Tensor lora_tensor = {(int8_t*)lora_buffers_.data(0), shape, "int8"};
      input_tensors.push_back(lora_tensor);
    }
    tryContinueOnException([&]() {
      RecordDuration(Metric::KernelExecution, [&]() {
        ss_->gemm_->execute_internal(input_tensors, output_tensors, cnt_wts);
      });
    });
  }
}

void AMDGQOKernel::executeMatMulNBitsAie(
  std::vector<xrt::bo>& inputs, uint16_t* out,
  std::vector<int64_t> output_shape, std::pair<size_t, size_t> wts_shape,
  int grp_size, int run_cnt
) {
  int M = output_shape[0] * output_shape[1];

  if (auto rebind = shared_buffer_.Validate(
        "gemm_out", ss_->gemm_->get_outputs(kGemmBOsSelector)[0]
      )) {
    ss_->gemm_->create_bo(rebind->ptr, rebind->len, 1, kGemmBOsSelector);
  }

  if (has_scratch_buffer_) {
    if (auto rebind = shared_buffer_.Validate(
          "scratch", ss_->gemm_last_scratch_ptr_, ss_->gemm_last_scratch_len_
        )) {
      ss_->gemm_->create_bo(rebind->ptr, rebind->len, 2, kGemmBOsSelector);

      ss_->gemm_last_scratch_ptr_ = rebind->ptr;
      ss_->gemm_last_scratch_len_ = rebind->len;
    }
  }

  std::vector<size_t> a_shape = {static_cast<size_t>(M), wts_shape.first};

  std::vector<size_t> wts_shape_dd = {wts_shape.first, wts_shape.second};
  std::vector<size_t> c_shape = {static_cast<size_t>(M), wts_shape.second};

  int cnt_wts = run_cnt;
  if (wait_for_data_) {
    if (isJitEnabled()) {
      cnt_wts = 0;
    }

    auto ready = weightsReady();
    if (ready < 0) {
      std::cerr << "Disable NPU JIT with `hybrid_opt_npu_read_ahead='-1'` in "
                   "session options\n";
      throw std::invalid_argument("Weights not loaded in " + name());
    }

    wait_for_data_ = false;
  }

  ss_->gemm_->set_shape(a_shape, wts_shape_dd, grp_size);

  auto o_wts = ss_->gemm_->get_const();
  mm_outputs_ = ss_->gemm_->get_outputs(kGemmBOsSelector);
  // inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // mm_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  tryContinueOnException([&]() {
    RecordDuration(Metric::KernelExecution, [&]() {
      if (shared_weights_.ready()) {
        std::vector<xrt::bo> o_in = {inputs[0]};
        if (Lora::isEnabled()) {
          o_in.push_back(xrt::bo());  // dummy BO to fix input schema
          o_in.push_back(lora_buffers_.getBo(0));
        }
        std::vector<uint64_t> input_addrs = {
          0, shared_weights_.weightAddr(0), 0
        };
        std::vector<uint64_t> out_addrs;
        ss_->gemm_->execute(
          o_in, input_addrs, mm_outputs_, out_addrs, !continueOnException()
        );
      } else {
        std::vector<xrt::bo> o_in = {inputs[0], o_wts[cnt_wts]};
        if (Lora::isEnabled()) o_in.push_back(lora_buffers_.getBo(0));
        ss_->gemm_->execute(
          o_in, mm_outputs_,
          mladfVersion() == MladfVersion::aie4_v1 ? true
                                                  : !continueOnException()
        );
      }
    });
  });

  {
    const size_t mm_out_bo_size = M * wts_shape.second * sizeof(std::uint16_t);
    const size_t mm_out_offset = 0;
    RecordDuration(Metric::XRTBOSync, [&]() {
      mm_outputs_[0].sync(
        XCL_BO_SYNC_BO_FROM_DEVICE, mm_out_bo_size, mm_out_offset
      );
    });
    uint16_t* mm_out = mm_outputs_[0].map<uint16_t*>();
    MemCpy(out, mm_out, M * wts_shape.second * sizeof(uint16_t));
  }
}

void AMDGQOKernel::rebind_flashmha_params() {
  if ((local_window_size_ > 0) && !sink_bo_) {
    sink_bo_ = flash_mha_->bind_bo(
      sink_data_.Data(), head_sink_.size() * sizeof(uint16_t)
    );
    sink_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  // TODO: shared_buffer_ Validate disable, just for Gemma3 slding attention
  if (auto rebind = shared_buffer_.Validate("bmm2_out", flash_out_)) {
    const auto buffer_bmm1_in0 = shared_buffer_.Get("bmm1_in0");
    const auto buffer_bmm2_in1 = shared_buffer_.Get("bmm2_in1");
    const auto buffer_bmm1_in1 = shared_buffer_.Get("bmm1_in1");
    const auto buffer_bmm2_out = shared_buffer_.Get("bmm2_out");
    auto flash_in_data_size =
      qkv_out_size_.at(0) + qkv_out_size_.at(1) + qkv_out_size_.at(2);
    auto flash_out_data_size = qkv_out_size_.at(3);
    // #ifdef _WIN32
    //   flash_in_data_ptr_ = _aligned_malloc(flash_in_data_size, 4096);
    //   flash_out_data_ptr_ = _aligned_malloc(flash_out_data_size, 4096);
    // #else
    //   if (posix_memalign(&flash_in_data_ptr_, 4096, flash_in_data_size) != 0)
    //   {
    //     flash_in_data_ptr_ = nullptr;
    //   }
    //   if (posix_memalign(&flash_out_data_ptr_, 4096, flash_out_data_size) !=
    //   0) {
    //     flash_out_data_ptr_ = nullptr;
    //   }
    // #endif

    ss_->flash_in_ = flash_mha_->bind_bo(
      buffer_bmm1_in0.ptr,
      buffer_bmm1_in0.len + buffer_bmm1_in1.len + buffer_bmm2_in1.len
    );
    ss_->flash_out_ =
      flash_mha_->bind_bo(buffer_bmm2_out.ptr, buffer_bmm2_out.len);

    auto bmm1_in0 = xrt::bo(ss_->flash_in_, buffer_bmm1_in0.len, 0);
    auto bmm1_in1 =
      xrt::bo(ss_->flash_in_, buffer_bmm1_in1.len, buffer_bmm1_in0.len);
    auto bmm2_in1 = xrt::bo(
      ss_->flash_in_, buffer_bmm2_in1.len,
      buffer_bmm1_in1.len + buffer_bmm1_in0.len
    );

    ss_->bmm1_inputs_ = {bmm1_in0, bmm1_in1};
    ss_->bmm2_inputs_ = {bmm1_in1, bmm2_in1};
    ss_->bmm2_outputs_ = {ss_->flash_out_};
  }
  ss_->flash_in_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ss_->flash_out_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void AMDGQOKernel::rebind_bmm_params() {
  auto rebound = false;
  if (auto rebind =
        shared_buffer_.Validate("bmm1_in1", ss_->bmm1_->get_inputs()[1])) {
    ss_->bmm1_->create_bo(rebind->ptr, rebind->len, 1);
    rebound = true;
  }

  if (auto rebind =
        shared_buffer_.Validate("bmm1_out", ss_->bmm1_->get_outputs()[0])) {
    ss_->bmm1_->create_bo(rebind->ptr, rebind->len, 2);
    rebound = true;
  }

  if (auto rebind =
        shared_buffer_.Validate("bmm2_in1", ss_->bmm2_->get_inputs()[1])) {
    ss_->bmm2_->create_bo(rebind->ptr, rebind->len, 1);
    rebound = true;
  }

  if (auto rebind =
        shared_buffer_.Validate("bmm2_out", ss_->bmm2_->get_outputs()[0])) {
    ss_->bmm2_->create_bo(rebind->ptr, rebind->len, 2);
    rebound = true;
  }

  if (rebound) {
    ss_->bmm1_outputs_ = ss_->bmm1_->get_outputs();

    ss_->bmm2_inputs_ = ss_->bmm2_->get_inputs();
    ss_->bmm2_inputs_[0] = ss_->bmm1_outputs_[0];
    ss_->bmm2_outputs_ = ss_->bmm2_->get_outputs();
    // Get data pointers
    ss_->bmm1_inputs_ = {ss_->bmm2_outputs_[0], ss_->bmm1_->get_inputs()[1]};
  }

  if (is_prefill_) {
    // Note: bmm1 output is typically sized to [num_heads, total_seq, total_seq]
    //       if large prefill is run and there is no buffer resizing this can be
    //       large syncing on this during token phase is expensive Original
    //       reason seemed to be back to back prompts would give incorrect
    //       results
    ss_->bmm1_outputs_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

void static inline fill_values16b(
  std::uint16_t* start_ptr, uint16_t val, size_t count
) {
#if defined(_MSC_VER)
  __stosw(start_ptr, val, count);
#else
  std::fill_n(start_ptr, count, val);
#endif
}

struct row_mask_info {
  size_t start_index;
  size_t end_index;
};

void static inline populate_mask(
  std::vector<row_mask_info>& softmax_mask_info, const int query_start_pos_id,
  const int key_start_pos_id, const int num_datums, const int num_key_datums,
  const int local_window_size
) {
  const auto key_end_pos_id = key_start_pos_id + num_key_datums - 1;

  const bool local_window_en = (-1 != local_window_size);

  for (int i = 0; i < num_datums; i++) {
    const auto query_pos_id = query_start_pos_id + i;

    auto window_start_pos_id = 0;
    auto window_end_pos_id = query_pos_id;

    if (local_window_en && (query_pos_id >= local_window_size)) {
      window_start_pos_id = query_pos_id - local_window_size + 1;
    }

    // we have inclusive window [window_start_pos_id, window_end_pos_id]
    // now need to find overlap with [key_start_pos_id, key_end_pos_id]

    // handle no overlap case
    if ((window_end_pos_id < key_start_pos_id) ||
        (window_start_pos_id > key_end_pos_id)) {
      softmax_mask_info.at(i).start_index = 0;
      softmax_mask_info.at(i).end_index = 0;
    } else {
      window_start_pos_id = std::max(window_start_pos_id, key_start_pos_id);
      window_end_pos_id = std::min(window_end_pos_id, key_end_pos_id);

      auto window_size = window_end_pos_id - window_start_pos_id + 1;

      // region needed for compute - half open [start, end)
      softmax_mask_info.at(i).start_index =
        window_start_pos_id - key_start_pos_id;
      softmax_mask_info.at(i).end_index =
        softmax_mask_info.at(i).start_index + window_size;
    }
  }
}

// align to 64 byte/cache line size to avoid false sharing
struct alignas(64) softmax_row_state {
  float curr_max = -std::numeric_limits<float>::max();
  float curr_norm = 0.0f;
  float new_max = -std::numeric_limits<float>::max();
  float new_norm = 0.0f;
  bool applied_head_sink = false;
  std::uint8_t state = 0;
};

struct softmax_info {
  const uint16_t* base_input_ptr;
  uint16_t* base_output_ptr;
  const row_mask_info* softmax_mask_info;
  const float* head_sink_ptr;
  int num_heads;
  int M;
  int K;
  int context_chunk_size_q;
  int context_chunk_size_k;
  float attention_scale;

  softmax_row_state* base_softmax_state_p;
};

static void calc_row_softmax_state(void* data, size_t index) {
  const softmax_info* payload = (softmax_info*)data;

  const auto num_heads = payload->num_heads;
  const auto M = payload->M;

  const auto attention_scale = payload->attention_scale;

  // layout in memory is [num_heads, context_chunk_size_q, context_chunk_size_k]
  // within each head only need to do compute on [M, K] chunk, rest is padding
  const auto context_chunk_size_q = payload->context_chunk_size_q;
  const auto context_chunk_size_k = payload->context_chunk_size_k;

  auto m_index = index % M;
  auto head_index = index / M;

  const uint16_t* base_input_ptr = payload->base_input_ptr;
  auto base_output_ptr = payload->base_output_ptr;

  // mask gives [start, end) interval that softmax should be calculated on
  const row_mask_info& info = payload->softmax_mask_info[m_index];

  auto start_index = info.start_index;
  auto end_index = info.end_index;

  const auto K = end_index - start_index;

  const uint16_t* input_ptr =
    &base_input_ptr
      [head_index * context_chunk_size_q * context_chunk_size_k +
       m_index * context_chunk_size_k];
  uint16_t* output_ptr =
    &base_output_ptr
      [head_index * context_chunk_size_q * context_chunk_size_k +
       m_index * context_chunk_size_k];

  // state layout is [num_heads, context_chunk_size_q]
  auto state_idx = head_index * context_chunk_size_q + m_index;

  softmax_row_state* softmax_state_p =
    &payload->base_softmax_state_p[state_idx];
  auto state = softmax_state_p->state;  // 0 means curr is curr, new is new
                                        // 1 means curr is new, new is curr

  auto& curr_max =
    (0 == state) ? softmax_state_p->curr_max : softmax_state_p->new_max;
  auto& new_max =
    (0 == state) ? softmax_state_p->new_max : softmax_state_p->curr_max;
  auto& curr_norm =
    (0 == state) ? softmax_state_p->curr_norm : softmax_state_p->new_norm;
  auto& new_norm =
    (0 == state) ? softmax_state_p->new_norm : softmax_state_p->curr_norm;

  if (0 == K) {
    // no update in this iteration, propagate state through
    new_norm = curr_norm;
    new_max = curr_max;

    // zero out entire output row
    // [head_idx, m_index, 0:context_chunk_size_k - 1]
    memset(output_ptr, 0, context_chunk_size_k * sizeof(std::uint16_t));

    return;
  }

  // if local_window_size is enabled, zero out beginning and skip compute
  memset(output_ptr, 0, start_index * sizeof(std::uint16_t));

  input_ptr = &input_ptr[start_index];
  output_ptr = &output_ptr[start_index];

  std::vector<float> intermediate(K);

  constexpr size_t kBlockSize = 16;
  const auto num_blocks = K / kBlockSize;

  const auto offset = num_blocks * kBlockSize;
  // update max
  __m512 max_vals = _mm512_set1_ps(curr_max);
  __m512 attention_scales = _mm512_set1_ps(attention_scale);

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    const std::uint16_t* curr_src_ptr = &input_ptr[kBlockSize * block_idx];

    float* curr_dest_ptr = &intermediate[kBlockSize * block_idx];

    // read 16 bfloat16 - 8 in each 128-bit
    // [src_0, src_1 | src_2, src_3]
    __m256i src_packed16_i =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr));

    // cast from bloat16 to float32
    __m512 src = _mm512_cvtpbh_ps((__m256bh)src_packed16_i);

    // apply scale 1/sqrt(head_dim)
    // val = val*attention_scale
    // causal mask handled by loop bounds
    src = _mm512_mul_ps(src, attention_scales);

    // new_max = std::max(val, new_max);
    max_vals = _mm512_max_ps(max_vals, src);

    // intermediate[j] = val;
    _mm512_storeu_ps(curr_dest_ptr, src);
  }

  new_max = _mm512_reduce_max_ps(max_vals);

  // apply head sink
  const bool has_head_sink = nullptr != payload->head_sink_ptr;

  if (has_head_sink && !softmax_state_p->applied_head_sink) {
    const float head_sink_val = payload->head_sink_ptr[head_index];
    new_max = std::max(new_max, head_sink_val);
  }

  for (auto j = offset; j < K; j++) {
    uint32_t val_u = input_ptr[j] << 16;

    float val = *reinterpret_cast<float*>(&val_u);

    // apply scale 1/sqrt(head_dim)
    // causal mask is handled by loop bounds
    val = val * attention_scale;

    new_max = std::max(val, new_max);

    intermediate[j] = val;
  }

  // subtract by max and calculate exp
  // also calculate sum of exp for final norm
  // use avx-512 which can process 16 32-bit floats

  max_vals = _mm512_set1_ps(new_max);
  __m512 sum_exp_vec = _mm512_setzero_ps();

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    float* curr_src_ptr = &intermediate[kBlockSize * block_idx];

    __m512 vals = _mm512_loadu_ps(curr_src_ptr);
    vals = _mm512_sub_ps(vals, max_vals);
#if defined(_WIN32)
    vals = _mm512_exp_ps(vals);
#else
    throw std::runtime_error("Need to implement exp");
#endif
    sum_exp_vec = _mm512_add_ps(sum_exp_vec, vals);

    _mm512_storeu_ps(reinterpret_cast<__m512*>(curr_src_ptr), vals);
  }

  float sum_exp = _mm512_reduce_add_ps(sum_exp_vec);

  for (auto j = offset; j < K; j++) {
    float val = intermediate[j];

    val -= new_max;
    val = std::expf(val);

    sum_exp += val;

    intermediate[j] = val;
  }

  if (has_head_sink && !softmax_state_p->applied_head_sink) {
    softmax_state_p->applied_head_sink = true;
    const float head_sink_val = payload->head_sink_ptr[head_index];
    sum_exp += std::expf(-new_max + head_sink_val);
  }

  // update softmax norm with older running sum
  new_norm = std::expf(curr_max - new_max) * curr_norm + sum_exp;

  // calculate reciprocal of sum of exp
  __m128 x = _mm_set_ss(new_norm);
  x = _mm_rcp_ss(x);

  float recip_sum_exp = 0.0f;
  _mm_store_ss(&recip_sum_exp, x);

  // apply softmax norm
  __m512 norm_scales = _mm512_set1_ps(recip_sum_exp);

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    float* curr_src_ptr = &intermediate[kBlockSize * block_idx];
    std::uint16_t* curr_dest_ptr = &output_ptr[kBlockSize * block_idx];

    __m512 vals = _mm512_loadu_ps(curr_src_ptr);

    vals = _mm512_mul_ps(vals, norm_scales);

    // cast from float32 to bfloat16
    __m256bh val_bf16 = _mm512_cvtneps_pbh(vals);

    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(curr_dest_ptr), (__m256i)val_bf16
    );
  }

  for (auto j = offset; j < K; j++) {
    float val = intermediate[j];
    val *= recip_sum_exp;

    output_ptr[j] = float_to_bfloat16(val);
  }

  // zero out padded region in [num_heads, 0:M, end_index:context_chunk_size_k]
  memset(
    &output_ptr[K], 0,
    (context_chunk_size_k - end_index) * sizeof(std::uint16_t)
  );
}

static void calc_softmax_state(
  std::vector<xrt::bo>& inputs, std::vector<xrt::bo>& outputs,
  const uint16_t* base_mask_ptr, int num_heads, int M, int K,
  int context_chunk_size, float attention_scale,
  const std::vector<float>& curr_max,
  const std::vector<float>& curr_softmax_norm, std::vector<float>& new_max,
  std::vector<float>& new_softmax_norm
) {
  // layout in memory is [num_heads, context_chunk_size, context_chunk_size]
  // within each head only need to do compute on [M, K] chunk, rest is padding
  uint16_t* base_input_ptr = inputs.at(0).map<uint16_t*>();
  uint16_t* base_output_ptr = outputs.at(0).map<uint16_t*>();

  for (auto head_idx = 0; head_idx < num_heads; head_idx++) {
    for (auto i = 0; i < M; i++) {
      uint16_t* input_ptr =
        &base_input_ptr
          [head_idx * context_chunk_size * context_chunk_size +
           i * context_chunk_size];
      uint16_t* output_ptr =
        &base_output_ptr
          [head_idx * context_chunk_size * context_chunk_size +
           i * context_chunk_size];

      const uint16_t* mask_ptr = &base_mask_ptr[i * context_chunk_size];

      auto state_idx = head_idx * context_chunk_size + i;

      // update max
      new_max.at(state_idx) = curr_max.at(state_idx);
      for (auto j = 0; j < K; j++) {
        uint32_t val_u = input_ptr[j] << 16;
        uint32_t mask_val_u = mask_ptr[j] << 16;

        float val = *reinterpret_cast<float*>(&val_u);
        float mask_val = *reinterpret_cast<float*>(&mask_val_u);

        // apply scale 1/sqrt(head_dim) and causal mask
        val = val * attention_scale + mask_val;

        new_max.at(state_idx) = std::max(val, new_max.at(state_idx));

        output_ptr[j] = float_to_bfloat16(val);
      }

      float sum_exp = 0.0f;

      // subtract by max and calculate exp
      // also calculate sum of exp for final norm
      for (auto j = 0; j < K; j++) {
        uint32_t val_u = output_ptr[j] << 16;
        float val = *reinterpret_cast<float*>(&val_u);

        val -= new_max.at(state_idx);
        val = std::expf(val);

        sum_exp += val;

        output_ptr[j] = float_to_bfloat16(val);
      }

      // update softmax norm with older running sum
      new_softmax_norm.at(state_idx) =
        std::expf(curr_max.at(state_idx) - new_max.at(state_idx)) *
          curr_softmax_norm.at(state_idx) +
        sum_exp;

      // calculate reciprocal of sum of exp
      __m128 x = _mm_set_ss(new_softmax_norm.at(state_idx));
      x = _mm_rcp_ss(x);

      float recip_sum_exp = 0.0f;
      _mm_store_ss(&recip_sum_exp, x);

      // apply softmax norm
      for (auto j = 0; j < K; j++) {
        uint32_t val_u = output_ptr[j] << 16;
        float val = *reinterpret_cast<float*>(&val_u);
        val *= recip_sum_exp;

        output_ptr[j] = float_to_bfloat16(val);
      }

      // zero out padded region in [num_heads, 0:M, N:context_chunk_size]
      memset(
        &output_ptr[K], 0, (context_chunk_size - K) * sizeof(std::uint16_t)
      );
    }
  }
}

static void accumulate_attention_scores(
  std::vector<float>& cpu_attention_score_acc, uint16_t* partial_attention_ptr,
  std::vector<softmax_row_state>& softmax_state, const int M,
  const int num_heads, const int head_dim, const int context_chunk_size
) {
  // layout is [M, num_heads, head_dim]
  uint16_t* base_input_ptr = partial_attention_ptr;
  float* base_accum_ptr = cpu_attention_score_acc.data();

  // layout for state vectors is [num_heads, context_chunk_size]
  // update rule is exp(curr_max - new_max) * (curr_softmax_norm /
  // new_softmax_norm) * prev_acc + partial_attention_output

  for (auto i = 0; i < M; i++) {
    for (auto j = 0; j < num_heads; j++) {
      uint16_t* input_ptr =
        &base_input_ptr[i * num_heads * head_dim + j * head_dim];
      float* accum_ptr =
        &base_accum_ptr[i * num_heads * head_dim + j * head_dim];

      auto state_index = j * context_chunk_size + i;
      auto& row_state = softmax_state.at(state_index);

      auto& curr_max =
        (0 == row_state.state) ? row_state.curr_max : row_state.new_max;
      auto& new_max =
        (0 == row_state.state) ? row_state.new_max : row_state.curr_max;
      auto& curr_norm =
        (0 == row_state.state) ? row_state.curr_norm : row_state.new_norm;
      auto& new_norm =
        (0 == row_state.state) ? row_state.new_norm : row_state.curr_norm;

      auto scale = std::expf(curr_max - new_max) * curr_norm / new_norm;

      for (auto k = 0; k < head_dim; k++) {
        std::uint32_t val_u = input_ptr[k] << 16;
        float val = *reinterpret_cast<float*>(&val_u);
        accum_ptr[k] = scale * accum_ptr[k] + val;
      }

      // swap state
      row_state.state = 1 - row_state.state;
    }
  }
}

struct accumulate_info {
  float* cpu_attention_score_acc_ptr;
  const std::uint16_t* partial_attention_score_ptr;
  softmax_row_state* softmax_state_ptr;
  const int M;
  const int num_heads;
  const int head_dim;
  const int context_chunk_size;
};

void accumulate_row_attention_scores(void* data, size_t index) {
  const accumulate_info* payload = (const accumulate_info*)data;

  // layout is [M, num_heads, head_dim]
  const uint16_t* base_input_ptr = payload->partial_attention_score_ptr;
  float* base_accum_ptr = payload->cpu_attention_score_acc_ptr;
  // layout for state vectors is [num_heads, context_chunk_size]
  softmax_row_state* base_softmax_state_ptr = payload->softmax_state_ptr;

  auto num_heads = payload->num_heads;
  auto head_dim = payload->head_dim;
  auto context_chunk_size = payload->context_chunk_size;

  auto head_idx = index % num_heads;
  auto m_index = index / num_heads;

  const std::uint16_t* input_ptr =
    &base_input_ptr[m_index * num_heads * head_dim + head_idx * head_dim];
  float* accum_ptr =
    &base_accum_ptr[m_index * num_heads * head_dim + head_idx * head_dim];

  auto state_index = head_idx * context_chunk_size + m_index;
  softmax_row_state* row_state = &base_softmax_state_ptr[state_index];

  auto& curr_max =
    (0 == row_state->state) ? row_state->curr_max : row_state->new_max;
  auto& new_max =
    (0 == row_state->state) ? row_state->new_max : row_state->curr_max;
  auto& curr_norm =
    (0 == row_state->state) ? row_state->curr_norm : row_state->new_norm;
  auto& new_norm =
    (0 == row_state->state) ? row_state->new_norm : row_state->curr_norm;

  // update rule is:
  // exp(curr_max - new_max) * (curr_softmax_norm / new_softmax_norm) * prev_acc
  //  + partial_attention_output
  auto scale = std::expf(curr_max - new_max);

  if (curr_norm != new_norm) {
    scale *= curr_norm / new_norm;
  }

  constexpr auto kBlockSize = 16;
  const auto num_blocks = head_dim / kBlockSize;
  const auto offset = num_blocks * kBlockSize;

  __m512 scale_1 = _mm512_set1_ps(scale);

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    const std::uint16_t* curr_src_ptr = &input_ptr[kBlockSize * block_idx];
    float* curr_dst_ptr = &accum_ptr[kBlockSize * block_idx];

    // load 16 bfloat16
    __m256i src_packed16 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr));

    // cast from bloat16 to float32
    __m512 src = _mm512_cvtpbh_ps((__m256bh)src_packed16);

    __m512 dst = _mm512_loadu_ps(curr_dst_ptr);

    dst = _mm512_fmadd_ps(scale_1, dst, src);

    _mm512_storeu_ps(reinterpret_cast<__m512*>(curr_dst_ptr), dst);
  }

  for (auto k = offset; k < head_dim; k++) {
    std::uint32_t val_u = input_ptr[k] << 16;
    float val = *reinterpret_cast<float*>(&val_u);
    accum_ptr[k] = scale * accum_ptr[k] + val;
  }

  // swap state
  row_state->state = 1 - row_state->state;
}

static inline void update_bmm2_output(
  uint16_t* bmm2_out_ptr, std::vector<float>& cpu_attention_score_acc,
  const int M, const int num_heads, const int head_dim
) {
  float_buffer_to_bfloat16(
    cpu_attention_score_acc.data(), M * num_heads * head_dim, bmm2_out_ptr
  );
}

struct bmm1_info {
  const std::uint16_t* bmm1_in0;  // [N, 1, H]
  const std::uint16_t* bmm1_in1;  // [kv_num_heads, key_chunk_size, H]
  std::uint16_t* bmm1_out;        // [N, 1, key_chunk_size]
  const int num_heads;
  const int64_t kv_num_heads;
  const int64_t ratio;
  const int head_dim;
  const size_t key_start_pos_id;  // half open interval [start, end)
  const size_t key_end_pos_id;
  const int64_t key_chunk_size;
};

static void qk_bmm1(void* payload, size_t index) {
  const bmm1_info* bmm_payload = (const bmm1_info*)payload;

  const auto head_index = index;
  const auto ratio = bmm_payload->ratio;
  const auto kv_head_idx = index / ratio;
  const auto head_dim = bmm_payload->head_dim;
  const auto key_chunk_size = bmm_payload->key_chunk_size;

  const auto key_start_pos_id = bmm_payload->key_start_pos_id;
  const auto key_end_pos_id = bmm_payload->key_end_pos_id;
  const auto num_elems = key_end_pos_id - key_start_pos_id;

  if (num_elems == 0) {
    return;
  }

  if (head_dim % 32 != 0) {
    throw std::runtime_error("Expect head dim to be multiple of 32");
  }

  const std::uint16_t* bmm1_in0_base =
    &bmm_payload->bmm1_in0[head_index * head_dim];
  const std::uint16_t* bmm1_in1_base =
    &bmm_payload->bmm1_in1
       [kv_head_idx * key_chunk_size * head_dim + key_start_pos_id * head_dim];
  std::uint16_t* bmm1_out_base =
    &bmm_payload->bmm1_out[head_index * key_chunk_size + key_start_pos_id];

  constexpr size_t kBlockSize = 8;
  const auto num_blocks = num_elems / kBlockSize;

  constexpr size_t kAccBlockSize = 32;
  const auto num_acc_blocks = head_dim / kAccBlockSize;

  std::vector<float> fp32_results(kBlockSize);

  // outer loop on populating elements
  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    auto start_col_idx = kBlockSize * block_idx;

    auto col0_idx = start_col_idx;
    auto col1_idx = start_col_idx + 1;
    auto col2_idx = start_col_idx + 2;
    auto col3_idx = start_col_idx + 3;
    auto col4_idx = start_col_idx + 4;
    auto col5_idx = start_col_idx + 5;
    auto col6_idx = start_col_idx + 6;
    auto col7_idx = start_col_idx + 7;

    // accumulators for 8 columns
    __m512 accum_0 = _mm512_setzero_ps();
    __m512 accum_1 = _mm512_setzero_ps();
    __m512 accum_2 = _mm512_setzero_ps();
    __m512 accum_3 = _mm512_setzero_ps();
    __m512 accum_4 = _mm512_setzero_ps();
    __m512 accum_5 = _mm512_setzero_ps();
    __m512 accum_6 = _mm512_setzero_ps();
    __m512 accum_7 = _mm512_setzero_ps();

    for (auto acc_block_idx = 0; acc_block_idx < num_acc_blocks;
         acc_block_idx++) {
      // load Q
      __m512i q_i =
        _mm512_loadu_si512(&bmm1_in0_base[acc_block_idx * kAccBlockSize]);
      // compiler cast
      __m512bh q = (__m512bh)q_i;

      // load K for columns [0, 1, ... 7]
      __m512i k_0_i = _mm512_loadu_si512(
        &bmm1_in1_base[col0_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_1_i = _mm512_loadu_si512(
        &bmm1_in1_base[col1_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_2_i = _mm512_loadu_si512(
        &bmm1_in1_base[col2_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_3_i = _mm512_loadu_si512(
        &bmm1_in1_base[col3_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_4_i = _mm512_loadu_si512(
        &bmm1_in1_base[col4_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_5_i = _mm512_loadu_si512(
        &bmm1_in1_base[col5_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_6_i = _mm512_loadu_si512(
        &bmm1_in1_base[col6_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      __m512i k_7_i = _mm512_loadu_si512(
        &bmm1_in1_base[col7_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      // compiler cast
      __m512bh k_0 = (__m512bh)k_0_i;
      __m512bh k_1 = (__m512bh)k_1_i;
      __m512bh k_2 = (__m512bh)k_2_i;
      __m512bh k_3 = (__m512bh)k_3_i;
      __m512bh k_4 = (__m512bh)k_4_i;
      __m512bh k_5 = (__m512bh)k_5_i;
      __m512bh k_6 = (__m512bh)k_6_i;
      __m512bh k_7 = (__m512bh)k_7_i;

      // dot product between Q and K_i
      accum_0 = _mm512_dpbf16_ps(accum_0, q, k_0);
      accum_1 = _mm512_dpbf16_ps(accum_1, q, k_1);
      accum_2 = _mm512_dpbf16_ps(accum_2, q, k_2);
      accum_3 = _mm512_dpbf16_ps(accum_3, q, k_3);
      accum_4 = _mm512_dpbf16_ps(accum_4, q, k_4);
      accum_5 = _mm512_dpbf16_ps(accum_5, q, k_5);
      accum_6 = _mm512_dpbf16_ps(accum_6, q, k_6);
      accum_7 = _mm512_dpbf16_ps(accum_7, q, k_7);
    }

    fp32_results.at(0) = _mm512_reduce_add_ps(accum_0);
    fp32_results.at(1) = _mm512_reduce_add_ps(accum_1);
    fp32_results.at(2) = _mm512_reduce_add_ps(accum_2);
    fp32_results.at(3) = _mm512_reduce_add_ps(accum_3);
    fp32_results.at(4) = _mm512_reduce_add_ps(accum_4);
    fp32_results.at(5) = _mm512_reduce_add_ps(accum_5);
    fp32_results.at(6) = _mm512_reduce_add_ps(accum_6);
    fp32_results.at(7) = _mm512_reduce_add_ps(accum_7);

    __m256 dst = _mm256_loadu_ps(fp32_results.data());
    __m128bh dst_bf16 = _mm256_cvtneps_pbh(dst);
    _mm_storeu_si128(
      reinterpret_cast<__m128i*>(&bmm1_out_base[kBlockSize * block_idx]),
      (__m128i)dst_bf16
    );
  }

  const auto offset = num_blocks * kBlockSize;

  for (auto i = offset; i < num_elems; i++) {
    auto col0_idx = i;

    // accumulator for column
    __m512 accum_0 = _mm512_setzero_ps();

    for (auto acc_block_idx = 0; acc_block_idx < num_acc_blocks;
         acc_block_idx++) {
      // load Q
      __m512i q_i =
        _mm512_loadu_si512(&bmm1_in0_base[acc_block_idx * kAccBlockSize]);
      // compiler cast
      __m512bh q = (__m512bh)q_i;

      // load K
      __m512i k_0_i = _mm512_loadu_si512(
        &bmm1_in1_base[col0_idx * head_dim + acc_block_idx * kAccBlockSize]
      );
      // compiler cast
      __m512bh k_0 = (__m512bh)k_0_i;

      // dot product between Q and K_i
      accum_0 = _mm512_dpbf16_ps(accum_0, q, k_0);
    }

    float output = _mm512_reduce_add_ps(accum_0);

    bmm1_out_base[i] = float_to_bfloat16(output);
  }
}

struct bmm2_info {
  const std::uint16_t* bmm2_in0;  // [N, 1, key_chunk_size]
  const std::uint16_t* bmm2_in1;  // [kv_num_heads, key_chunk_size, H]
  std::uint16_t* bmm2_out;        // [N, 1, H]
  const int num_heads;
  const int64_t kv_num_heads;
  const int64_t ratio;
  const int head_dim;
  const size_t key_start_pos_id;  // half open interval [start, end)
  const size_t key_end_pos_id;
  const int64_t key_chunk_size;
};

static void av_bmm2(void* payload, size_t index) {
  const bmm2_info* bmm_payload = (const bmm2_info*)payload;

  const auto head_index = index;
  const auto ratio = bmm_payload->ratio;
  const auto kv_head_idx = index / ratio;
  const auto head_dim = bmm_payload->head_dim;
  const auto key_chunk_size = bmm_payload->key_chunk_size;

  const auto key_start_pos_id = bmm_payload->key_start_pos_id;
  const auto key_end_pos_id = bmm_payload->key_end_pos_id;
  const auto num_elems = key_end_pos_id - key_start_pos_id;

  const std::uint16_t* bmm2_in0_base =
    &bmm_payload->bmm2_in0[head_index * key_chunk_size + key_start_pos_id];
  const std::uint16_t* bmm2_in1_base =
    &bmm_payload->bmm2_in1
       [kv_head_idx * key_chunk_size * head_dim + key_start_pos_id * head_dim];
  std::uint16_t* bmm2_out_base =
    &bmm_payload->bmm2_out[head_index * 1 * head_dim];

  if (num_elems == 0) {
    memset(bmm2_out_base, 0, head_dim * sizeof(std::uint16_t));
    return;
  }

  std::vector<float> fp32_result(16, 0);

  if (head_dim % 16 != 0) {
    throw std::runtime_error("assume head dim is multiple of 16");
  }

  constexpr size_t kBlkSize = 16;
  const auto num_blocks = head_dim / kBlkSize;

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    constexpr size_t kRowBlkSize = 4;
    const auto num_row_blocks = num_elems / kRowBlkSize;

    __m512 accum_0 = _mm512_setzero_ps();
    __m512 accum_1 = _mm512_setzero_ps();
    __m512 accum_2 = _mm512_setzero_ps();
    __m512 accum_3 = _mm512_setzero_ps();

    for (auto row_block_idx = 0; row_block_idx < num_row_blocks;
         row_block_idx++) {
      // load 4 vals from single row of a
      float a_0_s =
        bfloat16_to_float_single(bmm2_in0_base[row_block_idx * kRowBlkSize]);
      float a_1_s = bfloat16_to_float_single(
        bmm2_in0_base[row_block_idx * kRowBlkSize + 1]
      );
      float a_2_s = bfloat16_to_float_single(
        bmm2_in0_base[row_block_idx * kRowBlkSize + 2]
      );
      float a_3_s = bfloat16_to_float_single(
        bmm2_in0_base[row_block_idx * kRowBlkSize + 3]
      );

      auto row0_idx = row_block_idx * kRowBlkSize;
      auto row1_idx = row_block_idx * kRowBlkSize + 1;
      auto row2_idx = row_block_idx * kRowBlkSize + 2;
      auto row3_idx = row_block_idx * kRowBlkSize + 3;

      // load 16 vals from 4 rows of V
      __m256i v_0_i = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
          &bmm2_in1_base[row0_idx * head_dim + block_idx * kBlkSize]
        )
      );
      __m256i v_1_i = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
          &bmm2_in1_base[row1_idx * head_dim + block_idx * kBlkSize]
        )
      );
      __m256i v_2_i = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
          &bmm2_in1_base[row2_idx * head_dim + block_idx * kBlkSize]
        )
      );
      __m256i v_3_i = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
          &bmm2_in1_base[row3_idx * head_dim + block_idx * kBlkSize]
        )
      );

      // broadcast scalr values
      __m512 a_0 = _mm512_set1_ps(a_0_s);
      __m512 a_1 = _mm512_set1_ps(a_1_s);
      __m512 a_2 = _mm512_set1_ps(a_2_s);
      __m512 a_3 = _mm512_set1_ps(a_3_s);

      // compiler cast
      __m256bh v_0_bf = (__m256bh)v_0_i;
      __m256bh v_1_bf = (__m256bh)v_1_i;
      __m256bh v_2_bf = (__m256bh)v_2_i;
      __m256bh v_3_bf = (__m256bh)v_3_i;

      // convert to float32
      __m512 v_0 = _mm512_cvtpbh_ps(v_0_bf);
      __m512 v_1 = _mm512_cvtpbh_ps(v_1_bf);
      __m512 v_2 = _mm512_cvtpbh_ps(v_2_bf);
      __m512 v_3 = _mm512_cvtpbh_ps(v_3_bf);

      // fused multiply-add
      accum_0 = _mm512_fmadd_ps(a_0, v_0, accum_0);
      accum_1 = _mm512_fmadd_ps(a_1, v_1, accum_1);
      accum_2 = _mm512_fmadd_ps(a_2, v_2, accum_2);
      accum_3 = _mm512_fmadd_ps(a_3, v_3, accum_3);
    }

    auto offset = num_row_blocks * kRowBlkSize;

    for (auto i = offset; i < num_elems; i++) {
      float a_0_s = bfloat16_to_float_single(bmm2_in0_base[i]);

      __m256i v_0_i = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
          &bmm2_in1_base[i * head_dim + block_idx * kBlkSize]
        )
      );

      __m512 a_0 = _mm512_set1_ps(a_0_s);

      __m256bh v_0_bf = (__m256bh)v_0_i;

      __m512 v_0 = _mm512_cvtpbh_ps(v_0_bf);

      accum_0 = _mm512_fmadd_ps(a_0, v_0, accum_0);
    }

    accum_0 = _mm512_add_ps(accum_0, accum_1);
    accum_2 = _mm512_add_ps(accum_2, accum_3);
    accum_0 = _mm512_add_ps(accum_0, accum_2);

    __m256bh out_bf16 = _mm512_cvtneps_pbh(accum_0);
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(&bmm2_out_base[block_idx * kBlkSize]),
      (__m256i)out_bf16
    );
  }
}

void AMDGQOKernel::token_mha_cpu(
  const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
  OrtTensor& value_states, const int rewind_pos, const int64_t total_seq_len,
  const int64_t local_window_size, const bool cast_kv_bfloat16
) {
  Ort::KernelContext ctx{context_p_};

  const int B = query_states.shape[0];    // Batch
  const int N = query_states.shape[1];    // Number of heads
  const int S_q = query_states.shape[2];  // Sequence length of query
  const int H = query_states.shape[3];    // Head_size
  const int S_k = key_states.shape[2];    // Sequence length of key

  // [N, 1, H] == [1, N, H]
  auto xCasted = static_cast<uint16_t*>(query_states.data);
  // [kv_num_heads, total_seq_len, head_dim]
  auto yCasted = static_cast<uint16_t*>(key_states.data);
  auto y2Casted = static_cast<uint16_t*>(value_states.data);

  const bool local_window_en = local_window_size != -1;

  float attention_scale = scale_;

  const std::int32_t seq_len = gqa_attrs.seq_len;

  if (1 != seq_len) {
    throw std::runtime_error("only supported for token phase");
  }

  auto key_start_pos_id = 0;
  auto key_end_pos_id = total_seq_len - 1;

  if ((0 < local_window_size) && (total_seq_len > local_window_size)) {
    key_start_pos_id = total_seq_len - local_window_size;
  }

  int key_chunk_size = std::min((int)total_seq_len, kTokenContextMaxSize);

  if (local_window_en) {
    key_chunk_size = std::min(key_chunk_size, (int)local_window_size);
  }

  constexpr auto kNumQueryToken = 1;

  const auto kv_chunk_size =
    kv_num_heads_ * key_chunk_size * H * sizeof(std::uint16_t);

  const auto bmm1_out_chunk_size =
    N * kNumQueryToken * key_chunk_size * sizeof(std::uint16_t);

  const auto bmm2_out_chunk_size =
    N * kNumQueryToken * H * sizeof(std::uint16_t);

  // xrt::bo bmm1_scratch_input = xrt::bo(
  //   ss_->bmm1_outputs_[0], bmm1_input_scratch_size_,
  //   bmm1_input_scratch_offset_
  //);

  //// layout is [bmm2_final_out | V_chunk]
  // xrt::bo bmm2_scratch_input = xrt::bo(
  //   ss_->bmm2_outputs_[0], bmm2_input_scratch_size_,
  //   bmm2_input_scratch_offset_
  //);

  //[k_chunk_bmm2_chunk_out | bmm1_chunk_out | v_chunk]
  auto mha_rmm_buffer = allocator_.AllocateBuffer(
    std::max(kv_chunk_size, bmm2_out_chunk_size) + bmm1_out_chunk_size +
    kv_chunk_size
  );

  uint8_t* buf = mha_rmm_buffer.Data<uint8_t>();

  uint16_t* k_bo_map =
    (std::uint16_t*)buf;  // bmm1_scratch_input.map<uint16_t*>(); K
  uint16_t* v_bo_map = (std::uint16_t*)&buf
    [std::max(kv_chunk_size, bmm2_out_chunk_size) +
     bmm1_out_chunk_size];  // bmm2_scratch_input.map<uint16_t*>(); V

  uint16_t* bmm1_out_map = (std::uint16_t*)&buf[std::max(
    kv_chunk_size, bmm2_out_chunk_size
  )];  // ss_->bmm1_outputs_[0].map<uint16_t*>(); bmm1 output
  uint16_t* bmm2_out_map =
    k_bo_map;  // in shared buffer this is assumed to be sized to hold both

  int remaining_key_len = key_end_pos_id - key_start_pos_id + 1;

  std::vector<row_mask_info> softmax_mask_info(kNumQueryToken);

  std::vector<softmax_row_state> softmax_state(N * kNumQueryToken);

  std::vector<float> cpu_attention_score_acc(kNumQueryToken * N * H, 0.0f);

  auto key_chunk_idx = 0;

  for (auto pos_id = key_start_pos_id; pos_id <= key_end_pos_id;
       pos_id += key_chunk_size) {
    const int num_key_datums = std::min(remaining_key_len, key_chunk_size);

    remaining_key_len -= num_key_datums;

    // copy over chunks for each kv head - K_j, V_j
    // memoy layout [kv_num_heads_, key_chunk_size, H]
    // valid data [kv_num_heads_, pos_id: pos_id + num_key_datums - 1, H]
    if (!cast_kv_bfloat16) {
      for (auto kv_head_idx = 0; kv_head_idx < kv_num_heads_; kv_head_idx++) {
        memcpy(
          &k_bo_map[kv_head_idx * key_chunk_size * H],
          &yCasted
            [kv_head_idx * S_k * H +
             (key_chunk_idx * key_chunk_size + key_start_pos_id) * H],
          num_key_datums * H * sizeof(uint16_t)
        );
        memcpy(
          &v_bo_map[kv_head_idx * key_chunk_size * H],
          &y2Casted
            [kv_head_idx * S_k * H +
             (key_chunk_idx * key_chunk_size + key_start_pos_id) * H],
          num_key_datums * H * sizeof(uint16_t)
        );
      }
    } else {
      // need to cast from float16 to bfloat16
      for (auto kv_head_idx = 0; kv_head_idx < kv_num_heads_; kv_head_idx++) {
        convert_buffer_float16_to_bfloat16(
          &k_bo_map[kv_head_idx * key_chunk_size * H],
          &yCasted
            [kv_head_idx * S_k * H +
             (key_chunk_idx * key_chunk_size + key_start_pos_id) * H],
          num_key_datums * H
        );
        convert_buffer_float16_to_bfloat16(
          &v_bo_map[kv_head_idx * key_chunk_size * H],
          &y2Casted
            [kv_head_idx * S_k * H +
             (key_chunk_idx * key_chunk_size + key_start_pos_id) * H],
          num_key_datums * H
        );
      }
    }

    // create mask for Q x K^T
    // rewind_pos is absolute starting position_id of query
    populate_mask(
      softmax_mask_info, rewind_pos, pos_id, kNumQueryToken, num_key_datums,
      local_window_size
    );

    // run Q x K^T
    // output memory layout [N, 1, key_chunk_size]

    bmm1_info payload_bmm1 = {
      xCasted,
      k_bo_map,
      bmm1_out_map,
      N,
      kv_num_heads_,
      N / kv_num_heads_,
      H,
      softmax_mask_info.at(0).start_index,
      softmax_mask_info.at(0).end_index,
      key_chunk_size
    };

    ctx.ParallelFor(qk_bmm1, static_cast<size_t>(N), 0, &payload_bmm1);

    // run masked softmax
    softmax_info payload_softmax = {
      bmm1_out_map,
      bmm1_out_map,
      softmax_mask_info.data(),
      has_head_sink_ ? head_sink_.data() : nullptr,
      N,
      kNumQueryToken,
      num_key_datums,
      kNumQueryToken,
      key_chunk_size,
      attention_scale,
      softmax_state.data()
    };

    ctx.ParallelFor(
      calc_row_softmax_state, static_cast<size_t>(N), 0, &payload_softmax
    );

    // run A x V
    // output is [N, 1, H]

    bmm2_info payload_bmm2 = {
      bmm1_out_map,
      v_bo_map,
      bmm2_out_map,
      N,
      kv_num_heads_,
      N / kv_num_heads_,
      H,
      softmax_mask_info.at(0).start_index,
      softmax_mask_info.at(0).end_index,
      key_chunk_size
    };

    ctx.ParallelFor(av_bmm2, static_cast<size_t>(N), 0, &payload_bmm2);

    // run scaled accumulation
    accumulate_info payload_accumulate = {
      cpu_attention_score_acc.data(),
      bmm2_out_map,
      softmax_state.data(),
      kNumQueryToken,
      N,
      H,
      kNumQueryToken
    };

    ctx.ParallelFor(
      accumulate_row_attention_scores, static_cast<size_t>(N), 0,
      &payload_accumulate
    );

    // update loop state
    key_chunk_idx++;
  }

  //// sync K/V chunk buffers back to device
  // bmm1_scratch_input.sync(
  //   XCL_BO_SYNC_BO_TO_DEVICE, std::max(kv_chunk_size, bmm2_out_chunk_size), 0
  //);
  // bmm2_scratch_input.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_chunk_size, 0);
  // ss_->bmm1_outputs_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, bmm1_out_chunk_size,
  // 0);

  // copy to bmm2 output that will feed o_proj matmul
  uint16_t* bmm2_out_ptr = ss_->bmm2_outputs_[0].map<uint16_t*>();
  update_bmm2_output(
    bmm2_out_ptr, cpu_attention_score_acc, kNumQueryToken, N, H
  );

  ss_->bmm2_outputs_[0].sync(
    XCL_BO_SYNC_BO_TO_DEVICE, kNumQueryToken * N * H * sizeof(std::uint16_t), 0
  );
}

void AMDGQOKernel::chunked_mha_aie(
  const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
  OrtTensor& value_states, const int rewind_pos, const int64_t total_seq_len,
  const int64_t local_window_size, const int32_t context_chunk_size,
  const bool use_aie_rope, const bool cast_kv_bfloat16
) {
  Ort::KernelContext ctx{context_p_};

  const int B = query_states.shape[0];    // Batch
  const int N = query_states.shape[1];    // Number of heads
  const int S_q = query_states.shape[2];  // Sequence length of query
  const int H = query_states.shape[3];    // Head_size
  const int S_k = key_states.shape[2];    // Sequence length of key

  auto xCasted = static_cast<uint16_t*>(query_states.data);
  auto yCasted = static_cast<uint16_t*>(key_states.data);
  auto y2Casted = static_cast<uint16_t*>(value_states.data);

  const bool local_window_en = local_window_size != -1;

  float attention_scale = scale_;

  const auto y_elements = B * kv_num_heads_ * S_k * H;

  std::vector<size_t> bmm1_shape{
    (size_t)N, (size_t)context_chunk_size, (size_t)H
  };
  std::vector<size_t> bmm1_trans_weight_shape{
    (size_t)kv_num_heads_, (size_t)H, (size_t)context_chunk_size
  };
  std::vector<size_t> softmax_shape{
    (size_t)N, (size_t)context_chunk_size, (size_t)context_chunk_size
  };
  std::vector<size_t> bmm2_shape{
    (size_t)N, (size_t)context_chunk_size, (size_t)context_chunk_size
  };
  std::vector<size_t> bmm2_weight_shape{
    (size_t)kv_num_heads_, (size_t)context_chunk_size, (size_t)H
  };

  ss_->bmm1_->set_execute_kernel_shape(bmm1_shape, bmm1_trans_weight_shape);
  ss_->bmm2_->set_execute_kernel_shape(bmm2_shape, bmm2_weight_shape);

  constexpr bool sync = true;

  std::vector<std::uint16_t> Q_tmp;

  if (use_aie_rope && (0 == rewind_pos)) {
    // Q and K are in bmm inputs already
    //  note: assumption is there is already sync on xrt::run
    //        currently in code K is run first, followed by Q with a sync
    //        Q will be in bmm1_inputs[0] and K in bmm1_inputs[1]

    // sync inputs back to cpu
    ss_->bmm1_inputs_[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    ss_->bmm1_inputs_[1].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // slice of final attention outputs will be written to bmm2_outputs[0]
    // which right now is the same buffer as bmm1_inputs_[0]
    // to avoid clobbering this make copy of Q

    Q_tmp.resize(N * S_q * H);
    xCasted = ss_->bmm1_inputs_[0].map<uint16_t*>();

    memcpy(Q_tmp.data(), xCasted, Q_tmp.size() * sizeof(std::uint16_t));

    xCasted = Q_tmp.data();
    yCasted = ss_->bmm1_inputs_[1].map<uint16_t*>();
  }

  {
    // need to copy from x/y casted to bmm1 inputs
    // layout for x/Query is [num_heads, seq_query, hidden_dim]
    // layout for y/Key is [num_kv_heads, total_seq_key, hidden_dim]
    // layout for y2/Value is [num_kv_heads, total_seq_val == total_seq_key,
    // hidden_dim] tile on dim 1

    // get sub bo, layout is [bmm_1_chunk_out | Q_chunk | K_chunk ==
    // bmm_2_chunk_out]
    xrt::bo bmm1_scratch_input0 = xrt::bo(
      ss_->bmm1_outputs_[0], bmm1_input0_scratch_size_,
      bmm1_input0_scratch_offset_
    );
    xrt::bo bmm1_scratch_input = xrt::bo(
      ss_->bmm1_outputs_[0], bmm1_input_scratch_size_,
      bmm1_input_scratch_offset_
    );

    // layout is [bmm2_final_out | V_chunk]
    xrt::bo bmm2_scratch_input = xrt::bo(
      ss_->bmm2_outputs_[0], bmm2_input_scratch_size_,
      bmm2_input_scratch_offset_
    );

    // reuse bmm2 scratch output as scratch input for bmm1 input 0 - Q
    // this is needed right now bmm1 input 0 just re-uses output of bmm2
    // however we will use bmm2 to store final accumulate result
    uint16_t* q_bo_map = bmm1_scratch_input0.map<uint16_t*>();  // Q
    uint16_t* k_bo_map = bmm1_scratch_input.map<uint16_t*>();   // K
    uint16_t* v_bo_map = bmm2_scratch_input.map<uint16_t*>();   // V

    std::vector<row_mask_info> softmax_mask_info(context_chunk_size);

    const auto q_chunk_size =
      N * context_chunk_size * H * sizeof(std::uint16_t);
    const auto kv_chunk_size =
      kv_num_heads_ * context_chunk_size * H * sizeof(std::uint16_t);
    const auto mask_size =
      context_chunk_size * context_chunk_size * sizeof(std::uint16_t);
    const auto bmm1_out_chunk_size =
      N * context_chunk_size * context_chunk_size * sizeof(std::uint16_t);

    const auto num_query_chunks =
      (gqa_attrs.seq_len + context_chunk_size - 1) / context_chunk_size;
    const auto num_key_chunks =
      (total_seq_len_ + context_chunk_size - 1) / context_chunk_size;

    std::int32_t remaining_seq = gqa_attrs.seq_len;

    auto bmm2_output_slice_offset = 0;

    // vertically chunk Q (sequence length)
    for (auto i = 0; i < num_query_chunks; i++) {
      // state for doing online softmax

      std::vector<softmax_row_state> softmax_state(N * context_chunk_size);

      // copy over chunks of query for each head - Q_i
      auto num_datums = std::min(context_chunk_size, remaining_seq);
      remaining_seq -= num_datums;

      std::vector<float> cpu_attention_score_acc(num_datums * N * H, 0.0f);

      const auto bmm2_out_chunk_size =
        num_datums * N * H * sizeof(std::uint16_t);

      for (auto head_idx = 0; head_idx < N; head_idx++) {
        memcpy(
          &q_bo_map[head_idx * context_chunk_size * H],
          &xCasted[head_idx * S_q * H + i * context_chunk_size * H],
          num_datums * H * sizeof(uint16_t)
        );
      }

      bmm1_scratch_input0.sync(XCL_BO_SYNC_BO_TO_DEVICE, q_chunk_size, 0);

      // vertically chunk K/V (total sequence length) and compute partial_j =
      // maskedsoftmax(Q_i x K_j^T) x V_j

      std::int32_t remaining_key_seq = total_seq_len_;

      for (auto j = 0; j < num_key_chunks; j++) {
        auto num_key_datums = std::min(context_chunk_size, remaining_key_seq);
        remaining_key_seq -= num_key_datums;

        // rewind_pos is absolute starting position_id of query
        // for key and value we assume they start from position_id 0
        // local_window_size can allow us to skip processing unnecessary chunked
        // tiles will be of size context_chunk_size x context_chunk_size

        int query_start_pos_id = rewind_pos + i * context_chunk_size;
        int query_end_pos_id = query_start_pos_id + num_datums - 1;
        int key_start_pos_id = j * context_chunk_size;
        int key_end_pos_id = key_start_pos_id + num_key_datums - 1;

        // outside of causal region
        if (query_end_pos_id < key_start_pos_id) {
          break;
        }

        // outside of banded region
        if (local_window_en &&
            (key_end_pos_id + local_window_size <= query_start_pos_id)) {
          continue;
        }

        // copy over chunks for each kv head - K_j, V_j
        if (!cast_kv_bfloat16) {
          for (auto kv_head_idx = 0; kv_head_idx < kv_num_heads_;
               kv_head_idx++) {
            memcpy(
              &k_bo_map[kv_head_idx * context_chunk_size * H],
              &yCasted[kv_head_idx * S_k * H + j * context_chunk_size * H],
              num_key_datums * H * sizeof(uint16_t)
            );
            memcpy(
              &v_bo_map[kv_head_idx * context_chunk_size * H],
              &y2Casted[kv_head_idx * S_k * H + j * context_chunk_size * H],
              num_key_datums * H * sizeof(uint16_t)
            );
          }
        } else {
          // need to cast from float16 to bfloat16
          for (auto kv_head_idx = 0; kv_head_idx < kv_num_heads_;
               kv_head_idx++) {
            convert_buffer_float16_to_bfloat16(
              &k_bo_map[kv_head_idx * context_chunk_size * H],
              &yCasted[kv_head_idx * S_k * H + j * context_chunk_size * H],
              num_key_datums * H
            );
            convert_buffer_float16_to_bfloat16(
              &v_bo_map[kv_head_idx * context_chunk_size * H],
              &y2Casted[kv_head_idx * S_k * H + j * context_chunk_size * H],
              num_key_datums * H
            );
          }
        }

        bmm1_scratch_input.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_chunk_size, 0);
        bmm2_scratch_input.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_chunk_size, 0);

        std::vector<xrt::bo> chunk_bmm1_inputs = {
          bmm1_scratch_input0, bmm1_scratch_input
        };

        // Q_i x K_j ^ T
        ss_->bmm1_->execute(chunk_bmm1_inputs, ss_->bmm1_outputs_, sync);

        ss_->bmm1_outputs_[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, bmm1_out_chunk_size, 0
        );

        // create mask for Q_i * K_j
        populate_mask(
          softmax_mask_info, query_start_pos_id, key_start_pos_id, num_datums,
          num_key_datums, local_window_size
        );
        // running softmax on cpu, so sync is not needed
        // softmax_mask_.sync(XCL_BO_SYNC_BO_TO_DEVICE, mask_size, 0);

        // maskedsoftmax(Q_i x K_j^T) - do on cpu since we need to update norm
        // and max serial implementation for reference
        // std::vector<xrt::bo> softmax_inputs = {bmm1_outputs_[0]};
        // std::vector<xrt::bo> softmax_outputs = {bmm2_inputs_[0]};
        // calc_softmax_state(softmax_inputs, softmax_outputs,
        // softmax_mask_bo_map,
        //                    N, num_datums, num_key_datums, context_chunk_size,
        //                    attention_scale, curr_max, curr_softmax_norm,
        //                    new_max, new_softmax_norm);

        softmax_info payload_softmax = {
          ss_->bmm1_outputs_[0].map<uint16_t*>(),
          ss_->bmm2_inputs_[0].map<uint16_t*>(),
          softmax_mask_info.data(),
          has_head_sink_ ? head_sink_.data() : nullptr,
          N,
          num_datums,
          num_key_datums,
          context_chunk_size,
          context_chunk_size,
          attention_scale,
          softmax_state.data()
        };

        ctx.ParallelFor(
          calc_row_softmax_state, static_cast<size_t>(N * num_datums), 0,
          &payload_softmax
        );

        // run bmm2 on NPU - need to sync back cpu softmax output
        ss_->bmm2_inputs_[0].sync(
          XCL_BO_SYNC_BO_TO_DEVICE, bmm1_out_chunk_size, 0
        );

        // calculate maskedsoftmax(Q_i x K_j^T) x V_j
        std::vector<xrt::bo> partial_bmm2_inputs = {
          ss_->bmm2_inputs_[0], bmm2_scratch_input
        };
        std::vector<xrt::bo> partial_bmm2_outputs = {bmm1_scratch_input};

        ss_->bmm2_->execute(partial_bmm2_inputs, partial_bmm2_outputs, sync);

        bmm1_scratch_input.sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, bmm2_out_chunk_size, 0
        );

        // need to accumulate partial attention score
        // note that there has been a transpose on output of bmm2
        // this also does final update of state for next iteration
        uint16_t* partial_attention_score = bmm1_scratch_input.map<uint16_t*>();
        // accumulate_attention_scores(
        //   cpu_attention_score_acc, partial_attention_score, softmax_state,
        //   num_datums, N, H, context_chunk_size
        //);

        accumulate_info payload_accumulate = {
          cpu_attention_score_acc.data(),
          partial_attention_score,
          softmax_state.data(),
          num_datums,
          N,
          H,
          context_chunk_size
        };

        ctx.ParallelFor(
          accumulate_row_attention_scores, static_cast<size_t>(num_datums * N),
          0, &payload_accumulate
        );
      }

      // copy to bmm2 output that will feed o_proj matmul
      uint16_t* bmm2_out_ptr = ss_->bmm2_outputs_[0].map<uint16_t*>();
      update_bmm2_output(
        &bmm2_out_ptr[bmm2_output_slice_offset], cpu_attention_score_acc,
        num_datums, N, H
      );

      bmm2_output_slice_offset += num_datums * N * H;
    }
  }

  ss_->bmm2_outputs_[0].sync(
    XCL_BO_SYNC_BO_TO_DEVICE, S_q * N * H * sizeof(std::uint16_t), 0
  );
}

void AMDGQOKernel::aie_execute(
  const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
  OrtTensor& value_states, OrtTensor& attention_mask, int rewind_pos,
  int64_t total_seq_len, int64_t local_window_size
) {
  // Code taken from:
  // https://gitenterprise.xilinx.com/VitisAI/transformers/blob/main/ops/torch_cpp/src/mha_npu_torch.cpp#L41

  // Get Shapes
  int B = query_states.shape[0];    // Batch
  int N = query_states.shape[1];    // Number of heads
  int S_q = query_states.shape[2];  // Sequence length of query
  int H = query_states.shape[3];    // Head_size
  int S_k = key_states.shape[2];    // Sequence length of key

  auto xCasted = static_cast<uint16_t*>(query_states.data);
  auto yCasted = static_cast<uint16_t*>(key_states.data);
  auto mCasted = static_cast<uint16_t*>(attention_mask.data);
  auto y2Casted = static_cast<uint16_t*>(value_states.data);

  if (!use_flash_mha_) {
    if (use_context_chunk_) {
      constexpr bool cast_kv_bfloat16 = false;
      chunked_mha_aie(
        gqa_attrs, query_states, key_states, value_states, rewind_pos,
        total_seq_len, local_window_size, context_chunk_size_, use_aie_rope_,
        cast_kv_bfloat16
      );

    } else {
      std::vector<size_t> bmm1_shape{(size_t)N, (size_t)S_q, (size_t)H};
      std::vector<size_t> bmm1_trans_weight_shape{
        (size_t)kv_num_heads_, (size_t)H, (size_t)S_q
      };
      std::vector<size_t> softmax_shape{(size_t)N, (size_t)S_q, (size_t)S_q};
      std::vector<size_t> bmm2_shape{(size_t)N, (size_t)S_q, (size_t)S_q};
      std::vector<size_t> bmm2_weight_shape{
        (size_t)kv_num_heads_, (size_t)S_q, (size_t)H
      };

      ss_->bmm1_->set_execute_kernel_shape(bmm1_shape, bmm1_trans_weight_shape);
      ss_->bmm2_->set_execute_kernel_shape(bmm2_shape, bmm2_weight_shape);
      ss_->softmax_->set_params("softmax", softmax_shape);

      // on Linux, cannot execute in async mode: xrt::run objects go out of
      // scope in DD eager execute. Windows makes a copy unlike Linux
#ifdef _WIN32
      // AIE4 is not able to run async mode
      const bool wait = mladfVersion() == MladfVersion::aie4_v1;
#else
      const bool wait = true;
#endif  // _WIN32

      const bool bmm1_cpu_en = false;
      const bool softmax_cpu_en = false;
      const bool bmm2_cpu_en = false;

      const bool bmm1_wait = softmax_cpu_en || wait;
      const bool softmax_wait = bmm2_cpu_en || wait;
      const bool bmm2_wait = continueOnException() || wait;

      PROFILING_START(execute_bmm1)
      execute_bmm1(
        bmm1_cpu_en, ss_->bmm1_.get(), *this, ss_->bmm1_inputs_,
        ss_->bmm1_outputs_, xCasted, yCasted, B, N, S_q, H, kv_num_heads_, S_k,
        bmm1_wait, continueOnException(), name(), use_aie_rope_, rewind_pos
      );
      PROFILING_END(execute_bmm1, false, name().c_str())

      PROFILING_START(execute_softmax)
      bool update_mask = (ss_->seq_len_ != S_q) ||
                         (ss_->rewind_pos_ != rewind_pos) ||
                         (ss_->curr_local_window_size_ != local_window_size);
      ;
      execute_softmax(
        softmax_cpu_en, ss_->softmax_.get(), *this, ss_->bmm1_outputs_,
        ss_->softmax_mask_, ss_->bmm2_inputs_, mCasted, S_q, S_k, scale_, N,
        softmax_wait, continueOnException(), name(), update_mask,
        local_window_size
      );
      PROFILING_END(execute_softmax, false, name().c_str())

      // next time check again, to skip uselss mask bo update
      if (update_mask) {
        ss_->seq_len_ = S_q;
        ss_->rewind_pos_ = rewind_pos;
        ss_->curr_local_window_size_ = local_window_size;
      }

      PROFILING_START(execute_bmm2)
      execute_bmm2(
        bmm2_cpu_en, ss_->bmm2_.get(), *this, ss_->bmm2_inputs_,
        ss_->bmm2_outputs_, y2Casted, B, N, S_q, H, kv_num_heads_, S_k,
        bmm2_wait, continueOnException(), name()
      );
      PROFILING_END(execute_bmm2, false, name().c_str())
    }

  } else {
    auto seq_mha = mha_aie_kernel_info_.tryPadSeq(S_q, maxSeqLength());
    // TODO: prefill chunk second seq_mha -> total_seq_len
    std::vector<size_t> kernel_shape = {
      (size_t)num_heads_, (size_t)kv_num_heads_, (size_t)gqa_attrs.seq_len,
      (size_t)total_seq_len, (size_t)head_size_
    };
    std::vector<size_t> a_shape = {
      static_cast<size_t>(num_heads_), static_cast<size_t>(seq_mha),
      static_cast<size_t>(head_size_)
    };
    size_t len = rewind_pos ? total_seq_len : seq_mha;
    std::vector<size_t> b_shape = {
      static_cast<size_t>(kv_num_heads_), static_cast<size_t>(len),
      static_cast<size_t>(head_size_)
    };

    NPUTensor input_tensor_0 = {nullptr, a_shape, "bfloat16"};
    NPUTensor input_tensor_1 = {nullptr, b_shape, "bfloat16"};
    std::vector<NPUTensor> inputs = {input_tensor_0, input_tensor_1};
    std::vector<NPUTensor> outputs;
    flash_mha_->set_tensor_shape(inputs, outputs);
    bool en_cpy = qkv_out_size_.at(0) != ss_->bmm1_inputs_[0].size();
    execute_mha(
      flash_mha_.get(), ss_->flash_in_, kernel_shape, gqa_attrs.seq_len,
      total_seq_len, npu_kernel_size_, ss_->bmm2_outputs_, xCasted, yCasted,
      y2Casted, qkv_out_size_, true, continueOnException(), name(),
      use_aie_rope_, en_cpy, rewind_pos, sink_bo_
    );
  }
}

struct KVCacheData {
  /// past/present k
  const uint16_t* past_k_bf16 = {};
  const float* present_k_fp32 = {};
  uint16_t* present_k_bf16 = {};
  /// past/present v
  const uint16_t* past_v_bf16 = {};
  const float* present_v_fp32 = {};
  uint16_t* present_v_bf16 = {};
  /// length
  int past_SxH = 0;
  int present_SxH = 0;
  int head_size = 0;
};

/// @brief save present_k/v[B, N, S, H] to shared buffer[B, N, S_buffer, H].
void AMDGQOKernel::save_present_kv_to_shared_buffer(
  uint16_t* dst_k, uint16_t* dst_v, uint16_t* past_k, uint16_t* past_v,
  int rewind_pos, const uint16_t* src_k, const uint16_t* src_v,
  const int num_heads,       // N_kv==2
  const int num_group,       // 1
  const int buffer_seq_len,  // 4096
  const int seq_len,         //  31
  const int head_size        // 128
) {
  /*                |------ buffer_seq_len==4096 --------|
     past kv cache: |---old_k_0--|-- ................ ---|
                    |---old_k_1--|-- ................ ---|
                    |-rewind_pos-|
     dst  kv cache: |---old_k_0--|--new_k_0--|--- ... ---|
                    |---old_k_1--|--new_k_1--|--- ... ---|
     current kv:    |--new_k_0--|--new_k_1--|
                    |--seq_len--|
     dst  kv cache: |--new_k_0--|----- .............. ---|
                    |--new_k_1--|----- .............. ---|   */

  int copy_size = seq_len * head_size;
  int copy_size_past = rewind_pos * head_size;
  for (int n = 0; n < num_heads; n++) {
    int offset_dst_withrewind = (n * buffer_seq_len + rewind_pos) * head_size;
    int offset_src_k = num_group * n * seq_len * head_size;
    int offset_src_v = n * seq_len * head_size;
    if (rewind_pos) {
      int offset_dst = n * buffer_seq_len * head_size;
      std::memcpy(
        dst_k + offset_dst, past_k + offset_dst,
        copy_size_past * sizeof(uint16_t)
      );
      std::memcpy(
        dst_v + offset_dst, past_v + offset_dst,
        copy_size_past * sizeof(uint16_t)
      );
      //   std::cout <<"
      //   N/h/of_d_w/of_d/of_s/cp_size/iRewind/buffer_seq_len/seql "
      //      <<num_heads<<" "<<head_size<<" "<<offset_dst_withrewind<<" "<<
      //      offset_dst <<" "<<offset_src<<" "<<copy_size<<" "<<rewind_pos<<"
      //      "<<buffer_seq_len<<" "<<seq_len<<"\n";
      // 32 96 5664=59*96 0 0 960 59 4096 10 //  32 96 398880 393216 960 960 59
      // 4096 10
    }
    std::memcpy(
      dst_k + offset_dst_withrewind, src_k + offset_src_k,
      copy_size * sizeof(uint16_t)
    );
    std::memcpy(
      dst_v + offset_dst_withrewind, src_v + offset_src_v,
      copy_size * sizeof(uint16_t)
    );
  }
}

struct SplitQKVData {
  /// qkv
  const uint16_t* packed_qkv = {};
  uint16_t* q = {};
  uint16_t* k = {};
  uint16_t* v = {};
  /// params
  const int N_q = 0;
  const int N_kv = 0;
  const int S = 0;
  const int H = 0;
};

inline void Split_QKV(void* raw_data, size_t n) {
  auto data = reinterpret_cast<SplitQKVData*>(raw_data);
  const int& N_q = data->N_q;
  const int& N_kv = data->N_kv;
  const int& S = data->S;
  const int& H = data->H;
  auto q_src_offset = n * (N_q + 2 * N_kv) * H;
  auto k_src_offset = q_src_offset + N_q * H;
  auto v_src_offset = k_src_offset + N_kv * H;
  auto q_dst_offset = n * N_q * H;
  auto k_dst_offset = n * N_kv * H;
  auto v_dst_offset = n * N_kv * H;
  std::memcpy(
    data->q + q_dst_offset, data->packed_qkv + q_src_offset,
    N_q * H * sizeof(uint16_t)
  );
  std::memcpy(
    data->k + k_dst_offset, data->packed_qkv + k_src_offset,
    N_kv * H * sizeof(uint16_t)
  );
  std::memcpy(
    data->v + v_dst_offset, data->packed_qkv + v_src_offset,
    N_kv * H * sizeof(uint16_t)
  );

  // since this code is parallelized, no nead to performance track memcpys
}

inline void Split_QKV_transV(void* raw_data, size_t n) {
  auto data = reinterpret_cast<SplitQKVData*>(raw_data);
  const int& N_q = data->N_q;
  const int& N_kv = data->N_kv;
  const int& S = data->S;
  const int& H = data->H;
  auto q_src_offset = n * (N_q + 2 * N_kv) * H;
  auto k_src_offset = q_src_offset + N_q * H;
  auto v_src_offset = k_src_offset + N_kv * H;
  auto q_dst_offset = n * N_q * H;
  auto k_dst_offset = n * N_kv * H;

  std::memcpy(
    data->q + q_dst_offset, data->packed_qkv + q_src_offset,
    N_q * H * sizeof(uint16_t)
  );
  std::memcpy(
    data->k + k_dst_offset, data->packed_qkv + k_src_offset,
    N_kv * H * sizeof(uint16_t)
  );

  const int& D1 = data->S;
  const int& D2 = data->N_kv;
  const int& D3 = data->H;
  const int& i1 = n;
  for (int i2 = 0; i2 < D2; i2++) {
    std::memcpy(
      (void*)(data->v + i2 * D1 * D3 + i1 * D3),
      (void*)((data->packed_qkv + v_src_offset) + i2 * D3),
      D3 * sizeof(uint16_t)
    );
  }

  // since this code is parallelized, no need to performance track memcpys
}

inline void Split_QKV_transAll(void* raw_data, size_t n) {
  auto data = reinterpret_cast<SplitQKVData*>(raw_data);
  const int& N_q = data->N_q;
  const int& N_kv = data->N_kv;
  const int& S = data->S;
  const int& H = data->H;
  auto q_src_offset = n * (N_q + 2 * N_kv) * H;
  auto k_src_offset = q_src_offset + N_q * H;
  auto v_src_offset = k_src_offset + N_kv * H;

  // packed qkv has columns are concatenated [Q | K | V]
  //  i.e. data packed in each row of S as [N_q * H | N_kv * H | N_kv * H]

  // transpose Q from [S, N_q, H] to [N_q, S, H]
  {
    const int& D1 = data->S;
    const int& D2 = data->N_q;
    const int& D3 = data->H;
    const int& i1 = n;
    for (int i2 = 0; i2 < D2; i2++) {
      std::memcpy(
        (void*)(data->q + i2 * D1 * D3 + i1 * D3),
        (void*)((data->packed_qkv + q_src_offset) + i2 * D3),
        D3 * sizeof(uint16_t)
      );
    }
  }

  // transpose KV from [S, N_kv, H] to [N_kv, S, H]
  {
    const int& D1 = data->S;
    const int& D2 = data->N_kv;
    const int& D3 = data->H;
    const int& i1 = n;
    for (int i2 = 0; i2 < D2; i2++) {
      std::memcpy(
        (void*)(data->k + i2 * D1 * D3 + i1 * D3),
        (void*)((data->packed_qkv + k_src_offset) + i2 * D3),
        D3 * sizeof(uint16_t)
      );
      std::memcpy(
        (void*)(data->v + i2 * D1 * D3 + i1 * D3),
        (void*)((data->packed_qkv + v_src_offset) + i2 * D3),
        D3 * sizeof(uint16_t)
      );
    }
  }

  // since this code is parallelized, no need to performance track memcpys
}

void AMDGQOKernel::set_kv_cache(void* k_cache, void* v_cache) {
  shared_k_cache_ = k_cache;
  shared_v_cache_ = v_cache;
}

int AMDGQOKernel::getContextChunkSize(
  int seq_len, int total_seq_len, int local_window_size
) {
  // TODO: query these from DD operator directly
  const std::vector<int> bmm_buckets = {256, 512, 1024, 2048};
  // cost is from running latency experiments for different shapes
  // treating 256 as base shape
  const std::vector<float> cost = {1.0f, 1.3f, 4.9f, 14.7f};

  float min_cost = -1.0f;
  int min_index = 0;
  int curr_index = 0;

  for (const auto& bucket_size : bmm_buckets) {
    auto num_chunks_q = (seq_len + bucket_size - 1) / bucket_size;

    auto key_seq_len = total_seq_len;

    if ((local_window_size > 0) && (seq_len < total_seq_len)) {
      key_seq_len = std::min(key_seq_len, seq_len + local_window_size - 1);
    }

    auto num_chunks_k = (key_seq_len + bucket_size - 1) / bucket_size;

    auto curr_cost = num_chunks_q * num_chunks_k * cost.at(curr_index);

    if ((-1.0f == min_cost) || (curr_cost < min_cost)) {
      min_cost = curr_cost;
      min_index = curr_index;
    }

    curr_index++;
  }

  // outside of latency we might want to clamp due to memory considerations
  return std::min(bmm_buckets.at(min_index), kContextMaxSize);
}

void AMDGQOKernel::UpdateSharedBuffer(size_t kernel_size) {
  std::vector<size_t> a_shape = {
    static_cast<size_t>(num_heads_), static_cast<size_t>(kernel_size),
    static_cast<size_t>(kernel_size)
  };

  std::vector<size_t> b_shape = {
    static_cast<size_t>(kv_num_heads_), static_cast<size_t>(kernel_size),
    static_cast<size_t>(head_size_)
  };
  NPUTensor input_tensor_0 = {nullptr, a_shape, "bfloat16"};
  NPUTensor input_tensor_1 = {nullptr, b_shape, "bfloat16"};
  std::vector<NPUTensor> inputs = {input_tensor_0, input_tensor_1};
  std::vector<NPUTensor> outputs;
  std::map<std::string, std::any> attrs;
  size_t bmm2_output = 0;
  size_t bmm1_out_size = 0;
  size_t bmm2_input_1 = 0;
  size_t bmm1_in0 = 0;
  size_t bmm1_in1 = 0;
  size_t gemm_kernel_size = npu_kernel_size_q_;
  if (!use_flash_mha_) {
    std::vector<OpArgMap> arg_map =
      ss_->bmm2_->get_buffer_reqs(inputs, outputs, attrs);
    auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());
    bmm2_output = size_map["out"];
    bmm1_out_size = size_map["in0"];
    bmm2_input_1 = size_map["in1"];

    if (use_context_chunk_) {
      std::vector<size_t> a_chunk_shape{
        (size_t)num_heads_, (size_t)kContextMaxSize, (size_t)head_size_
      };
      std::vector<size_t> b_chunk_shape{
        (size_t)kv_num_heads_, (size_t)head_size_, (size_t)kContextMaxSize
      };

      NPUTensor chunk_input_tensor_0 = {nullptr, a_chunk_shape, "bfloat16"};
      NPUTensor chunk_input_tensor_1 = {nullptr, b_chunk_shape, "bfloat16"};
      std::vector<NPUTensor> chunk_inputs = {
        chunk_input_tensor_0, chunk_input_tensor_1
      };

      std::vector<OpArgMap> chunk_arg_map =
        ss_->bmm1_->get_buffer_reqs(chunk_inputs, outputs, attrs);
      auto chunk_size_map = get_NPU_tensor_size(chunk_arg_map, mladfVersion());

      // bmm1 output is largest tensor by far, will have dims [num_heads,
      // seq_len, seq_len] reduce size to whats only needed for chunked
      // computation within GQA
      bmm1_out_size = chunk_size_map["out"];

      // have "scratch" buffer at end of bmm1 output - used to hold slices of Q
      // and K
      bmm1_input0_scratch_offset_ = bmm1_out_size;
      bmm1_input0_scratch_size_ = chunk_size_map["in0"];
      bmm1_out_size += bmm1_input0_scratch_size_;

      // re-use K for scratch output of bmm2
      bmm1_input_scratch_offset_ = bmm1_out_size;
      bmm1_input_scratch_size_ =
        std::max(chunk_size_map["in0"], chunk_size_map["in1"]);
      bmm1_out_size += bmm1_input_scratch_size_;

      // have "scratch" buffer at end of bmm2 output - used for slices of V
      bmm2_input_scratch_offset_ = bmm2_output;
      bmm2_input_scratch_size_ = chunk_size_map["in1"];
      bmm2_output += bmm2_input_scratch_size_;
    }
  } else {
    gemm_kernel_size = q_seq_len_;
    std::vector<size_t> a_shape_mha = {
      static_cast<size_t>(num_heads_), static_cast<size_t>(q_seq_len_),
      static_cast<size_t>(head_size_)
    };
    size_t b_seq = total_seq_len_ > q_seq_len_ ? total_seq_len_ : q_seq_len_;
    std::vector<size_t> b_shape_mha = {
      static_cast<size_t>(kv_num_heads_), static_cast<size_t>(b_seq),
      static_cast<size_t>(head_size_)
    };
    NPUTensor input_tensor_0_mha = {nullptr, a_shape_mha, "bfloat16"};
    NPUTensor input_tensor_1_mha = {nullptr, b_shape_mha, "bfloat16"};
    std::vector<NPUTensor> inputs_mha = {
      input_tensor_0_mha, input_tensor_1_mha
    };
    std::vector<NPUTensor> outputs_mha;
    // std::vector<size_t> kernel_shape = {
    //   (size_t)num_heads_, (size_t)kv_num_heads_, 128,
    //   (size_t)kernel_size, (size_t)head_size_
    // };
    auto attr = getCommonAttrs();
    if (local_window_size_ > 0) {
      attr["window_size"] = local_window_size_;
    }
    flash_mha_->set_tensor_shape(inputs_mha, outputs_mha, attr);
    std::vector<OpArgMap> arg_map =
      flash_mha_->get_buffer_reqs(inputs_mha, outputs_mha, attrs);
    auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());
    bmm2_output = size_map["out"];
    bmm1_in0 = size_map["in0"];
    bmm1_in1 = size_map["in1"];
    bmm2_input_1 = size_map["in2"];
    qkv_out_size_ = {bmm1_in0, bmm1_in1, bmm2_input_1, bmm2_output};
  }

  std::vector<size_t> a_shape_gemm = {
    1, static_cast<size_t>(gemm_kernel_size),
    static_cast<size_t>(matmulnbits_attrs_.k)
  };
  std::vector<size_t> b_shape_gemm = {
    static_cast<size_t>(matmulnbits_attrs_.k),
    static_cast<size_t>(matmulnbits_attrs_.n)
  };
  std::vector<size_t> c_shape_gemm = {
    1, static_cast<size_t>(gemm_kernel_size),
    static_cast<size_t>(matmulnbits_attrs_.n)
  };

  NPUTensor input_tensor = {nullptr, a_shape_gemm, "bfloat16"};
  NPUTensor wts_tensor = {nullptr, b_shape_gemm, "bfloat16"};
  NPUTensor out_tensor = {nullptr, c_shape_gemm, "bfloat16"};
  NPUTensor placeholder;
  std::vector<NPUTensor> inputs_gemm = {input_tensor, wts_tensor,  placeholder,
                                        placeholder,  placeholder, out_tensor};
  // to keep tensor index same with fusion for get_buffer_reqs()
  if (Lora::isEnabled()) {
    inputs_gemm.insert(inputs_gemm.begin() + 1, placeholder);
  }

  std::vector<NPUTensor> outputs_gemm;
  std::map<std::string, std::any> attrs_gemm;
  std::vector<int> grp_size = {static_cast<int>(matmulnbits_attrs_.block_size)};
  attrs_gemm["group_size"] = grp_size;
  attrs_gemm["mem_opt"] = 1;
  std::vector<OpArgMap> arg_map_gemm =
    ss_->gemm_->get_buffer_reqs(inputs_gemm, outputs_gemm, attrs_gemm);
  auto size_map_gemm = get_NPU_tensor_size(arg_map_gemm, mladfVersion());
  // the order of shared buffers matters
  SharedBuffer::Requirements shared_buffer_reqs;

  if (!use_flash_mha_) {
    shared_buffer_reqs.emplace_back("bmm1_out", alignTo4096(bmm1_out_size));
    shared_buffer_reqs.emplace_back("bmm2_in1", alignTo4096(bmm2_input_1));
    shared_buffer_reqs.emplace_back("bmm1_in1", alignTo4096(bmm2_input_1));
  } else {
    shared_buffer_reqs.emplace_back("bmm1_in0", alignTo4096(bmm1_in0));
    shared_buffer_reqs.emplace_back("bmm1_in1", alignTo4096(bmm2_input_1));
    shared_buffer_reqs.emplace_back("bmm2_in1", alignTo4096(bmm2_input_1));
  }
  const auto bmm2_size = std::max(bmm2_output, size_map_gemm["in0"]);
  shared_buffer_reqs.emplace_back("bmm2_out", alignTo4096(bmm2_size));
  // TODO this scratch buffer can be reused with bmm1 bos
  if (size_map_gemm.find("scratch") != size_map_gemm.end()) {
    // TODO(gaoyue): scratch shares same space with bmm but current design make
    // it not able to share inside one group_name and the *2 factor is needed
    shared_buffer_reqs.emplace_back(
      "scratch", alignTo4096(size_map_gemm["scratch"] * 2)
    );
    has_scratch_buffer_ = true;
  } else {
    has_scratch_buffer_ = false;
  }

  shared_buffer_reqs.emplace_back(
    "gemm_out", alignTo4096(size_map_gemm["out"])
  );

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}

void AMDGQOKernel::updateKvCache(
  OrtKernelContext* context, Ort::BFloat16_t* k_rope_data,
  Ort::BFloat16_t* v_data_ptr, uint16_t* present_k_data,
  uint16_t* present_v_data, uint16_t* past_k_data, uint16_t* past_v_data,
  uint64_t total_seq_len, const int head_size, int rewind_pos, const int N_kv,
  const int seq_len, ONNXTensorElementDataType present_k_data_type,
  bool with_custom_allocator
) {
  size_t K_size = N_kv * seq_len * head_size;

  const std::vector<int64_t> K_shape_2 = {1, N_kv, seq_len, head_size};

  // Convert K from BFloat16 to Float16
  Ort::BFloat16_t* curr_k_data_bf16 = k_rope_data;
  Ort::BFloat16_t* curr_v_data_bf16 = v_data_ptr;

#ifdef DUMP_KV_CACHE
  if (name() == kDumpGqoName) {
    saveKvCache("Curr K", "BF16", curr_k_data_bf16, K_size);
  }
#endif  // DUMP_KV_CACHE

  uint16_t* src_k_data;
  uint16_t* src_v_data;
  RyzenMM::BufferRef cast_buffer_object;

  if (present_k_data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    // Convert V from BFloat16 to Float16
    size_t V_size = N_kv * seq_len * head_size;
    auto cast_buffer_size =
      K_size * sizeof(Ort::Float16_t) + V_size * sizeof(Ort::Float16_t);
    cast_buffer_object = allocator_.AllocateBuffer(cast_buffer_size);
    uint16_t* cast_buffer = cast_buffer_object.Data<uint16_t>();

    Ort::Float16_t* curr_k_data_fp16 =
      reinterpret_cast<Ort::Float16_t*>(cast_buffer);

    auto k_sync = std::async(std::launch::async, [=]() {
      ort_cast_bf16_to_fp16_.execute(
        curr_k_data_fp16, curr_k_data_bf16, K_shape_2, context
      );
    });

#ifdef DUMP_KV_CACHE
    if (name() == kDumpGqoName) {
      saveKvCache("Curr K", "FP16", curr_k_data_fp16, K_size);
    }
#endif  // DUMP_KV_CACHE

    const std::array<int64_t, 4> V_shape = {1, N_kv, seq_len, head_size};
    const std::vector<int64_t> V_shape_2 = {1, N_kv, seq_len, head_size};

#ifdef DUMP_KV_CACHE
    if (name() == kDumpGqoName) {
      saveKvCache("Curr V", "BF16", curr_v_data_bf16, V_size);
    }
#endif  // DUMP_KV_CACHE

    Ort::Float16_t* curr_v_data_fp16 =
      reinterpret_cast<Ort::Float16_t*>(&cast_buffer[K_size]);
    // std::memset(curr_v_data_fp16, 0, V_size *
    // sizeof(Ort::Float16_t));
    ort_cast_bf16_to_fp16_.execute(
      curr_v_data_fp16, curr_v_data_bf16, V_shape_2, context
    );
    k_sync.wait();
#ifdef DUMP_KV_CACHE
    std::cout << "Dumping KV cache for GQO: " << name() << std::endl;
    if (name() == kDumpGqoName) {
      saveKvCache("Curr V", "FP16", curr_v_data_fp16, V_size);
    }
#endif  // DUMP_KV_CACHE

    src_k_data = (uint16_t*)curr_k_data_fp16;
    src_v_data = (uint16_t*)curr_v_data_fp16;
  } else {
    src_k_data = (uint16_t*)curr_k_data_bf16;
    src_v_data = (uint16_t*)curr_v_data_bf16;
  }

  // Enable shared kv cache only if custom alloc is unavailable
  // As of now, custom alloc query is not available in gqo, hence
  // reusing the same macro to disable shared kv cache
  bool use_shared_cache = !with_custom_allocator;
#ifndef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  use_shared_cache = false;
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  auto k_cache_src_p =
    use_shared_cache ? (uint16_t*)shared_k_cache_ : present_k_data;
  auto v_cache_src_p =
    use_shared_cache ? (uint16_t*)shared_v_cache_ : present_v_data;
  save_present_kv_to_shared_buffer(
    k_cache_src_p, v_cache_src_p, past_k_data, past_v_data, rewind_pos,
    src_k_data, src_v_data, N_kv, 1, total_seq_len, seq_len, head_size
  );
#ifdef DUMP_KV_CACHE
  if (name() == kDumpGqoName) {
    saveKvCache(
      "Present K", "FP16", reinterpret_cast<Ort::Float16_t*>(present_k_data),
      N_kv * total_seq_len * head_size
    );
    saveKvCache(
      "Present V", "FP16", reinterpret_cast<Ort::Float16_t*>(present_v_data),
      N_kv * total_seq_len * head_size
    );
  }
#endif  // DUMP_KV_CACHE

  cast_buffer_object = {};
}

void AMDGQOKernel::Compute(
  OrtKernelContext* context, bool with_custom_allocator
) {
#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif

  PROFILING_START(Compute)
  PROFILING_START(setup)
  if (ss_->bmm1_inputs_.empty()) {
    initializeKernels();
  }
  shared_weights_.setWeightAddr(0);
  wait_for_data_ = false;

  if (useExternalData()) {
    readData();
    wait_for_data_ = true;
  }

  context_p_ = context;

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  // Prepare input/output tensors
  // Extracting the input and output information
  auto packed_qkv = ctx.GetInput(0);
  auto key = ctx.GetInput(1);
  auto value = ctx.GetInput(2);
  bool is_packed_qkv = key == nullptr;
  auto past_k = ctx.GetInput(kPastKeyIdx);
  auto past_v = ctx.GetInput(4);
  auto seqlens_k = ctx.GetInput(5);
  auto total_seqlen = ctx.GetInput(6);
  auto cos_cache = ctx.GetInput(kCosIdx);
  auto sin_cache = ctx.GetInput(kSinIdx);
  const uint16_t* mm_inp = nullptr;
  RyzenMM::BufferRef output_ptr1;
  // Query data / shape
  auto qkv_shape = packed_qkv.GetTensorTypeAndShapeInfo().GetShape();
  auto past_k_shape = past_k.GetTensorTypeAndShapeInfo().GetShape();

  GQAattrs gqa_attrs;
  auto batch_size = qkv_shape[0];
  gqa_attrs.batch_size = batch_size;
  auto seq_len = qkv_shape[1];
  const bool is_prefill = seq_len != 1;
  is_prefill_ = is_prefill;

  // TODO aie rope just work for model just has single rope cache
  if (use_aie_rope_ &&
      (!ss_->cos_sin_cache_ ||
       ((ss_->rope_name_ != rope_cache_name_) && is_prefill))) {
    std::tie(ss_->cos_sin_cache_, ss_->cos_shape_) = getRopeCache(
      &allocator_, cos_cache, sin_cache, rotary_interleaved_, mladfVersion()
    );
  } else if (ss_->cos_shape_.empty()) {
    ss_->cos_shape_ = cos_cache.GetTensorTypeAndShapeInfo().GetShape();
  }

  // todo gqa_attrs.seq_len
  gqa_attrs.seq_len = seq_len;
  auto head_size = past_k_shape[3];
  gqa_attrs.head_size = head_size;

  gqa_attrs.q_shape = {batch_size, seq_len, num_heads_ * head_size};
  gqa_attrs.q_num = batch_size * seq_len * num_heads_ * head_size;
  gqa_attrs.k_shape = {batch_size, seq_len, kv_num_heads_ * head_size};
  gqa_attrs.k_num = batch_size * seq_len * kv_num_heads_ * head_size;
  gqa_attrs.v_shape = gqa_attrs.k_shape;
  gqa_attrs.v_num = gqa_attrs.k_num;

  const int32_t* seq_len_k = seqlens_k.GetTensorData<int32_t>();
  int total_seq_len = seq_len_k[0] + 1;
  const auto& flash_mha_granularity = setFlashMhaGranularity();
  npu_kernel_size_ = getNPUKernelGranularity(
    total_seq_len, use_flash_mha_ ? flash_mha_granularity
                                  : std::vector<size_t>{1024, 2048, 3072, 4096}
  );

  const auto is_supported =
    mha_aie_kernel_info_.isSeqSupported(gqa_attrs.seq_len);
  if (mladfVersion() == MladfVersion::aie4_v1 && !is_supported) {
    // TODO: The output is incorrect when both RoPE and FlashMHA are enabled,
    // and TXN padding is required on AIE4.
    use_aie_rope_ = 0;
  }

  npu_kernel_size_q_ = getNPUKernelGranularity(seq_len);
  context_chunk_size_ =
    getContextChunkSize(seq_len, total_seq_len, local_window_size_);

  use_context_chunk_ = run_context_chunk(total_seq_len, seq_len);

  q_seq_len_ = npu_kernel_size_;
  total_seq_len_ = total_seq_len;

  bool use_aie_gqa = is_prefill && use_aie_gqo_;

  const bool use_aie_chunk_gqa = use_aie_gqo_ && use_context_chunk_;
  const bool use_aie_single_gqa =
    use_aie_gqa && (seq_len <= maxSeqLength()) && !use_aie_chunk_gqa;

  // regular flow is upsize both query size and total seq to same size
  // in chunked flow do not want memory to grow with total seq len - only hold
  // current K/V
  UpdateSharedBuffer(
    (use_aie_chunk_gqa || !is_prefill) ? npu_kernel_size_q_ : npu_kernel_size_
  );
  PROFILING_END(setup, false, name().c_str())

#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point input_format_start = Clock::now();
#endif

  PROFILING_START(input_format)
  Ort::MemoryInfo memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::BFloat16_t> npu_qkv_input;
  const auto* qkv_data = inputCast(npu_qkv_input, packed_qkv, context);

  Ort::BFloat16_t* q_data_ptr = nullptr;
  uint16_t* k_data_ptr = nullptr;

  bool create_fp32_cache =
    (ss_->rope_name_ != rope_cache_name_) || ss_->cos_cache_fp32_.empty();
  // AIE RoPE is disabled, AIE MHA is disabled, context chunking flow
  bool use_cpu_rope = (!use_aie_rope_) || (!use_aie_gqo_) || use_context_chunk_;
  // use CPU during token phase
  use_cpu_rope = use_cpu_rope || (!is_prefill);

  if (use_cpu_rope && create_fp32_cache) {
    size_t cos_cache_elm = std::accumulate(
      ss_->cos_shape_.begin(), ss_->cos_shape_.end(), 1ULL, std::multiplies<>()
    );

    const auto cache_dtype =
      cos_cache.GetTensorTypeAndShapeInfo().GetElementType();

    // expect bfloat16/float16/float32
    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == cache_dtype) {
      // resize to non-empty
      ss_->sin_cache_fp32_.resize(1);
      ss_->cos_cache_fp32_.resize(1);

      // no need to do any casting
      ss_->sin_cache_fp32_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(sin_cache.GetTensorData<float>()),
        cos_cache_elm, ss_->cos_shape_.data(), ss_->cos_shape_.size()
      );

      ss_->cos_cache_fp32_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(cos_cache.GetTensorData<float>()),
        cos_cache_elm, ss_->cos_shape_.data(), ss_->cos_shape_.size()
      );

    } else {
      ss_->sin_cache_fp32_.resize(cos_cache_elm);
      ss_->cos_cache_fp32_.resize(cos_cache_elm);

      ss_->sin_cache_fp32_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, ss_->sin_cache_fp32_.data(), ss_->sin_cache_fp32_.size(),
        ss_->cos_shape_.data(), ss_->cos_shape_.size()
      );

      ss_->cos_cache_fp32_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, ss_->cos_cache_fp32_.data(), ss_->cos_cache_fp32_.size(),
        ss_->cos_shape_.data(), ss_->cos_shape_.size()
      );

      if (ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 == cache_dtype) {
        RecordDuration(Metric::Casting, [&]() {
          ort_cast_bf16_to_fp32_.execute(
            ss_->sin_cache_fp32_.data(),
            const_cast<Ort::BFloat16_t*>(
              sin_cache.GetTensorData<Ort::BFloat16_t>()
            ),
            ss_->cos_shape_, context
          );
          ort_cast_bf16_to_fp32_.execute(
            ss_->cos_cache_fp32_.data(),
            const_cast<Ort::BFloat16_t*>(
              cos_cache.GetTensorData<Ort::BFloat16_t>()
            ),
            ss_->cos_shape_, context
          );
        });
      } else if (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 == cache_dtype) {
        RecordDuration(Metric::Casting, [&]() {
          ort_cast_fp16_to_fp32_.execute(
            ss_->sin_cache_fp32_.data(),
            const_cast<Ort::Float16_t*>(
              sin_cache.GetTensorData<Ort::Float16_t>()
            ),
            ss_->cos_shape_, context
          );
          ort_cast_fp16_to_fp32_.execute(
            ss_->cos_cache_fp32_.data(),
            const_cast<Ort::Float16_t*>(
              cos_cache.GetTensorData<Ort::Float16_t>()
            ),
            ss_->cos_shape_, context
          );
        });
      }
    }
  }
  ss_->rope_name_ = rope_cache_name_;
  Ort::ConstValue sin_cache_fp32 = ss_->sin_cache_fp32_tensor_.GetConst();
  Ort::ConstValue cos_cache_fp32 = ss_->cos_cache_fp32_tensor_.GetConst();

  bool aie_supp_seq_flag = mha_aie_kernel_info_.isSeqSupported(seq_len);

  RyzenMM::BufferRef v_data_buffer;
  Ort::BFloat16_t* v_data_ptr = nullptr;
  // BF16
  if (use_aie_single_gqa) {
    if (!use_flash_mha_) {
      rebind_bmm_params();
    } else {
      rebind_flashmha_params();
    }
  }
  if (aie_supp_seq_flag || use_aie_chunk_gqa) {
    v_data_ptr = (Ort::BFloat16_t*)shared_buffer_.Get("bmm2_in1").ptr;
  } else {
    v_data_buffer =
      allocator_.AllocateBuffer(gqa_attrs.v_num * sizeof(Ort::BFloat16_t));
    v_data_ptr = (Ort::BFloat16_t*)v_data_buffer.Data();
  }

  if (is_packed_qkv) {
    q_data_ptr = (Ort::BFloat16_t*)shared_buffer_.Get("bmm2_out").ptr;
    k_data_ptr = (uint16_t*)shared_buffer_.Get("bmm1_in1").ptr;

    SplitQKVData split_qkv_data = {
      (const uint16_t*)qkv_data,
      (uint16_t*)q_data_ptr,
      k_data_ptr,
      (uint16_t*)v_data_ptr,
      static_cast<int>(num_heads_),
      static_cast<int>(kv_num_heads_),
      static_cast<int>(seq_len),
      static_cast<int>(head_size)
    };
    RecordDuration(Metric::SplitQKV, [&]() {
      if (use_aie_single_gqa) {
        ctx.ParallelFor(
          Split_QKV_transV, static_cast<size_t>(seq_len), 0, &split_qkv_data
        );
      } else if (use_aie_chunk_gqa) {
        ctx.ParallelFor(
          Split_QKV_transAll, static_cast<size_t>(seq_len), 0, &split_qkv_data
        );
      } else {  // For Running ORT gqo Op
        ctx.ParallelFor(
          Split_QKV, static_cast<size_t>(seq_len), 0, &split_qkv_data
        );
      }
    });
  } else {
    auto k_data_type = key.GetTensorTypeAndShapeInfo().GetElementType();
    auto v_data_type = value.GetTensorTypeAndShapeInfo().GetElementType();
    auto v_data =
      const_cast<Ort::BFloat16_t*>(value.GetTensorData<Ort::BFloat16_t>());

    auto q_data = const_cast<Ort::BFloat16_t*>(qkv_data);

    q_data_ptr = q_data;
    k_data_ptr = const_cast<uint16_t*>(key.GetTensorData<uint16_t>());

    if (use_aie_single_gqa) {
      RecordDuration(Metric::Transpose, [&]() {
        ort_transpose_.execute(
          v_data_ptr, v_data, {batch_size, seq_len, kv_num_heads_, head_size},
          context
        );
      });
    } else if (use_aie_chunk_gqa) {
      auto k_data =
        const_cast<Ort::BFloat16_t*>(key.GetTensorData<Ort::BFloat16_t>());

      q_data_ptr = (Ort::BFloat16_t*)shared_buffer_.Get("bmm2_out").ptr;
      k_data_ptr = (uint16_t*)shared_buffer_.Get("bmm1_in1").ptr;

      RecordDuration(Metric::Transpose, [&]() {
        ort_transpose_.execute(
          q_data_ptr, q_data, {batch_size, seq_len, num_heads_, head_size},
          context
        );
        ort_transpose_.execute(
          (Ort::BFloat16_t*)k_data_ptr, k_data,
          {batch_size, seq_len, kv_num_heads_, head_size}, context
        );
        ort_transpose_.execute(
          v_data_ptr, v_data, {batch_size, seq_len, kv_num_heads_, head_size},
          context
        );
      });

    } else {  // For Running ORT gqo Op
      v_data_buffer = {};
      v_data_ptr =
        const_cast<Ort::BFloat16_t*>(value.GetTensorData<Ort::BFloat16_t>());
    }
  }

  // Past Key data / shape
  gqa_attrs.past_k_shape = past_k.GetTensorTypeAndShapeInfo().GetShape();
  gqa_attrs.past_k_num = past_k.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_k_type = past_k.GetTensorTypeAndShapeInfo().GetElementType();
  uint16_t* past_k_data =
    const_cast<uint16_t*>(past_k.GetTensorData<uint16_t>());

  // Past Value data / shape
  gqa_attrs.past_v_shape = past_v.GetTensorTypeAndShapeInfo().GetShape();
  gqa_attrs.past_v_num = past_v.GetTensorTypeAndShapeInfo().GetElementCount();
  auto past_v_type = past_v.GetTensorTypeAndShapeInfo().GetElementType();
  uint16_t* past_v_data =
    const_cast<uint16_t*>(past_v.GetTensorData<uint16_t>());

  // Allocate output (primary output shape is same as query shape
  // [batch_size, sequence_length, hidden_size])

  std::vector<int64_t> output_shape(
    {batch_size, seq_len, head_size * num_heads_}
  );

  //  calculate shape for present k/v
  int past_sequence_length = static_cast<int>(past_k_shape[2]);
  int total_sequence_length =
    total_seq_len > past_sequence_length ? total_seq_len : past_sequence_length;
  bool past_present_share_buffer =
    past_sequence_length == total_sequence_length;
  gqa_attrs.present_k_shape = {
    batch_size, kv_num_heads_, static_cast<int64_t>(total_sequence_length),
    head_size
  };
  gqa_attrs.present_v_shape = gqa_attrs.present_k_shape;

  auto present_k = ctx.GetOutput(0, gqa_attrs.present_k_shape);
  gqa_attrs.present_k_num =
    present_k.GetTensorTypeAndShapeInfo().GetElementCount();
  gqa_attrs.present_k_data_type =
    present_k.GetTensorTypeAndShapeInfo().GetElementType();
  uint16_t* present_k_data = present_k.GetTensorMutableData<uint16_t>();

  auto present_v = ctx.GetOutput(1, gqa_attrs.present_v_shape);
  gqa_attrs.present_v_num =
    present_v.GetTensorTypeAndShapeInfo().GetElementCount();
  auto present_v_data_type =
    present_v.GetTensorTypeAndShapeInfo().GetElementType();
  uint16_t* present_v_data = present_v.GetTensorMutableData<uint16_t>();
  PROFILING_END(input_format, false, name().c_str())

  // Note(ltp): Using aie kernel when:
  // - prefill phase
  // - USE_AIE_GQO = 1
  // - seq_len <= aie max seq_len length
#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point gqo_compute_start = Clock::now();
#endif
  if (seq_len >= ss_->cos_shape_[0]) {
    throw std::invalid_argument(
      "Input is larger than model rope cache shape[0]"
    );
  }
  std::future<void> kv_cache_update;
  if (use_aie_chunk_gqa) {
    runAieChunkGqa(
      gqa_attrs, total_seq_len, q_data_ptr, k_data_ptr, v_data_ptr, past_k_data,
      past_v_data, present_k_data, present_v_data, cos_cache_fp32,
      sin_cache_fp32, past_sequence_length, seq_len_k,
      past_present_share_buffer, with_custom_allocator, context
    );
  } else if (use_aie_single_gqa) {
    kv_cache_update = runAieGqa(
      gqa_attrs, total_seq_len, q_data_ptr, k_data_ptr, v_data_ptr, past_k_data,
      past_v_data, present_k_data, present_v_data, cos_cache_fp32,
      sin_cache_fp32, past_sequence_length, seq_len_k,
      past_present_share_buffer, with_custom_allocator, context
    );
  } else {
    output_ptr1 = runCpuGqa(
      gqa_attrs, q_data_ptr, k_data_ptr, v_data_ptr, past_k_data, past_v_data,
      present_k_data, present_v_data, cos_cache_fp32, sin_cache_fp32,
      past_sequence_length, seq_len_k[0], total_seq_len,
      past_present_share_buffer, context
    );

    mm_inp = static_cast<const uint16_t*>(output_ptr1.Data());
  }

  v_data_ptr = nullptr;
#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point gqo_compute_end = Clock::now();
#endif

  auto output_cast = executeMatMulNBits(
    context, qkv_shape, is_prefill, seq_len, mm_inp, use_aie_chunk_gqa
  );
  PROFILING_START(output_format2)

  output_ptr1 = {};

  if (kv_cache_update.valid()) {
    RecordDuration(Metric::KVCacheUpdate, [&]() { kv_cache_update.wait(); });
  }

  // releasing this buffers after `kv_cache_update` because it might have a
  // reference to the buffer
  v_data_buffer = {};
  PROFILING_END(output_format2, false, name().c_str())

#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point output_format_end = Clock::now();

  const Duration setup_duration = input_format_start - compute_start;
  const Duration input_format_duration = gqo_compute_start - input_format_start;
  const Duration gqo_execute_duration = gqo_compute_end - gqo_compute_start;
  const Duration oproj_execute_duration = o_proj_end - gqo_compute_end;
  const Duration output_format_duration = output_format_end - o_proj_end;

  measurements_.at(EventID::SETUP_ID) += setup_duration;
  measurements_.at(EventID::INPUT_FORMAT_ID) += input_format_duration;
  measurements_.at(EventID::OUTPUT_FORMAT_ID) += output_format_duration;
  measurements_.at(EventID::GQO_EXECUTE_ID) += gqo_execute_duration;
  measurements_.at(EventID::OPROJ_EXECUTE_ID) += oproj_execute_duration;
#endif
  if (output_cast) std::vector<Ort::BFloat16_t>().swap(npu_output_);

  if (useExternalData()) {
    unloadData();
  }

  if (flash_in_data_ptr_) {
#ifdef _WIN32
    _aligned_free(flash_in_data_ptr_);
#else
    free(flash_in_data_ptr_);
#endif
    flash_in_data_ptr_ = nullptr;
  }
  if (flash_out_data_ptr_) {
#ifdef _WIN32
    _aligned_free(flash_out_data_ptr_);
#else
    free(flash_out_data_ptr_);
#endif
    flash_out_data_ptr_ = nullptr;
  }
  if (freeAfterPrefill(name())) {
    ss_->rope_inbos_.clear();
    ss_->rope_outbos_.clear();
    ss_->bmm1_inputs_.clear();
    ss_->bmm1_outputs_.clear();
    ss_->bmm2_inputs_.clear();
    ss_->bmm2_outputs_.clear();
    ss_->rope_.reset();
    ss_->bmm1_.reset();
    ss_->bmm2_.reset();
    ss_->softmax_.reset();
    if (lastNode(name()) && getPrefillBufferRelease(npu_kernel_size_)) {
      shared_buffer_.Reset();
    }
  }
  PROFILING_END(Compute, false, name().c_str())
}

bool AMDGQOKernel::executeMatMulNBits(
  OrtKernelContext* context, const std::vector<int64_t>& qkv_shape,
  bool is_prefill, int64_t seq_len, const uint16_t* mm_inp,
  bool use_aie_chunk_gqa
) {
  PROFILING_START(execute_matmul)
  Ort::KernelContext ctx{context};

  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < (qkv_shape.size() - 1); i++)
    out_shape.push_back(qkv_shape[i]);

  out_shape.push_back(matmulnbits_attrs_.n);

  auto output_tensor = ctx.GetOutput(2, {out_shape.begin(), out_shape.end()});
  auto out = output_tensor.GetTensorMutableData<uint16_t>();
  auto output_shape1 = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto out_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  uint16_t* out_ptr;
  bool output_cast =
    !output_cast_indices_.empty() && output_cast_indices_.size() == 3;

  if (output_cast) {
    npu_output_.resize(out_cnt);
    out_ptr = (uint16_t*)npu_output_.data();
  } else
    out_ptr = (uint16_t*)out;

  std::pair<size_t, size_t> weights_shape = {
    matmulnbits_attrs_.k, matmulnbits_attrs_.n
  };

  bool is_gqa_aie_output =
    (is_prefill && use_aie_gqo_ && seq_len <= maxSeqLength()) ||
    use_aie_chunk_gqa;

  if (is_gqa_aie_output) {
    executeMatMulNBitsAie(
      ss_->bmm2_outputs_, out_ptr, output_shape1, weights_shape,
      static_cast<int>(matmulnbits_attrs_.block_size), cnt_
    );
  } else {
    executeMatMulNBitsAie(
      mm_inp, out_ptr, output_shape1, weights_shape,
      static_cast<int>(matmulnbits_attrs_.block_size), cnt_
    );
  }
  PROFILING_END(execute_matmul, false, name().c_str())

#ifdef NPU_GQO_PROFILE_EN
  const Clock::time_point o_proj_end = Clock::now();
#endif
  PROFILING_START(output_format1)
  if (useExternalData()) {
    loadData();
  }
  if (output_cast) {
    RecordDuration(Metric::Casting, [&]() {
      ort_cast_bf16_to_fp16_.execute(
        (Ort::Float16_t*)out, npu_output_.data(), output_shape1, context
      );
    });
  }
  PROFILING_END(output_format1, false, name().c_str())

  return output_cast;
}

bool AMDGQOKernel::setUseAieRope(
  const Ort::ConstKernelInfo& info, int64_t& rotary_embedding_dim,
  const std::unordered_map<std::string, std::string>& session_configs
) {
  int is_cos_const = 0;
  const_cos_ = info.GetTensorConstantInput(kCosIdx, &is_cos_const);
  int is_sin_const = 0;
  const_sin_ = info.GetTensorConstantInput(kSinIdx, &is_sin_const);
  /// check if cos/sin cache is const tensor
  auto is_const_cache = true;
  if (is_cos_const == 0 || is_sin_const == 0) {
    is_const_cache = false;
  }
  /// if cos/sin cache is const tensor, then set rotary_embedding_dim to
  /// cos_shape[1] * 2, we need to pass this param to ort RoPE kernel.
  /// if not, set it to default value 0.f
  if (is_const_cache) {
    auto cos_shape = const_cos_.GetTensorTypeAndShapeInfo().GetShape();
    auto inferred_rotary_embedding_dim = cos_shape[1] * 2;
    if (rotary_embedding_dim != 0 &&
        rotary_embedding_dim != inferred_rotary_embedding_dim) {
      throw std::invalid_argument(
        "Inconsistent value for rotary_embedding_dim"
      );
    }
    if (rotary_embedding_dim == 0) {
      rotary_embedding_dim = inferred_rotary_embedding_dim;
    }
  }

  // AIE rope requires head_size == rotary_embedding_dim
  // if (head_size_ != rotary_embedding_dim) {
  //   use_aie_rope_ = false;
  // }

  const auto& value = session_configs.at("hybrid_dbg_use_aie_rope");
  if (!value.empty()) {
    use_aie_rope_ = value == "1";
  }

  return is_const_cache;
}

void AMDGQOKernel::setUseAieGqa(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& value = session_configs.at("hybrid_dbg_use_aie_gqa");
  if (!value.empty()) {
    use_aie_gqo_ = value == "1";
  }
}

void AMDGQOKernel::setEnableContextChunk(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& value = session_configs.at("hybrid_opt_chunk_context");
  if (!value.empty()) {
    context_chunk_en_ = value == "1";
  }
}

void AMDGQOKernel::setContextChunkSize(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& value = session_configs.at("hybrid_opt_chunk_context_threshold");
  if (!value.empty()) {
    context_size_threshold_ = std::stoi(value);
  }
}

bool AMDGQOKernel::run_context_chunk(size_t total_seq_len, size_t seq_len) {
  return (total_seq_len > context_size_threshold_) &&
         (seq_len > context_size_threshold_) && context_chunk_en_ &&
         (!use_flash_mha_);
}

void AMDGQOKernel::setUseFlashMHA(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& value = session_configs.at("hybrid_dbg_use_flash_mha");
  if (!value.empty()) {
    use_flash_mha_ = value == "1";
  }
}

const Ort::BFloat16_t* AMDGQOKernel::inputCast(
  std::vector<Ort::BFloat16_t>& npu_qkv_input,
  const Ort::ConstValue& packed_qkv, OrtKernelContext* context
) {
  if (bool input_cast = !input_cast_indices_.empty(); input_cast) {
    auto qkv_size = packed_qkv.GetTensorTypeAndShapeInfo().GetElementCount();
    npu_qkv_input.resize(qkv_size);
    RecordDuration(Metric::Casting, [&]() {
      ort_cast_fp16_to_bf16_.execute(npu_qkv_input.data(), packed_qkv, context);
    });
    return npu_qkv_input.data();
  }
  return packed_qkv.GetTensorData<Ort::BFloat16_t>();
}

struct kv_cast_in_payload {
  int64_t src_kv_sequence_length;
  int64_t dst_kv_sequence_length;
  int64_t curr_kv_seq_len;
  int64_t head_size;
  ONNXTensorElementDataType kv_src_dtype;
  uint16_t* kv_cache_src;
  float* kv_cache_dst;
};

static void kv_cast_row(void* data, size_t index) {
  const kv_cast_in_payload* payload = (const kv_cast_in_payload*)data;

  const auto kv_src_dtype = payload->kv_src_dtype;
  const auto src_offset =
    index * payload->src_kv_sequence_length * payload->head_size;
  const auto dst_offset =
    index * payload->dst_kv_sequence_length * payload->head_size;
  const auto convert_size = payload->curr_kv_seq_len * payload->head_size;

  if (kv_src_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    bfloat16_buffer_to_float(
      payload->kv_cache_src + src_offset, convert_size,
      payload->kv_cache_dst + dst_offset
    );

  } else if (kv_src_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    float16_buffer_to_float(
      payload->kv_cache_src + src_offset, convert_size,
      payload->kv_cache_dst + dst_offset
    );
  }
}

RyzenMM::BufferRef AMDGQOKernel::runCpuGqa(
  const GQAattrs& gqa_attrs, Ort::BFloat16_t* q_data_ptr, uint16_t* k_data_ptr,
  Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data, uint16_t* past_v_data,
  uint16_t* present_k_data, uint16_t* present_v_data,
  const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
  int64_t past_sequence_length, uint64_t seq_len_k, uint64_t total_seq_len,
  bool past_present_share_buffer, OrtKernelContext* context
) {
  // assume only 1 set of KV cache buffers
  if (!past_present_share_buffer) {
    throw std::runtime_error(
      "In Hybrid Flow, GQO only support past_kv with shared buffer."
    );
  }

  // how much of KV cache is currently populated
  int64_t curr_kv_seq_len = total_seq_len - gqa_attrs.seq_len;

  // inputs: q, k, v, past_key and past_value
  auto float_q_data_converter =
    allocator_.AllocateBuffer(gqa_attrs.q_num * sizeof(float));
  auto float_k_data_converter =
    allocator_.AllocateBuffer(gqa_attrs.k_num * sizeof(float));
  auto float_v_data_converter =
    allocator_.AllocateBuffer(gqa_attrs.v_num * sizeof(float));

  auto kv_cache_shape = gqa_attrs.past_k_shape;

  float* k_cache_ptr = (float*)past_k_data;
  float* v_cache_ptr = (float*)past_v_data;

  const auto kv_dtype = gqa_attrs.present_k_data_type;
  bool alloc_kv_fp32 = (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT != kv_dtype);

  RyzenMM::BufferRef float_past_k_data_converter;
  RyzenMM::BufferRef float_past_v_data_converter;

  if (alloc_kv_fp32) {
    // NOTE: batch size is assumed to be 1
    // auto batch_size = kv_cache_shape.at(1);
    auto kv_num_heads = kv_cache_shape.at(1);
    // with past present this can be maximally allocated kv buffer
    // for temporary buffer only allocate what we need right now
    // auto kv_seq_len = kv_cache_shape.at(2);
    auto kv_head_dim = kv_cache_shape.at(3);
    auto kv_buffer_size =
      kv_num_heads * total_seq_len_ * kv_head_dim * sizeof(float);

    kv_cache_shape.at(2) = total_seq_len_;

    float_past_k_data_converter = allocator_.AllocateBuffer(kv_buffer_size);
    float_past_v_data_converter = allocator_.AllocateBuffer(kv_buffer_size);

    k_cache_ptr = float_past_k_data_converter.Data<float>();
    v_cache_ptr = float_past_v_data_converter.Data<float>();
  }

  // outputs
  auto output_size = num_heads_ * gqa_attrs.seq_len * gqa_attrs.head_size;
  auto float_output_data_converter =
    allocator_.AllocateBuffer(output_size * sizeof(float));

  Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  RecordDuration(Metric::Casting, [&]() {
    bfloat16_buffer_to_float(
      (uint16_t*)q_data_ptr, gqa_attrs.q_num,
      float_q_data_converter.Data<float>()
    );
    bfloat16_buffer_to_float(
      k_data_ptr, gqa_attrs.k_num, float_k_data_converter.Data<float>()
    );
    bfloat16_buffer_to_float(
      (uint16_t*)v_data_ptr, gqa_attrs.v_num,
      float_v_data_converter.Data<float>()
    );
  });

  // KV cache can be bfloat16/float16 or float32
  // if float32 pass models KV cache
  if ((gqa_attrs.present_k_data_type ==
       ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) ||
      (gqa_attrs.present_k_data_type ==
       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)) {
    Ort::KernelContext ctx{context};

    kv_cast_in_payload k_payload = {
      past_sequence_length,
      kv_cache_shape.at(2),
      curr_kv_seq_len,
      gqa_attrs.head_size,
      gqa_attrs.present_k_data_type,
      past_k_data,
      float_past_k_data_converter.Data<float>()
    };

    ctx.ParallelFor(
      kv_cast_row, static_cast<size_t>(kv_num_heads_), 0, &k_payload
    );

    kv_cast_in_payload v_payload = {
      past_sequence_length,
      kv_cache_shape.at(2),
      curr_kv_seq_len,
      gqa_attrs.head_size,
      gqa_attrs.present_k_data_type,
      past_v_data,
      float_past_v_data_converter.Data<float>()
    };

    ctx.ParallelFor(
      kv_cast_row, static_cast<size_t>(kv_num_heads_), 0, &v_payload
    );
  }

  // Create single Ort tensors
  auto t = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ort_gqa_.execute(
    float_output_data_converter.Data<float>(), k_cache_ptr, v_cache_ptr,
    float_q_data_converter.Data<float>(), float_k_data_converter.Data<float>(),
    float_v_data_converter.Data<float>(), k_cache_ptr, v_cache_ptr,
    gqa_attrs.q_shape, gqa_attrs.k_shape, kv_cache_shape, kv_cache_shape,
    kv_cache_shape, kv_cache_shape, gqa_attrs.seq_len, cos_cache, sin_cache,
    head_sink_, context
  );

  /// convert float32 output to bfloat16
  auto output_ptr1 = allocator_.AllocateBuffer(output_size * sizeof(uint16_t));

  RecordDuration(Metric::Casting, [&]() {
    float_buffer_to_bfloat16(
      float_output_data_converter.Data<float>(), output_size,
      (uint16_t*)output_ptr1.Data()
    );
  });

  for (int n = 0; n < kv_num_heads_; n++) {
    size_t src_offset = n * kv_cache_shape.at(2) * gqa_attrs.head_size +
                        curr_kv_seq_len * gqa_attrs.head_size;
    size_t dst_offset = n * past_sequence_length * gqa_attrs.head_size +
                        curr_kv_seq_len * gqa_attrs.head_size;
    int convert_size =
      (kv_cache_shape.at(2) - curr_kv_seq_len) * gqa_attrs.head_size;

    RecordDuration(Metric::Casting, [&]() {
      if (gqa_attrs.present_k_data_type ==
          ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        float_buffer_to_bfloat16(
          float_past_k_data_converter.Data<float>() + src_offset, convert_size,
          present_k_data + dst_offset
        );
        float_buffer_to_bfloat16(
          float_past_v_data_converter.Data<float>() + src_offset, convert_size,
          present_v_data + dst_offset
        );
      } else if (gqa_attrs.present_k_data_type ==
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        float_buffer_to_float16(
          float_past_k_data_converter.Data<float>() + src_offset, convert_size,
          present_k_data + dst_offset
        );
        float_buffer_to_float16(
          float_past_v_data_converter.Data<float>() + src_offset, convert_size,
          present_v_data + dst_offset
        );
      }
    });
  }
  return output_ptr1;
}

std::pair<Ort::BFloat16_t*, Ort::BFloat16_t*> AMDGQOKernel::copyRewindKvCache(
  RyzenMM::BufferRef& kv_concat_buf_bf16, uint16_t* past_k_data,
  uint16_t* past_v_data, const GQAattrs& gqa_attrs, int64_t total_seq_len,
  int64_t past_sequence_length, int64_t rewind_pos, OrtKernelContext* context
) {
  int full_S_padded =
    mha_aie_kernel_info_.tryPadSeq(total_seq_len, maxSeqLength());

  // 1. allocator dest buffer;
  size_t fullkv_size =
    gqa_attrs.batch_size * kv_num_heads_ * full_S_padded * gqa_attrs.head_size;
  kv_concat_buf_bf16 = allocator_.AllocateBuffer(
    2 * fullkv_size * sizeof(uint16_t)
  );  // bf16 for K & V
  auto k_concat_buf_bf16_p = kv_concat_buf_bf16.Data<Ort::BFloat16_t>();
  auto v_concat_buf_bf16_p =
    kv_concat_buf_bf16.Data<Ort::BFloat16_t>() + fullkv_size;
  memset(k_concat_buf_bf16_p, 0, 2 * fullkv_size * sizeof(uint16_t));

  // |--old(rewind_pos)-----|---new(seq_len)---|--- 0(blank part) -----|
  // |-----------  total_seq_len    -------------|
  // |-------------  full_S_padded   ----------- ------ ----------- ---|

  // 2. convert old kv from fp16 to bf16 and copy to dest buffer
  uint16_t* pask_k_p_base = past_k_data;
  uint16_t* pask_v_p_base = const_cast<uint16_t*>(past_v_data);
  size_t N1_K_size_old = 1 * 1 * rewind_pos * gqa_attrs.head_size;
  const std::array<int64_t, 4> N1_K_shape_old = {
    1, 1, rewind_pos, gqa_attrs.head_size
  };
  const std::vector<int64_t> N1_K_shape_old_2 = {
    1, 1, rewind_pos, gqa_attrs.head_size
  };

  if (gqa_attrs.present_k_data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    RyzenMM::BufferRef bf16_buf =
      allocator_.AllocateBuffer(N1_K_size_old * sizeof(uint16_t));

    Ort::BFloat16_t* data_bf16 =
      reinterpret_cast<Ort::BFloat16_t*>(bf16_buf.Data<uint16_t>());
    Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value bf16_kv = Ort::Value::CreateTensor<Ort::BFloat16_t>(
      memory_info, data_bf16, N1_K_size_old, N1_K_shape_old.data(),
      N1_K_shape_old.size()
    );

    // past_sequence_length : 4096
    for (int i = 0; i < kv_num_heads_; i++) {
      // convert K
      uint16_t* pask_k_p = pask_k_p_base + i * past_sequence_length *
                                             gqa_attrs.head_size;  // i*4096*128
      Ort::Float16_t* data_fp16 = reinterpret_cast<Ort::Float16_t*>(pask_k_p);
      ort_cast_fp16_to_bf16_.execute(
        data_bf16, data_fp16, N1_K_shape_old_2, context
      );

      // copy converted old k to dest
      size_t offset_old_dst =
        i * full_S_padded * gqa_attrs.head_size;  // i*full_S_padded*128
      memcpy(
        k_concat_buf_bf16_p + offset_old_dst, bf16_buf.Data<uint16_t>(),
        N1_K_size_old * sizeof(uint16_t)
      );

      // next, convert V
      uint16_t* pask_v_p = pask_v_p_base + i * past_sequence_length *
                                             gqa_attrs.head_size;  // i*4096*128
      data_fp16 = reinterpret_cast<Ort::Float16_t*>(pask_v_p);
      ort_cast_fp16_to_bf16_.execute(
        data_bf16, data_fp16, N1_K_shape_old_2, context
      );

      // copy old v to dest
      memcpy(
        v_concat_buf_bf16_p + offset_old_dst, bf16_buf.Data<uint16_t>(),
        N1_K_size_old * sizeof(uint16_t)
      );
    }
  } else {
    for (int i = 0; i < kv_num_heads_; i++) {
      // copy converted old k to dest
      size_t offset_old_dst =
        i * full_S_padded * gqa_attrs.head_size;  // i*full_S_padded*128
      memcpy(
        k_concat_buf_bf16_p + offset_old_dst,
        pask_k_p_base + i * past_sequence_length * gqa_attrs.head_size,
        N1_K_size_old * sizeof(uint16_t)
      );
      // copy old v to dest
      memcpy(
        v_concat_buf_bf16_p + offset_old_dst,
        pask_v_p_base + i * past_sequence_length * gqa_attrs.head_size,
        N1_K_size_old * sizeof(uint16_t)
      );
    }
  }

  return {k_concat_buf_bf16_p, v_concat_buf_bf16_p};
}

void AMDGQOKernel::runAieMhaRewind(
  const GQAattrs& gqa_attrs, Ort::BFloat16_t* k_concat_buf_bf16_p,
  Ort::BFloat16_t* v_concat_buf_bf16_p, uint64_t rewind_pos, uint64_t S_padded,
  uint64_t total_seq_len, uint16_t* bf16_q_rope,
  Ort::BFloat16_t* bf16_k_rope_data, Ort::BFloat16_t* v_data_ptr
) {
  /* still need pad k & q!
     for q, original pad (seq_len->S_Padded) may not be same as current
     (seq_len->total_seq_len_Padded) */

  int old_S_padded = S_padded;
  int full_S_padded =
    mha_aie_kernel_info_.tryPadSeq(total_seq_len, maxSeqLength());
  size_t fullkv_size =
    gqa_attrs.batch_size * kv_num_heads_ * full_S_padded * gqa_attrs.head_size;

  // |--old(rewind_pos)-------|---new(seq_len)---|--- 0(blank part) -----|
  // |-----------  total_seq_len    -------------|
  // |-------------  full_S_padded     ----------- ------ ----------- ---|

  // copy new kv in bf16 format to dest buffer;
  size_t N1_K_size_new =
    1 * 1 * gqa_attrs.seq_len *
    gqa_attrs.head_size;  // batch==1, each piece in kv_num_heads_ =1
  for (int i = 0; i < kv_num_heads_; i++) {
    size_t offset_new_dst =
      (i * full_S_padded + rewind_pos) *
      gqa_attrs.head_size;  // no need *sizeof(uint16_t) here
    size_t offset_new_src = i * gqa_attrs.seq_len * gqa_attrs.head_size;
    MemCpy(
      k_concat_buf_bf16_p + offset_new_dst, bf16_k_rope_data + offset_new_src,
      N1_K_size_new * sizeof(uint16_t)
    );
    MemCpy(
      v_concat_buf_bf16_p + offset_new_dst, v_data_ptr + offset_new_src,
      N1_K_size_new * sizeof(uint16_t)
    );
  }

  size_t attention_mask_size =
    gqa_attrs.batch_size * 1 * full_S_padded * full_S_padded;
  assert(atten_mask_provider_ != nullptr);
  auto bf16_attention_mask = atten_mask_provider_->get(
    gqa_attrs.seq_len, total_seq_len, full_S_padded, local_window_size_
  );

  // always need pad in this rewind_pos !=0 case
  std::vector<int64_t> io_padded_shape{
    gqa_attrs.batch_size, num_heads_, full_S_padded, gqa_attrs.head_size
  };
  std::vector<int64_t> rpb_padded_shape{
    gqa_attrs.batch_size, 1, full_S_padded, full_S_padded
  };
  size_t io_padded_size =
    gqa_attrs.batch_size * num_heads_ * full_S_padded * gqa_attrs.head_size;
  size_t rpb_padded_size = gqa_attrs.batch_size * full_S_padded * full_S_padded;

  const auto output_padded_bf16 =
    allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));

  OrtTensor k_padded_tensor = {
    io_padded_shape, io_padded_size, (void*)k_concat_buf_bf16_p
  };
  OrtTensor v_padded_tensor = {
    io_padded_shape, io_padded_size, (void*)v_concat_buf_bf16_p
  };
  OrtTensor rpb_padded_tensor = {
    rpb_padded_shape, rpb_padded_size, bf16_attention_mask.Data()
  };
  // only for debug:
  // importKVFull_aie(k_padded_bf16.Data<uint16_t>(),
  // v_padded_bf16.Data<uint16_t>() );

  RyzenMM::BufferRef q_padded_bf16;
  if (!use_aie_rope_ || (use_aie_rope_ && old_S_padded != full_S_padded)) {
    q_padded_bf16 =
      allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));
  }
  if (!use_flash_mha_) {
    if (!use_aie_rope_) {
      pad_qkv(
        *this, bf16_q_rope, q_padded_bf16.Data<uint16_t>(), gqa_attrs.seq_len,
        full_S_padded, num_heads_, gqa_attrs.head_size
      );
    } else if (old_S_padded != full_S_padded) {
      pad_q(
        *this, bf16_q_rope, q_padded_bf16.Data<uint16_t>(), gqa_attrs.seq_len,
        old_S_padded, full_S_padded, num_heads_, gqa_attrs.head_size
      );
    }
  }
  OrtTensor q_padded_tensor = {
    io_padded_shape, io_padded_size,
    use_flash_mha_ || ((use_aie_rope_ && (old_S_padded == full_S_padded)))
      ? (void*)bf16_q_rope
      : q_padded_bf16.Data()
  };
  const_cast<AMDGQOKernel*>(this)->aie_execute(
    gqa_attrs, q_padded_tensor, k_padded_tensor, v_padded_tensor,
    rpb_padded_tensor, rewind_pos, total_seq_len, local_window_size_
  );
}

void AMDGQOKernel::runAieMha(
  const GQAattrs& gqa_attrs, int64_t total_seq_len, uint16_t* bf16_q_rope,
  Ort::BFloat16_t* bf16_k_rope_data, Ort::BFloat16_t* v_data_ptr,
  uint16_t* bf16_k_rope_padded, int64_t S_padded
) {
  const int64_t rewind_pos = 0;
  // inputs: q, k, v, attention_mask
  // outputs: q, k, v
  // if aie_supp_seq_flag is true, then no need to pad q and k
  bool aie_supp_seq_flag =
    mha_aie_kernel_info_.isSeqSupported(gqa_attrs.seq_len);
  int batch_size = gqa_attrs.batch_size;
  int head_size = gqa_attrs.head_size;
  size_t attention_mask_size = batch_size * 1 * S_padded * S_padded;
  assert(atten_mask_provider_ != nullptr);
  auto bf16_attention_mask =
    atten_mask_provider_->get(S_padded, local_window_size_);

  // any other seq_len in between need pad.
  if (aie_supp_seq_flag) {
    std::vector<int64_t> q_shape_transposed{
      batch_size, num_heads_, gqa_attrs.seq_len, head_size
    };
    /// Q K V tensor for aie kernel
    OrtTensor qTensor = {
      q_shape_transposed, gqa_attrs.q_num, (void*)bf16_q_rope
    };
    // TODO(varunsh): shouldn't this be using k_num and v_num?
    OrtTensor kTensor = {
      q_shape_transposed, gqa_attrs.q_num, (void*)bf16_k_rope_data
    };
    OrtTensor vTensor = {
      q_shape_transposed, gqa_attrs.q_num, (void*)v_data_ptr
    };

    OrtTensor attention_mask = {
      {batch_size, 1, S_padded, S_padded},
      attention_mask_size,
      bf16_attention_mask.Data()
    };
    const_cast<AMDGQOKernel*>(this)->aie_execute(
      gqa_attrs, qTensor, kTensor, vTensor, attention_mask, rewind_pos,
      total_seq_len, local_window_size_
    );
  } else {
    std::vector<int64_t> io_padded_shape{
      batch_size, num_heads_, S_padded, head_size
    };
    std::vector<int64_t> rpb_padded_shape{batch_size, 1, S_padded, S_padded};
    size_t io_padded_size = batch_size * num_heads_ * S_padded * head_size;
    size_t rpb_padded_size = batch_size * S_padded * S_padded;

    const auto v_padded_bf16 =
      allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));
    /* const auto rpb_padded_bf16 =
      allocator_.AllocateBuffer(rpb_padded_size * sizeof(uint16_t)); */
    const auto output_padded_bf16 =
      allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));

    try_pad_qkv(
      *this, v_padded_bf16.Data<uint16_t>(), (uint16_t*)v_data_ptr,
      kv_num_heads_, gqa_attrs.seq_len, S_padded, head_size
    );
    // try_pad_mask(rpb_padded_bf16, bf16_attention_mask, seq_len,
    // S_padded); pad_rpb(bf16_attention_mask, rpb_padded_bf16, seq_len,
    // S_padded);

    OrtTensor v_padded_tensor = {
      io_padded_shape, io_padded_size, (void*)v_padded_bf16.Data()
    };
    OrtTensor rpb_padded_tensor = {
      rpb_padded_shape, rpb_padded_size, bf16_attention_mask.Data()
    };
    OrtTensor output_padded_tensor = {
      io_padded_shape, io_padded_size, (void*)output_padded_bf16.Data()
    };

    // no need to pad for q and k when using aie rope
    if (use_aie_rope_) {
      OrtTensor q_padded_tensor = {
        io_padded_shape, io_padded_size, (void*)bf16_q_rope
      };
      OrtTensor k_padded_tensor = {
        io_padded_shape, io_padded_size, (void*)bf16_k_rope_padded
      };

      const_cast<AMDGQOKernel*>(this)->aie_execute(
        gqa_attrs, q_padded_tensor, k_padded_tensor, v_padded_tensor,
        rpb_padded_tensor, rewind_pos, total_seq_len, local_window_size_
      );
    } else {
      const auto q_padded_bf16 =
        allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));
      const auto k_padded_bf16 =
        allocator_.AllocateBuffer(io_padded_size * sizeof(uint16_t));

      pad_qkv(
        *this, bf16_q_rope, q_padded_bf16.Data<uint16_t>(), gqa_attrs.seq_len,
        S_padded, num_heads_, head_size
      );
      try_pad_qkv(
        *this, k_padded_bf16.Data<uint16_t>(), (uint16_t*)bf16_k_rope_data,
        kv_num_heads_, gqa_attrs.seq_len, S_padded, head_size
      );

      OrtTensor q_padded_tensor = {
        io_padded_shape, io_padded_size, (void*)q_padded_bf16.Data()
      };
      OrtTensor k_padded_tensor = {
        io_padded_shape, io_padded_size, (void*)k_padded_bf16.Data()
      };

      const_cast<AMDGQOKernel*>(this)->aie_execute(
        gqa_attrs, q_padded_tensor, k_padded_tensor, v_padded_tensor,
        rpb_padded_tensor, rewind_pos, total_seq_len, local_window_size_
      );
    }
  }
}

RyzenMM::BufferRef AMDGQOKernel::runCpuRope(
  const GQAattrs& gqa_attrs, int64_t total_seq_len, Ort::BFloat16_t* q_data_ptr,
  Ort::BFloat16_t* bf16_k_rope_data, uint16_t* k_data_ptr,
  const int32_t* seq_len_k, uint64_t rewind_pos,
  const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
  OrtKernelContext* context
) {
  // rope pos ids
  // fix from
  // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/bert/group_query_attention.cc#L146
  int64_t seq_len = gqa_attrs.seq_len;
  int64_t past_seqlen = total_seq_len - seq_len;
  std::vector<int64_t> pos_ids(
    seq_len == 1 ? gqa_attrs.batch_size : gqa_attrs.batch_size * seq_len
  );
  if (seq_len == 1) {
    for (int b = 0; b < gqa_attrs.batch_size; b++) {
      pos_ids[b] = static_cast<int64_t>(seq_len_k[b]);
    }
  } else {
    if (0) {  //(is_prefill && prefill_cnt < 32){
      pos_ids[0] = static_cast<int64_t>(0);
    } else {
      for (int b = 0; b < gqa_attrs.batch_size; b++) {
        for (auto s = 0; s < seq_len; s++) {
          if (past_seqlen + s < total_seq_len) {
            pos_ids[b * seq_len + s] = static_cast<int64_t>(past_seqlen) + s;
          } else {
            pos_ids[b * seq_len + s] = static_cast<int64_t>(1);
          }
        }
      }
    }
  }

  // shared buffer for q and kv
  auto buffer_size =
    2 * alignTo4096(std::max(gqa_attrs.q_num, gqa_attrs.k_num) * sizeof(float));

  // process q
  auto buffer_q_buffer = allocator_.AllocateBuffer(buffer_size);
  auto* buffer_q = (uint16_t*)buffer_q_buffer.Data();
  uint16_t* bf16_q_data_transposed =
    buffer_q + (buffer_size / 2 / sizeof(float));
  RecordDuration(Metric::Transpose, [&]() {
    ort_transpose_.execute(
      (Ort::BFloat16_t*)bf16_q_data_transposed, q_data_ptr,
      {gqa_attrs.batch_size, gqa_attrs.seq_len, num_heads_,
       gqa_attrs.head_size},
      context
    );
  });
  float* fp32_q_data = reinterpret_cast<float*>(buffer_q);
  RecordDuration(Metric::Casting, [&]() {
    bfloat16_buffer_to_float(
      bf16_q_data_transposed, gqa_attrs.q_num, fp32_q_data
    );
  });
  auto fp32_q_rope_buffer =
    allocator_.AllocateBuffer(gqa_attrs.q_num * sizeof(float));
  auto fp32_q_rope = (float*)fp32_q_rope_buffer.Data();
  ort_rope_q_.execute(
    fp32_q_rope, fp32_q_data, pos_ids.data(), cos_cache, sin_cache,
    {gqa_attrs.batch_size, num_heads_, gqa_attrs.seq_len, gqa_attrs.head_size},
    context, rewind_pos
  );
  // bf16_q_rope_buffer = {};
  auto bf16_q_rope = (uint16_t*)fp32_q_rope;
  RecordDuration(Metric::Casting, [&]() {
    float_buffer_to_bfloat16(fp32_q_rope, gqa_attrs.q_num, bf16_q_rope);
  });

  // process k
  uint16_t* bf16_k_data_transposed =
    buffer_q + (buffer_size / 2 / sizeof(float));
  RecordDuration(Metric::Transpose, [&]() {
    ort_transpose_.execute(
      (Ort::BFloat16_t*)bf16_k_data_transposed, (Ort::BFloat16_t*)k_data_ptr,
      {gqa_attrs.batch_size, gqa_attrs.seq_len, kv_num_heads_,
       gqa_attrs.head_size},
      context
    );
  });
  float* fp32_k_data = reinterpret_cast<float*>(buffer_q);
  RecordDuration(Metric::Casting, [&]() {
    bfloat16_buffer_to_float(
      bf16_k_data_transposed, gqa_attrs.k_num, fp32_k_data
    );
  });
  auto fp32_k_rope_buffer =
    allocator_.AllocateBuffer(gqa_attrs.k_num * sizeof(float));
  float* fp32_k_rope = (float*)fp32_k_rope_buffer.Data();
  ort_rope_k_.execute(
    fp32_k_rope, fp32_k_data, pos_ids.data(), cos_cache, sin_cache,
    {gqa_attrs.batch_size, kv_num_heads_, gqa_attrs.seq_len,
     gqa_attrs.head_size},
    context, rewind_pos
  );
  RecordDuration(Metric::Casting, [&]() {
    float_buffer_to_bfloat16(
      fp32_k_rope, gqa_attrs.k_num,
      reinterpret_cast<uint16_t*>(bf16_k_rope_data)
    );
  });

  return fp32_q_rope_buffer;
}

std::pair<uint16_t*, uint16_t*> AMDGQOKernel::runAieRope(
  const GQAattrs& gqa_attrs, Ort::BFloat16_t* q_data_ptr, uint16_t* k_data_ptr,
  int64_t S_padded, int64_t rewind_pos, Ort::BFloat16_t* bf16_k_rope_data
) {
  PROFILING_START(runAieRope)

  uint16_t* bf16_k_rope_map_ptr = nullptr;
  if (rotary_interleaved_ ||
      mha_aie_kernel_info_.isNumHeadsSupported(kv_num_heads_)) {
    PROFILING_START(execute_mharopeK)
    bf16_k_rope_map_ptr = rope_aie_execute(
      k_data_ptr, kv_num_heads_, gqa_attrs.seq_len, gqa_attrs.head_size,
      S_padded, true, rewind_pos
    );
    PROFILING_END(execute_mharopeK, false, name().c_str())
  } else {
    // need to pad and tile for not supported num_head
    // since the supported num_head is enough, just maintain this branch
    // TODO(varunsh): shouldn't this be using k_num
    auto bf16_k_rope_or_pad =
      allocator_.AllocateBuffer(gqa_attrs.q_num * sizeof(uint16_t));
    pad_group_kv_BSNH(
      *this, (uint16_t*)bf16_k_rope_or_pad.Data(), k_data_ptr, num_heads_,
      kv_num_heads_, gqa_attrs.seq_len, gqa_attrs.head_size
    );
    bf16_k_rope_map_ptr = rope_aie_execute(
      (uint16_t*)bf16_k_rope_or_pad.Data(), kv_num_heads_, gqa_attrs.seq_len,
      gqa_attrs.head_size, S_padded, true, rewind_pos
    );
  }

  // maybe copy k data
  bool aie_supp_seq_flag =
    mha_aie_kernel_info_.isSeqSupported(gqa_attrs.seq_len);
  uint16_t* bf16_k_rope_padded = nullptr;
  if (aie_supp_seq_flag) {
    // no pad, copy all data
    MemCpy(
      bf16_k_rope_data, bf16_k_rope_map_ptr, gqa_attrs.k_num * sizeof(uint16_t)
    );
  } else {
    bf16_k_rope_padded = bf16_k_rope_map_ptr;
    // split data for k
    for (auto n = 0; n < kv_num_heads_; n++) {
      MemCpy(
        (void*)(bf16_k_rope_data + n * gqa_attrs.seq_len * gqa_attrs.head_size),
        (void*)(bf16_k_rope_map_ptr + n * S_padded * gqa_attrs.head_size),
        gqa_attrs.seq_len * gqa_attrs.head_size * sizeof(uint16_t)
      );
    }
  }
  PROFILING_START(execute_mharope2)
  auto bf16_q_rope = rope_aie_execute(
    (uint16_t*)q_data_ptr, num_heads_, gqa_attrs.seq_len, gqa_attrs.head_size,
    S_padded, false, rewind_pos
  );
  PROFILING_END(execute_mharope2, false, name().c_str())
  // PROFILING_END(sync_cp_ropeK_out, false, name().c_str())
  PROFILING_END(runAieRope, false, name().c_str())

  return {bf16_k_rope_padded, bf16_q_rope};
}

struct append_row_payload {
  std::uint16_t* present_v_data;
  const std::uint16_t* v_data_ptr;
  int kv_total_seq_len;
  int rewind_pos;
  int seq_len;
  int kv_num_heads;
  int head_size;
  bool cast_kv_to_fp16;
};

static void append_row_v_cache(void* payload, size_t index) {
  const append_row_payload* row_payload = (append_row_payload*)payload;

  auto seq_len = row_payload->seq_len;
  auto seq_idx = index % seq_len;
  auto kv_head_idx = index / seq_len;

  auto seq_id = seq_idx + row_payload->rewind_pos;
  auto head_size = row_payload->head_size;
  auto cast_kv_to_fp16 = row_payload->cast_kv_to_fp16;
  auto kv_total_seq_len = row_payload->kv_total_seq_len;

  // src layout [num_kv_heads, seq_len, head_size]
  // dst layout [num_kv_heads, kv_total_seq_len, head_size]

  std::uint16_t* dest_ptr =
    &row_payload->present_v_data
       [kv_head_idx * kv_total_seq_len * head_size + seq_id * head_size];
  const std::uint16_t* src_ptr =
    &row_payload
       ->v_data_ptr[kv_head_idx * seq_len * head_size + seq_idx * head_size];

  if (!cast_kv_to_fp16) {
    memcpy(dest_ptr, src_ptr, head_size * sizeof(std::uint16_t));
  } else {
    convert_buffer_bfloat16_to_float16(dest_ptr, src_ptr, head_size);
  }
}

static void append_v_cache(
  std::uint16_t* present_v_data, const std::uint16_t* v_data_ptr,
  int kv_total_seq_len, int rewind_pos, int seq_len, int kv_num_heads,
  int head_size, bool cast_kv_to_fp16, OrtKernelContext* context
) {
  Ort::KernelContext ctx{context};

  append_row_payload payload = {present_v_data,   v_data_ptr,
                                kv_total_seq_len, rewind_pos,
                                seq_len,          kv_num_heads,
                                head_size,        cast_kv_to_fp16};

  ctx.ParallelFor(
    append_row_v_cache, static_cast<size_t>(kv_num_heads * seq_len), 0, &payload
  );
}

struct rotate_and_append_row_payload {
  std::uint16_t* present_k_data;
  const std::uint16_t* k_data_ptr;
  int kv_total_seq_len;
  int rewind_pos;
  int dest_pos_offset;
  int seq_len;
  int kv_num_heads;
  int head_size;
  bool cast_kv_to_fp16;
  int do_rotary;
  int rotary_interleaved;
  const float* cos_cache_data;
  const float* sin_cache_data;
  int max_context_size;
  int half_rotary_dim;
};

static void rotate_and_append_row_k_cache(void* payload, size_t index) {
  const rotate_and_append_row_payload* rotate_payload =
    (rotate_and_append_row_payload*)payload;

  // src layout [num_kv_heads, seq_len, head_size]
  // dst layout [num_kv_heads, rewind_pos:rewind_pos+seq_len, head_size]
  // rotation applied looking at position id = rewind_pos + seq_idx and
  // and on [num_kv_heads, seq_len, 0:rotary_dim elements] remainder is pass
  // through

  auto seq_len = rotate_payload->seq_len;
  auto kv_total_seq_len = rotate_payload->kv_total_seq_len;
  auto rewind_pos = rotate_payload->rewind_pos;
  auto max_rot_pos_id = rotate_payload->max_context_size - 1;

  auto head_size = rotate_payload->head_size;

  auto seq_idx = index % seq_len;
  auto kv_head_idx = index / seq_len;

  int pos_id = seq_idx + rewind_pos;
  auto dest_pos_id = pos_id - rotate_payload->dest_pos_offset;

  const std::uint16_t* src_ptr =
    &rotate_payload
       ->k_data_ptr[kv_head_idx * seq_len * head_size + seq_idx * head_size];
  std::uint16_t* dest_ptr =
    &rotate_payload->present_k_data
       [kv_head_idx * kv_total_seq_len * head_size + dest_pos_id * head_size];

  auto half_rotary_dim = rotate_payload->half_rotary_dim;
  const float* cos_cache_base =
    &rotate_payload
       ->cos_cache_data[std::min(pos_id, max_rot_pos_id) * half_rotary_dim];
  const float* sin_cache_base =
    &rotate_payload
       ->sin_cache_data[std::min(pos_id, max_rot_pos_id) * half_rotary_dim];

  auto rotary_interleaved = rotate_payload->rotary_interleaved;
  auto cast_kv_to_fp16 = rotate_payload->cast_kv_to_fp16;

  constexpr int kRoundingMode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

  if (rotary_interleaved) {
    constexpr auto kBlockSize = 16;
    constexpr auto kCacheBlockSize = 8;
    const auto num_blocks = (2 * half_rotary_dim) / kBlockSize;

    // each pair  [x_2i, x_2i+1] -> look up cos[i], sin[i]
    // calculate [cos[i]*x_2i - sin[i]*x_2i+1, sin[i]*x_2i + cos[i]*x_2i+1]

    const __m512i even_idx = _mm512_setr_epi32(
      0, 2, 4, 6, 8, 10, 12, 14,
      // upper 8 values don't matter because we output only 256 bits
      0, 0, 0, 0, 0, 0, 0, 0
    );

    const __m512i odd_idx = _mm512_setr_epi32(
      1, 3, 5, 7, 9, 11, 13, 15,
      // upper 8 values don't matter because we output only 256 bits
      0, 0, 0, 0, 0, 0, 0, 0
    );

    const __m512i interleave_idx =
      _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

    for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
      const std::uint16_t* curr_src_ptr = &src_ptr[kBlockSize * block_idx];

      const float* cos_cache_ptr = &cos_cache_base[kCacheBlockSize * block_idx];
      const float* sin_cache_ptr = &sin_cache_base[kCacheBlockSize * block_idx];

      std::uint16_t* curr_dest_ptr = &dest_ptr[kBlockSize * block_idx];

      // read 16 bfloat16 - 8 in each 128-bit
      __m256i src_packed16_i =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr));

      // cast from bloat16 to float32
      __m512 src = _mm512_cvtpbh_ps((__m256bh)src_packed16_i);

      // construct de-interleaved src
      __m512 x_0_tmp = _mm512_permutexvar_ps(even_idx, src);
      __m256 x_0 = _mm512_castps512_ps256(x_0_tmp);

      __m512 x_1_tmp = _mm512_permutexvar_ps(odd_idx, src);
      __m256 x_1 = _mm512_castps512_ps256(x_1_tmp);

      // load cos/sin cache
      __m256 cos = _mm256_loadu_ps(cos_cache_ptr);
      __m256 sin = _mm256_loadu_ps(sin_cache_ptr);

      // apply rotation
      __m256 x_0_rot = _mm256_mul_ps(x_1, sin);
      x_0_rot = _mm256_fmsub_ps(x_0, cos, x_0_rot);

      __m256 x_1_rot = _mm256_mul_ps(x_1, cos);
      x_1_rot = _mm256_fmadd_ps(x_0, sin, x_1_rot);

      // interleave output
      __m512 out = _mm512_castps256_ps512(x_0_rot);
      out = _mm512_insertf32x8(out, x_1_rot, 1);
      out = _mm512_permutexvar_ps(interleave_idx, out);

      if (cast_kv_to_fp16) {
        // cast from float32 to float16
        __m256i out_fp16 = _mm512_cvtps_ph(out, kRoundingMode);
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr), out_fp16
        );
      } else {
        // cast from float32 to bfloat16
        __m256bh out_bf16 = _mm512_cvtneps_pbh(out);
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr), (__m256i)out_bf16
        );
      }
    }

    auto offset = num_blocks * kBlockSize;

    if (offset != 2 * half_rotary_dim) {
      throw std::runtime_error("Implement leftover interleaved rotate K");
    }

  } else {
    constexpr auto kBlockSize = 16;
    constexpr auto kCacheBlockSize = 16;
    const auto num_blocks = half_rotary_dim / kBlockSize;

    // each pair  [x_i, x_i+half_rotary_dim] -> look up cos[i], sin[i]
    // calculate [cos[i]*x_i - sin[i]*x_i+half_rotary_dim, sin[i]*x_i +
    // cos[i]*x_i+half_rotary_dim]

    for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
      const std::uint16_t* curr_src_ptr_0 = &src_ptr[kBlockSize * block_idx];
      const std::uint16_t* curr_src_ptr_1 =
        &src_ptr[kBlockSize * block_idx + half_rotary_dim];

      const float* cos_cache_ptr = &cos_cache_base[kCacheBlockSize * block_idx];
      const float* sin_cache_ptr = &sin_cache_base[kCacheBlockSize * block_idx];

      std::uint16_t* curr_dest_ptr_0 = &dest_ptr[kBlockSize * block_idx];
      std::uint16_t* curr_dest_ptr_1 =
        &dest_ptr[kBlockSize * block_idx + half_rotary_dim];

      // read 16 bfloat16 from each half
      __m256i src_packed16_0_i =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr_0));

      __m256i src_packed16_1_i =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr_1));

      // cast from bloat16 to float32
      __m512 x_0 = _mm512_cvtpbh_ps((__m256bh)src_packed16_0_i);
      __m512 x_1 = _mm512_cvtpbh_ps((__m256bh)src_packed16_1_i);

      // load cos/sin cache
      __m512 cos = _mm512_loadu_ps(cos_cache_ptr);
      __m512 sin = _mm512_loadu_ps(sin_cache_ptr);

      // apply rotation
      __m512 x_0_rot = _mm512_mul_ps(x_1, sin);
      x_0_rot = _mm512_fmsub_ps(x_0, cos, x_0_rot);

      __m512 x_1_rot = _mm512_mul_ps(x_1, cos);
      x_1_rot = _mm512_fmadd_ps(x_0, sin, x_1_rot);

      if (cast_kv_to_fp16) {
        // cast from float32 to float16
        __m256i out_0_fp16 = _mm512_cvtps_ph(x_0_rot, kRoundingMode);
        __m256i out_1_fp16 = _mm512_cvtps_ph(x_1_rot, kRoundingMode);
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr_0), out_0_fp16
        );
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr_1), out_1_fp16
        );
      } else {
        // cast from float32 to bfloat16
        __m256bh out_0_bf16 = _mm512_cvtneps_pbh(x_0_rot);
        __m256bh out_1_bf16 = _mm512_cvtneps_pbh(x_1_rot);

        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr_0), (__m256i)out_0_bf16
        );
        _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(curr_dest_ptr_1), (__m256i)out_1_bf16
        );
      }
    }

    auto offset = num_blocks * kBlockSize;

    if (offset != half_rotary_dim) {
      throw std::runtime_error("Implement leftover rotate K");
    }
  }

  // copy over remainder
  if (head_size > 2 * half_rotary_dim) {
    auto non_rot_elems = head_size - 2 * half_rotary_dim;
    const std::uint16_t* curr_src_ptr = &src_ptr[2 * half_rotary_dim];
    std::uint16_t* curr_dest_ptr = &dest_ptr[2 * half_rotary_dim];

    if (!cast_kv_to_fp16) {
      memcpy(
        curr_dest_ptr, curr_src_ptr, non_rot_elems * sizeof(std::uint16_t)
      );
    } else {
      convert_buffer_bfloat16_to_float16(
        curr_dest_ptr, curr_src_ptr, non_rot_elems
      );
    }
  }
}

static void rotate_and_append_k_cache(
  std::uint16_t* present_k_data, const std::uint16_t* k_data_ptr,
  int kv_total_seq_len, int rewind_pos, int dest_pos_offset, int seq_len,
  int kv_num_heads, int head_size, bool cast_kv_to_fp16, int do_rotary,
  int rotary_interleaved, Ort::Value& cos_cache, Ort::Value& sin_cache,
  OrtKernelContext* context
) {
  Ort::KernelContext ctx{context};

  const float* cos_cache_data = cos_cache.GetTensorData<float>();
  const float* sin_cache_data = sin_cache.GetTensorData<float>();

  int max_context_size = 0;
  int half_rotary_dim = 0;

  if (0 != do_rotary) {
    auto cache_shape = cos_cache.GetTensorTypeAndShapeInfo().GetShape();

    max_context_size = cache_shape.at(0);
    half_rotary_dim = cache_shape.at(1);
  }

  rotate_and_append_row_payload payload = {
    present_k_data,  k_data_ptr,       kv_total_seq_len,   rewind_pos,
    dest_pos_offset, seq_len,          kv_num_heads,       head_size,
    cast_kv_to_fp16, do_rotary,        rotary_interleaved, cos_cache_data,
    sin_cache_data,  max_context_size, half_rotary_dim
  };

  ctx.ParallelFor(
    rotate_and_append_row_k_cache, static_cast<size_t>(kv_num_heads * seq_len),
    0, &payload
  );
}

void AMDGQOKernel::runAieChunkGqa(
  const GQAattrs& gqa_attrs, int64_t total_seq_len, Ort::BFloat16_t* q_data_ptr,
  uint16_t* k_data_ptr, Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data,
  uint16_t* past_v_data, uint16_t* present_k_data, uint16_t* present_v_data,
  const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
  int64_t past_sequence_length, const int32_t* seq_len_k,
  bool past_present_share_buffer, bool with_custom_allocator,
  OrtKernelContext* context
) {
  if (!past_present_share_buffer) {
    throw std::runtime_error(
      "Only supported for past/present share buffer case"
    );
  }

  // NOTE: right now MHA only implemented using BMM1/CPU softmax/BMM2
  rebind_bmm_params();

  // start position id
  auto rewind_pos = total_seq_len - gqa_attrs.seq_len;
  bool is_token_phase = (1 == gqa_attrs.seq_len);

  // bfloat16 or float16
  const auto kv_dtype = gqa_attrs.present_k_data_type;

  if (ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 != kv_dtype &&
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 != kv_dtype) {
    throw std::runtime_error("Unsupported KV dtype, only bfloat16/float16");
  }

  // QKV that comes here is assumed to be bfloat16
  const bool cast_kv = (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 == kv_dtype);

  // assume transpose [0, 1, 2, 3] to [0, 2, 1, 3] has already been applied to
  // Q/K/V [batch_size, seq_len, num_heads, head_size] -> [batch_size,
  // num_heads, seq_len, head_size]

  // append current V to V cache
  // format is [kv_num_head, alloc_total_seq_len, hidden_dim]
  // will do strided copy to [kv_num_head, rewind_pos:rewind_pos + seq_len,
  // hidden_dim]
  const auto kv_total_seq_len = gqa_attrs.present_k_shape.at(2);

  append_v_cache(
    present_v_data, (const std::uint16_t*)v_data_ptr, kv_total_seq_len,
    rewind_pos, gqa_attrs.seq_len, kv_num_heads_, gqa_attrs.head_size, cast_kv,
    context
  );

  // similar to K
  // NOTE: we need a separate copy for Q since this is currently in bmm1 in0
  //       buffer which gets clobbered by bmm2 output buffer
  std::vector<std::uint16_t> q_rot(
    num_heads_ * gqa_attrs.seq_len * gqa_attrs.head_size
  );

  // then apply rotation and finally strided write back to KV cache similar to V
  if (do_rotary_) {
    rotate_and_append_k_cache(
      present_k_data, k_data_ptr, kv_total_seq_len, rewind_pos, 0,
      gqa_attrs.seq_len, kv_num_heads_, gqa_attrs.head_size, cast_kv,
      do_rotary_, rotary_interleaved_, ss_->cos_cache_fp32_tensor_,
      ss_->sin_cache_fp32_tensor_, context
    );

    rotate_and_append_k_cache(
      q_rot.data(), (const std::uint16_t*)q_data_ptr, gqa_attrs.seq_len,
      rewind_pos, rewind_pos, gqa_attrs.seq_len, num_heads_,
      gqa_attrs.head_size, false, do_rotary_, rotary_interleaved_,
      ss_->cos_cache_fp32_tensor_, ss_->sin_cache_fp32_tensor_, context
    );
  } else {
    // if no rotation reuse what is done for V
    append_v_cache(
      present_k_data, k_data_ptr, kv_total_seq_len, rewind_pos,
      gqa_attrs.seq_len, kv_num_heads_, gqa_attrs.head_size, cast_kv, context
    );

    // copy Q without rotation
    memcpy(
      q_rot.data(), q_data_ptr,
      num_heads_ * gqa_attrs.seq_len * gqa_attrs.head_size *
        sizeof(std::uint16_t)
    );
  }

  // run MHA
  std::vector<int64_t> q_shape = {
    gqa_attrs.batch_size, num_heads_, gqa_attrs.seq_len, gqa_attrs.head_size
  };

  std::vector<int64_t> kv_shape = gqa_attrs.present_k_shape;

  OrtTensor query_states = {q_shape, gqa_attrs.q_num, (void*)q_rot.data()};

  OrtTensor key_states = {kv_shape, gqa_attrs.k_num, (void*)present_k_data};

  OrtTensor value_states = {kv_shape, gqa_attrs.k_num, (void*)present_v_data};

  if (is_token_phase) {
    token_mha_cpu(
      gqa_attrs, query_states, key_states, value_states, rewind_pos,
      total_seq_len, local_window_size_, cast_kv
    );
  } else {
    const bool use_aie_rope = false;
    chunked_mha_aie(
      gqa_attrs, query_states, key_states, value_states, rewind_pos,
      total_seq_len, local_window_size_, context_chunk_size_, use_aie_rope,
      cast_kv
    );
  }
}

std::future<void> AMDGQOKernel::runAieGqa(
  const GQAattrs& gqa_attrs, int64_t total_seq_len, Ort::BFloat16_t* q_data_ptr,
  uint16_t* k_data_ptr, Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data,
  uint16_t* past_v_data, uint16_t* present_k_data, uint16_t* present_v_data,
  const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
  int64_t past_sequence_length, const int32_t* seq_len_k,
  bool past_present_share_buffer, bool with_custom_allocator,
  OrtKernelContext* context
) {
  RyzenMM::BufferRef fp32_q_rope_buffer;
  RyzenMM::BufferRef bf16_k_rope_buffer;
  RyzenMM::BufferRef kv_concat_buf_bf16;
  // RyzenMM::BufferRef bf16_k_rope_padded_buffer;
  uint16_t* bf16_q_rope = nullptr;              // for q buffer padded or not
  uint16_t* bf16_k_rope_padded = nullptr;       // for k buffer padded if needed
  Ort::BFloat16_t* bf16_k_rope_data = nullptr;  // for k buffer no padded

  // TODO
  auto rewind_pos = total_seq_len - gqa_attrs.seq_len;

  std::future<std::pair<Ort::BFloat16_t*, Ort::BFloat16_t*>> init_kv_past;
  if (rewind_pos) {
    init_kv_past = std::async(
      std::launch::async, &AMDGQOKernel::copyRewindKvCache, this,
      std::ref(kv_concat_buf_bf16), past_k_data, past_v_data, gqa_attrs,
      total_seq_len, past_sequence_length, rewind_pos, context
    );
  }

  auto S_padded = gqa_attrs.seq_len;
  bool aie_supp_seq_flag =
    mha_aie_kernel_info_.isSeqSupported(gqa_attrs.seq_len);
  if (!aie_supp_seq_flag) {
    S_padded =
      mha_aie_kernel_info_.tryPadSeq(gqa_attrs.seq_len, maxSeqLength());
  }
  if (use_flash_mha_) {
    S_padded = npu_kernel_size_;
  }
  if (aie_supp_seq_flag)
    bf16_k_rope_data = (Ort::BFloat16_t*)shared_buffer_.Get("bmm1_in1").ptr;
  else {
    bf16_k_rope_buffer =
      allocator_.AllocateBuffer(gqa_attrs.k_num * sizeof(uint16_t));
    bf16_k_rope_data = (Ort::BFloat16_t*)bf16_k_rope_buffer.Data();
  }

  if (use_aie_rope_) {
    uint16_t* tmp = nullptr;
    std::tie(tmp, bf16_q_rope) = runAieRope(
      gqa_attrs, q_data_ptr, k_data_ptr,
      mha_aie_kernel_info_.tryPadSeq(gqa_attrs.seq_len, maxSeqLength()),
      rewind_pos, bf16_k_rope_data
    );
    if (tmp) {
      bf16_k_rope_padded = tmp;
    }
  } else {
    fp32_q_rope_buffer = runCpuRope(
      gqa_attrs, total_seq_len, q_data_ptr, bf16_k_rope_data, k_data_ptr,
      seq_len_k, rewind_pos, cos_cache, sin_cache, context
    );
    bf16_q_rope = reinterpret_cast<uint16_t*>(fp32_q_rope_buffer.Data());
  }

  /// save present k/v
  std::future<void> kv_cache_update;
  if (past_present_share_buffer) {
    kv_cache_update = std::async(
      std::launch::async, &AMDGQOKernel::updateKvCache, this, context,
      bf16_k_rope_data, v_data_ptr, present_k_data, present_v_data, past_k_data,
      past_v_data, gqa_attrs.past_k_shape[2],
      static_cast<int>(gqa_attrs.head_size), static_cast<int>(rewind_pos),
      static_cast<int>(kv_num_heads_), static_cast<int>(gqa_attrs.seq_len),
      gqa_attrs.present_k_data_type, with_custom_allocator
    );
  } else {
    auto group_num = num_heads_ / kv_num_heads_;
    for (int n = 0; n < kv_num_heads_; n++) {
      int offset_dst = n * gqa_attrs.seq_len * gqa_attrs.head_size;
      int offset_src = group_num * n * gqa_attrs.seq_len * gqa_attrs.head_size;
      MemCpy(
        present_k_data + offset_dst, bf16_k_rope_data + offset_src,
        gqa_attrs.seq_len * gqa_attrs.head_size * sizeof(uint16_t)
      );
    }
    MemCpy(present_v_data, v_data_ptr, gqa_attrs.v_num * sizeof(uint16_t));
  }
  if (rewind_pos == 0) {  // original mode
    runAieMha(
      gqa_attrs, total_seq_len, bf16_q_rope, bf16_k_rope_data, v_data_ptr,
      bf16_k_rope_padded, S_padded
    );
  } else {  // rewind_pos != 0
    init_kv_past.wait();
    auto [k_concat_buf_bf16_p, v_concat_buf_bf16_p] = init_kv_past.get();
    runAieMhaRewind(
      gqa_attrs, k_concat_buf_bf16_p, v_concat_buf_bf16_p, rewind_pos, S_padded,
      total_seq_len, bf16_q_rope, bf16_k_rope_data, v_data_ptr
    );
  }

  bf16_k_rope_buffer = {};
  kv_concat_buf_bf16 = {};

  return kv_cache_update;
  // bf16_k_rope_padded_buffer = {};
}

}  // namespace ryzenai::onnx_utils
