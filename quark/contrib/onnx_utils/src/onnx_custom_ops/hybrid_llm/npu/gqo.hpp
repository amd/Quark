// Copyright (c) 2022 Xilinx Inc.
// Copyright (c) 2023 Advanced Micro Devices, Inc.

#pragma once

#include <algorithm>
#include <cassert>
#include <future>
#include <mutex>
#include <ops/transformer/flash_mha.hpp>
#include <string>

#include "external_data.hpp"
#include "hybrid_llm/ort/cast.hpp"
#include "hybrid_llm/ort/gqa.hpp"
#include "hybrid_llm/ort/rotary_embedding.hpp"
#include "hybrid_llm/ort/transpose.hpp"
#include "jit_node.hpp"
#include "lora_op_interface.hpp"
#include "matmulnbits.hpp"
#include "npu_op.hpp"
#include "npu_utils.hpp"
#include "onnxruntime_cxx_api.h"
#include "ops/bmm/bmm.hpp"
#include "ops/maskedsoftmax/maskedsoftmax.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "ops/mladfmharope/mladfmharope.hpp"
#include "profile.h"
#include "ryzenai/ryzen_mm.h"
#include "xrt/xrt_bo.h"

namespace ryzenai::onnx_utils {

// #define NPU_GQO_PROFILE
#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && defined(NPU_GQO_PROFILE)
#define NPU_GQO_PROFILE_EN
#endif

class GqaKernelInfo {
 public:
  int64_t tryPadSeq(int64_t S, size_t max_seq_len) const;
  // check if the seq_len is supported by aie
  bool isSeqSupported(int seq_len) const;
  // check if the num_head is supported by aie
  bool isNumHeadsSupported(int num_head) const;
  void setVersion(const MladfVersion& version);

 private:
  const static inline std::set<int> kSupportedSeqLenV1{128,  256,  512, 1024,
                                                       2048, 3072, 4096};
  const static inline std::set<int> kSupportedSeqLenV2{128,  256,  512,  640,
                                                       1024, 1152, 1280, 1536,
                                                       1664, 2048, 2176, 2304,
                                                       2560, 2816, 3072, 4096};
  const static inline std::set<int> kSupportedNumHeads{1,  2,  3,  4,  8, 9,
                                                       12, 16, 24, 28, 32};

  std::set<int> supported_seqlen_;
};

class AttnMaskLut {
 public:
  AttnMaskLut();

  bool has(int32_t seq_len) const;

  auto get(int32_t seq_len) const;

  static void fill(uint16_t* attn_mask, int seq_len, int64_t local_window_size);

 private:
  std::unordered_map<int32_t, ryzenai::RyzenMM::BufferRef> attn_masks_lut_;
};

class AttnMaskProvider {
 public:
  AttnMaskProvider(
    const GqaKernelInfo* aie_kernel_info,
    std::weak_ptr<const AttnMaskLut>& attn_mask_lut_weak
  )
    : aie_kernel_info_(aie_kernel_info) {
    if (!(attn_mask_lut_ = attn_mask_lut_weak.lock())) {
      attn_mask_lut_ = std::make_shared<AttnMaskLut>();
      attn_mask_lut_weak = attn_mask_lut_;
    }
  }

  ~AttnMaskProvider() = default;

  RyzenMM::BufferRef get(int32_t seq_len, int64_t local_window_size);
  RyzenMM::BufferRef get(
    int32_t seq_len, int32_t T, int32_t T_pad, int64_t local_window_size
  );

 private:
  bool use_lut(int32_t seq_len) const;
  const GqaKernelInfo* aie_kernel_info_{nullptr};
  std::shared_ptr<const AttnMaskLut> attn_mask_lut_;
};

struct GQAattrs {
  int64_t batch_size;
  int64_t head_size;
  int64_t seq_len;

  std::vector<int64_t> past_k_shape;
  size_t past_k_num;
  std::vector<int64_t> past_v_shape;
  size_t past_v_num;
  std::vector<int64_t> present_k_shape;
  size_t present_k_num;
  ONNXTensorElementDataType present_k_data_type;
  std::vector<int64_t> present_v_shape;
  size_t present_v_num;

  std::vector<int64_t> q_shape;
  size_t q_num;
  std::vector<int64_t> k_shape;
  size_t k_num;
  std::vector<int64_t> v_shape;
  size_t v_num;
};

class AMDGQOKernel : public JitNode<AMDGQOKernel>,
                     public NpuOp,
                     public LoraOpInterface {
 public:
  AMDGQOKernel(
    const OrtKernelInfo* info, const OrtApi& api,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void set_kv_cache(void* shared_k_cache, void* shared_v_cache);
  void set_params_bmm(size_t index);
  void rebind_bmm_params();
  void rebind_flashmha_params();

  // void initialize_matmul_nbits(const OrtKernelInfo* k_info);
  uint16_t* rope_aie_execute(
    const uint16_t* input, const int N, const int S, const int H,
    const int pad_S, bool b_isK, int iRewindPos
  );
  void executeMatMulNBitsAie(
    const uint16_t* input_data, uint16_t* out, std::vector<int64_t> input_shape,
    std::pair<size_t, size_t> wts_shape, int grp_size, int run_cnt
  );
  void executeMatMulNBitsAie(
    std::vector<xrt::bo>& inputs, uint16_t* out,
    std::vector<int64_t> output_shape, std::pair<size_t, size_t> wts_shape,
    int grp_size, int run_cnt
  );
  void aie_execute(
    const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
    OrtTensor& value_states, OrtTensor& attention_mask, int iRewindPos,
    int64_t total_seq_len, int64_t local_window_size
  );
  void Compute(OrtKernelContext* context, bool with_custom_allocator);
  ~AMDGQOKernel();

  void readDataImpl(int idx) override;
  void loadDataImpl(int idx) override;
  void unloadDataImpl(int idx) override {};
  void UpdateSharedBuffer(size_t kernel_size) override final;

  void initializeLora();
  void loadLoraData();
  void LoadLora() override;

 private:
  void initializeKernels() final;
  void initializeMatMulNBitsKernel();
  bool setUseAieRope(
    const Ort::ConstKernelInfo& info, int64_t& rotary_embedding_dim,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void setUseAieGqa(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void setUseFlashMHA(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void setEnableContextChunk(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void setContextChunkSize(
    const std::unordered_map<std::string, std::string>& session_configs
  );

  bool run_context_chunk(size_t total_seq_len, size_t seq_len);

  void parseUnpackedWeights(const Ort::ConstKernelInfo& info);
  void initializeMatMulNBits(
    const std::vector<int8_t>& b, const std::vector<int8_t>& zeros,
    const std::vector<float>& scales, const std::vector<float>& bias
  );
  void initializeMatMulNBits(
    const uint8_t* preformat_consts, size_t size, bool use_shared_buffer = false
  );
  bool executeMatMulNBits(
    OrtKernelContext* context, const std::vector<int64_t>& qkv_shape,
    bool is_prefill, int64_t seq_len, const uint16_t* mm_inp,
    bool use_aie_chunk_gqa
  );
  void updateKvCache(
    OrtKernelContext* context, Ort::BFloat16_t* k_rope_data,
    Ort::BFloat16_t* v_data_ptr, uint16_t* present_k_data,
    uint16_t* present_v_data, uint16_t* past_k_data, uint16_t* past_v_data,
    uint64_t total_seq_len, const int head_size, int iRewindPos, const int N_kv,
    const int seq_len, ONNXTensorElementDataType present_k_data_type,
    bool with_custom_allocator
  );
  const Ort::BFloat16_t* inputCast(
    std::vector<Ort::BFloat16_t>& npu_qkv_input,
    const Ort::ConstValue& packed_qkv, OrtKernelContext* context
  );
  RyzenMM::BufferRef runCpuGqa(
    const GQAattrs& gqa_attrs, Ort::BFloat16_t* q_data_ptr,
    uint16_t* k_data_ptr, Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data,
    uint16_t* past_v_data, uint16_t* present_k_data, uint16_t* present_v_data,
    const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
    int64_t past_sequence_length, uint64_t seq_len_k, uint64_t total_seq_len,
    bool past_present_share_buffer, OrtKernelContext* context
  );
  std::future<void> runAieGqa(
    const GQAattrs& gqa_attrs, int64_t total_seq_len,
    Ort::BFloat16_t* q_data_ptr, uint16_t* k_data_ptr,
    Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data, uint16_t* past_v_data,
    uint16_t* present_k_data, uint16_t* present_v_data,
    const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
    int64_t past_sequence_length, const int32_t* seq_len_k,
    bool past_present_share_buffer, bool with_custom_allocator,
    OrtKernelContext* context
  );

  void runAieChunkGqa(
    const GQAattrs& gqa_attrs, int64_t total_seq_len,
    Ort::BFloat16_t* q_data_ptr, uint16_t* k_data_ptr,
    Ort::BFloat16_t* v_data_ptr, uint16_t* past_k_data, uint16_t* past_v_data,
    uint16_t* present_k_data, uint16_t* present_v_data,
    const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
    int64_t past_sequence_length, const int32_t* seq_len_k,
    bool past_present_share_buffer, bool with_custom_allocator,
    OrtKernelContext* context
  );

  RyzenMM::BufferRef runCpuRope(
    const GQAattrs& gqa_attrs, int64_t total_seq_len,
    Ort::BFloat16_t* q_data_ptr, Ort::BFloat16_t* bf16_k_rope_data,
    uint16_t* k_data_ptr, const int32_t* seq_len_k, uint64_t rewind_pos,
    const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
    OrtKernelContext* context
  );
  std::pair<uint16_t*, uint16_t*> runAieRope(
    const GQAattrs& gqa_attrs, Ort::BFloat16_t* q_data_ptr,
    uint16_t* k_data_ptr, int64_t S_padded, int64_t rewind_pos,
    Ort::BFloat16_t* bf16_k_rope_data
  );

  void save_present_kv_to_shared_buffer(
    uint16_t* dst_k, uint16_t* dst_v, uint16_t* past_k, uint16_t* past_v,
    int irewindpos, const uint16_t* src_k, const uint16_t* src_v,
    const int num_heads, const int num_group, const int buffer_seq_len,
    const int seq_len, const int head_size
  );

  std::pair<Ort::BFloat16_t*, Ort::BFloat16_t*> copyRewindKvCache(
    RyzenMM::BufferRef& kv_concat_buf_bf16, uint16_t* past_k_data,
    uint16_t* past_v_data, const GQAattrs& gqa_attrs, int64_t total_seq_len,
    int64_t past_sequence_length, int64_t rewind_pos, OrtKernelContext* context
  );

  void runAieMhaRewind(
    const GQAattrs& gqa_attrs, Ort::BFloat16_t* k_concat_buf_bf16_p,
    Ort::BFloat16_t* v_concat_buf_bf16_p, uint64_t rewind_pos,
    uint64_t S_padded, uint64_t total_seq_len, uint16_t* bf16_q_rope,
    Ort::BFloat16_t* bf16_k_rope_data, Ort::BFloat16_t* v_data_ptr
  );

  void runAieMha(
    const GQAattrs& gqa_attrs, int64_t total_seq_len, uint16_t* bf16_q_rope,
    Ort::BFloat16_t* bf16_k_rope_data, Ort::BFloat16_t* v_data_ptr,
    uint16_t* bf16_k_rope_padded, int64_t S_padded
  );

  void chunked_mha_aie(
    const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
    OrtTensor& value_states, const int rewind_pos, const int64_t total_seq_len,
    const int64_t local_window_size, const int32_t context_chunk_size,
    const bool use_aie_rope, const bool cast_kv_bfloat16
  );

  void token_mha_cpu(
    const GQAattrs& gqa_attrs, OrtTensor& query_states, OrtTensor& key_states,
    OrtTensor& value_states, const int rewind_pos, const int64_t total_seq_len,
    const int64_t local_window_size, const bool cast_kv_bfloat16
  );

  int getContextChunkSize(
    int seq_len, int total_seq_len, int local_window_size
  );

  const std::vector<size_t>& setFlashMhaGranularity() const;

  bool is_prefill_ = true;
  int64_t q_seq_len_ = 0;
  int64_t total_seq_len_ = 0;
  int64_t do_rotary_ = 0;
  int64_t kv_num_heads_ = 0;
  int64_t num_heads_ = 0;
  int64_t rotary_interleaved_ = 0;
  int64_t local_window_size_ = -1;
  float scale_;
  int64_t head_size_ = 0;
  Ort::Logger m_logger{nullptr};
  static inline const std::vector<size_t> kFlashMhaGranularityAie2_ = {
    128,  256,  512,  640,  1024, 1152, 1280, 1536,
    1664, 2048, 2176, 2304, 2560, 2816, 3072, 4096
  };
  static inline const std::vector<size_t> kFlashMhaGranularityAie4_ = {
    128, 256, 512, 1024, 2048, 3072, 4096
  };

  static constexpr size_t kHeadSinkIdx = 11;
  // threshold over which we will calculate attention in chunks
  // for now chosen based on npu kernel performance scaling
  // smaller shapes are currently not as efficient
  // which would lead to unnecessary overhead if tiled with those shapes
  static inline constexpr std::int32_t kContextMaxSize = 2048;
  // token phase currently runs on CPU and dims
  // will be [num_heads, seq_len = 1, kTokenContextMaxSize]
  // so have higher threshold before needing to chunk
  static inline constexpr std::int32_t kTokenContextMaxSize = 16384;
  std::int32_t context_size_threshold_ = kContextMaxSize;
  std::int32_t context_chunk_size_ = kContextMaxSize;
  size_t bmm1_input0_scratch_offset_ = 0;
  size_t bmm1_input0_scratch_size_ = 0;
  size_t bmm1_input_scratch_offset_ = 0;
  size_t bmm1_input_scratch_size_ = 0;
  size_t bmm2_input_scratch_offset_ = 0;
  size_t bmm2_input_scratch_size_ = 0;

  void* shared_k_cache_{nullptr};
  void* shared_v_cache_{nullptr};

  OrtKernelContext* context_p_;

  std::vector<xrt::bo> mm_outputs_{};
  /// RoPE
  uint16_t* sin_ = nullptr;
  uint16_t* cos_ = nullptr;
  uint16_t* trig_max_len_ = nullptr;
  Ort::ConstValue const_cos_, const_sin_;
  // head sink
  std::vector<float> head_sink_;
  // mha aie kernel info
  GqaKernelInfo mha_aie_kernel_info_;
  // this matches the bit width in external_data.proto
  ExternalTensorInfo jit_tensor_info_;

  int requantize_in_scale_;
  int requantize_out_scale_;
  MatMulNBitsAttributes matmulnbits_attrs_;
  // int k_k_, k_n_, k_bits_, k_block_size_;
  int k_asymmetric_ = 0;
  std::tuple<int, int> wts_shape_;
  std::vector<int32_t> wts_sum_;
  std::string impl_;
  std::string quant_mode_;
  Ort::ConstValue m_packed_consts_;

  std::vector<size_t> qkv_out_size_{};
  void* flash_in_data_ptr_ = nullptr;
  void* flash_out_data_ptr_ = nullptr;
  bool use_flash_mha_ = false;
  bool use_aie_rope_ = true;
  bool use_aie_gqo_ = true;
  bool context_chunk_en_ = false;
  size_t npu_kernel_size_ = 0;
  size_t npu_kernel_size_q_ = 0;
  bool use_context_chunk_ = false;
  bool has_head_sink_ = false;
  int64_t rotary_embedding_dim_ = 0;
  bool wait_for_data_;
  int cnt_;
  bool has_scratch_buffer_{false};

  // attention provider
  std::unique_ptr<AttnMaskProvider> atten_mask_provider_;

  std::vector<int64_t> input_cast_indices_;
  std::vector<int64_t> output_cast_indices_;

  std::vector<Ort::BFloat16_t> npu_output_;

  LoraBuffer lora_buffers_;

  inline static const char* const op_type_ = "GQO";

  OrtTranspose<Ort::BFloat16_t> ort_transpose_;
  OrtRotaryEmbedding ort_rope_q_;
  OrtRotaryEmbedding ort_rope_k_;
  OrtCast<Ort::Float16_t, Ort::BFloat16_t> ort_cast_fp16_to_bf16_;
  OrtCast<Ort::BFloat16_t, Ort::Float16_t> ort_cast_bf16_to_fp16_;
  OrtCast<Ort::BFloat16_t, float> ort_cast_bf16_to_fp32_;
  OrtCast<Ort::Float16_t, float> ort_cast_fp16_to_fp32_;
  OrtGQA ort_gqa_;
  RyzenMM::BufferRef sink_data_;
  xrt::bo flash_out_;
  std::string rope_cache_name_;
  xrt::bo sink_bo_;
  std::unique_ptr<ryzenai::dynamic_dispatch::transformer::flash_mha<
    uint16_t, uint16_t, uint16_t>>
    flash_mha_{nullptr};
#ifdef NPU_GQO_PROFILE_EN
  std::vector<Duration> measurements_;
#endif

  RyzenMM::NPUAllocator<'GQO1'> allocator_;

  struct State {
    std::weak_ptr<const AttnMaskLut> attn_mask_lut_weak_;

    // aie kernels from DD
    std::unique_ptr<ryzenai::bmm<uint16_t, uint16_t, uint16_t>> bmm1_{nullptr};
    std::unique_ptr<ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>>
      softmax_{nullptr};
    std::unique_ptr<ryzenai::bmm<uint16_t, uint16_t, uint16_t>> bmm2_{nullptr};
    std::unique_ptr<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>> rope_{
      nullptr
    };

    // aie kernel bos
    std::vector<xrt::bo> rope_inbos_{};
    std::vector<xrt::bo> rope_outbos_{};
    std::vector<xrt::bo> bmm1_inputs_{};
    std::vector<xrt::bo> bmm1_outputs_{};
    std::vector<xrt::bo> bmm2_inputs_{};
    std::vector<xrt::bo> bmm2_outputs_{};
    xrt::bo flash_in_;
    xrt::bo flash_out_;
    xrt::bo softmax_mask_{};
    RyzenMM::BufferRef cos_sin_cache_;
    std::vector<int64_t> cos_shape_;
    int seq_len_{0};
    int rewind_pos_{0};
    int curr_local_window_size_{0};
    int64_t jit_max_bo_size_{0};
    std::vector<RyzenMM::BufferRef> bo_data_{};
    // Matmul Nbits  variables for output projection
    std::unique_ptr<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>
      gemm_{nullptr};
    void* gemm_last_scratch_ptr_{nullptr};
    size_t gemm_last_scratch_len_{0};
    int instances_ = 0;
    std::vector<float> cos_cache_fp32_, sin_cache_fp32_;
    Ort::Value cos_cache_fp32_tensor_, sin_cache_fp32_tensor_;
    std::string rope_name_;
  };

  SessionState<State> ss_;
};

extern template class JitNode<AMDGQOKernel>;

}  // namespace ryzenai::onnx_utils
