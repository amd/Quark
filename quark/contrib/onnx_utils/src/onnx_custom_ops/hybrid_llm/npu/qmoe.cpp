// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "qmoe.hpp"

#include <queue>

#include "common.hpp"
#include "npu_utils.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "ort.hpp"
#include "ryzenai/onnx_utils/string.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

/*
GPU and NPU both defines Tensor class in different namespaces, GPU defines
in ryzenai::onnx_utils and NPU defines in global namespace.In order to avoid
discrepancies we are renaming NPU tensors type.
*/

using NPUTensor = ::Tensor;

namespace ryzenai::onnx_utils {

#ifdef _WIN32
static void map_experts(
  const int64_t num_fc, const int64_t num_experts,
  const std::unordered_set<size_t>& expert_ids,
  const uint64_t base_tensor_offset, const int64_t packed_expert_sz_fc,
  DWORD gran, HANDLE hMapping, HANDLE hFile, size_t fc_index,
  std::map<std::pair<size_t, size_t>, LPVOID>& pBufs,
  std::map<std::pair<size_t, size_t>, const std::uint8_t*>& packed_fc_ptrs
) {
  for (size_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    auto fc_expert_idx = std::make_pair(fc_index, expert_idx);

    bool map_expert = (expert_ids.end() != expert_ids.find(expert_idx)) &&
                      (pBufs.end() == pBufs.find(fc_expert_idx));

    if (!map_expert) {
      continue;
    }

    const auto tensor_offset =
      base_tensor_offset + expert_idx * packed_expert_sz_fc;

    uint64_t viewOffset = (tensor_offset / gran) * gran;
    size_t delta = tensor_offset - viewOffset;

    DWORD offset_hi = viewOffset >> 32;
    DWORD offset_lo = static_cast<std::uint32_t>(viewOffset);
    DWORD size = packed_expert_sz_fc + (delta);

    auto mapped_ptr =
      MapViewOfFile(hMapping, FILE_MAP_READ, offset_hi, offset_lo, size);

    if (mapped_ptr == nullptr) {
      CloseHandle(hMapping);
      CloseHandle(hFile);
      throw std::runtime_error("Could not map view of file");
    }

    const std::uint8_t* packed_weights_ptr =
      static_cast<const std::uint8_t*>(mapped_ptr) + delta;

    pBufs[fc_expert_idx] = mapped_ptr;
    packed_fc_ptrs[fc_expert_idx] = packed_weights_ptr;
  }
}

static void unmap_experts(
  const std::unordered_set<size_t>& expert_ids,
  std::map<std::pair<size_t, size_t>, LPVOID>& pBufs,
  std::map<std::pair<size_t, size_t>, const std::uint8_t*>& packed_fc_ptrs,
  size_t fc_index
) {
  std::set<std::pair<size_t, size_t>> erase_indices;
  for (auto& [fc_expert_idx, p_buf] : pBufs) {
    if (fc_index != fc_expert_idx.first) {
      continue;
    }

    auto expert_idx = fc_expert_idx.second;

    bool unmap = expert_ids.end() == expert_ids.find(expert_idx);

    if (unmap) {
      UnmapViewOfFile(p_buf);
      erase_indices.insert(fc_expert_idx);
    }
  }

  for (const auto& fc_expert_index : erase_indices) {
    pBufs.erase(fc_expert_index);
    packed_fc_ptrs.erase(fc_expert_index);
  }
}
#endif

static void bind_experts(
  const int64_t num_experts, const std::unordered_set<size_t>& expert_ids,
  std::unordered_map<size_t, xrt::bo>& const_bo_map,
  const std::map<std::pair<size_t, size_t>, const std::uint8_t*>&
    packed_expert_ptrs,
  size_t fc_index, const int64_t packed_expert_sz,
  std::unique_ptr<
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>& gemm
) {
  constexpr bool kReadOnly = true;

  for (auto expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    bool need_expert = expert_ids.end() != expert_ids.find(expert_idx);
    bool is_loaded = const_bo_map.end() != const_bo_map.find(expert_idx);

    auto fc_expert_idx = std::make_pair(fc_index, expert_idx);

    if (need_expert && !is_loaded) {
      const_bo_map[expert_idx] = gemm->bind_bo(
        (void*)packed_expert_ptrs.at(fc_expert_idx), packed_expert_sz, kReadOnly
      );
    }
  }
}

static void unbind_experts(
  const int64_t num_experts, const std::unordered_set<size_t>& expert_ids,
  std::unordered_map<size_t, xrt::bo>& const_bo_map
) {
  for (auto expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    bool need_expert = expert_ids.end() != expert_ids.find(expert_idx);
    bool is_loaded = const_bo_map.end() != const_bo_map.find(expert_idx);

    if (!need_expert && is_loaded) {
      const_bo_map.erase(expert_idx);
    }
  }
}

void AMDQMoEKernel::clearExperts(bool is_gate_up) {
  if (is_gate_up) {
    gate_up_experts_.clear();
  } else {
    down_experts_.clear();
  }
}

void AMDQMoEKernel::initializeKernels() {
  auto mm_attrs = getCommonAttrs();
  mm_attrs["skip_create_input"] = 1;
  mm_attrs["skip_create_output"] = 1;
  mm_attrs["skip_create_token"] = 1;

  // Create operator instance
  if (!ss_->gate_up_gemm_) {
    ss_->gate_up_gemm_ = std::make_unique<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, mm_attrs
    );
  }

  if (!ss_->down_gemm_) {
    ss_->down_gemm_ = std::make_unique<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, mm_attrs
    );
  }
}

// Ctor
AMDQMoEKernel::AMDQMoEKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : NpuOp(k_info, session_configs), ss_(session_configs) {
#ifdef NPU_QMOE_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif
  layer_id = ss_->instances_++;
  // Get constant info for the node
  Ort::ConstKernelInfo kernel_info{k_info};

  auto header = initializeNpuOp(op_type_, session_configs, kernel_info);

  // setup default op attributes
  act_attributes_.activation_alpha =
    kernel_info.GetAttribute<float>("activation_alpha");
  act_attributes_.activation_beta =
    kernel_info.GetAttribute<float>("activation_beta");
  act_attributes_.swiglu_fusion =
    kernel_info.GetAttribute<int64_t>("swiglu_fusion");
  act_attributes_.activation_type =
    kernel_info.GetAttribute<std::string>("activation_type");
  act_attributes_.swiglu_limit =
    kernel_info.GetAttribute<float>("swiglu_limit");

  expert_weight_bits_ = kernel_info.GetAttribute<int64_t>("expert_weight_bits");
  top_k_ = kernel_info.GetAttribute<int64_t>("k");

  normalize_routing_weights_ =
    kernel_info.GetAttribute<int64_t>("normalize_routing_weights");

  use_sparse_mixer_ = kernel_info.GetAttribute<int64_t>("use_sparse_mixer");

  // custom op attributes
  num_experts_ = kernel_info.GetAttribute<int64_t>("num_experts");
  num_fc_ = kernel_info.GetAttribute<int64_t>("num_fc");

  if (2 != num_fc_) {
    throw std::runtime_error("QMoE only supports FC1 and FC2");
  }

  expert_attributes_[kFC1Idx].K_fc = kernel_info.GetAttribute<int64_t>("k_FC1");
  expert_attributes_[kFC1Idx].N_fc = kernel_info.GetAttribute<int64_t>("n_FC1");
  expert_attributes_[kFC1Idx].block_size_fc =
    kernel_info.GetAttribute<int64_t>("block_size_FC1");
  expert_attributes_[kFC1Idx].packed_expert_sz_fc =
    kernel_info.GetAttribute<int64_t>("packed_expert_sz_FC1");

  expert_attributes_[kFC2Idx].K_fc = kernel_info.GetAttribute<int64_t>("k_FC2");
  expert_attributes_[kFC2Idx].N_fc = kernel_info.GetAttribute<int64_t>("n_FC2");
  expert_attributes_[kFC2Idx].block_size_fc =
    kernel_info.GetAttribute<int64_t>("block_size_FC2");
  expert_attributes_[kFC2Idx].packed_expert_sz_fc =
    kernel_info.GetAttribute<int64_t>("packed_expert_sz_FC2");

  expert_attributes_[kFC3Idx].K_fc =
    getAttribute<int64_t>(kernel_info, "k_FC3", -1);
  expert_attributes_[kFC3Idx].N_fc =
    getAttribute<int64_t>(kernel_info, "n_FC3", -1);
  expert_attributes_[kFC3Idx].block_size_fc =
    getAttribute<int64_t>(kernel_info, "block_size_FC3", -1);
  expert_attributes_[kFC3Idx].packed_expert_sz_fc =
    getAttribute<int64_t>(kernel_info, "packed_expert_sz_FC3", -1);

  setMladfVersion(kernel_info);

  initializeDynamicDpm(session_configs);

  // get packed consts
  const auto input_count = kernel_info.GetInputCount();

  const auto last_input_name = kernel_info.GetInputName(input_count - 1);
  const bool packed_consts = endsWith(last_input_name, "packed.qexperts");
  use_external_data_ = false;

  if (!packed_consts) {
    throw std::runtime_error("Need packed inputs for FC1/FC2 in qmoe!\n");
  } else {
    int is_constant = 0;
    Ort::ConstValue packed_const =
      kernel_info.GetTensorConstantInput(input_count - 1, &is_constant);

    use_external_data_ =
      (0 == packed_const.GetTensorTypeAndShapeInfo().GetElementCount());
  }

  if (use_external_data_) {
    const auto& external_data_file = session_configs.at("external_data_file");

    if (!header->external_data().qmoe()) {
      throw std::runtime_error("Missing packed const in external data file!");
    }
    external_data_path_ = (fs::path(external_data_file).parent_path() /
                           header->external_data().filename())
                            .string();
  }

  // for now we will bind all by default
  // since it has been observed OS can delay releasing
  hybrid_opt_qmoe_dynamic_experts_ = 0;

  // this flag is for dynamically binding bo
  // if the packed weights are not in external file
  const auto& hybrid_opt_qmoe_dynamic_experts_str =
    session_configs.at("hybrid_opt_qmoe_dynamic_experts");

  // if ORT has already loaded data ignore user passed attribute
  if (!hybrid_opt_qmoe_dynamic_experts_str.empty() && use_external_data_) {
    hybrid_opt_qmoe_dynamic_experts_ =
      std::stoi(hybrid_opt_qmoe_dynamic_experts_str);
  }

  const auto& hybrid_opt_qmoe_num_dynamic_layers_str =
    session_configs.at("hybrid_opt_qmoe_num_dynamic_layers");

  int num_dynamic_layers = -1;

  if (!hybrid_opt_qmoe_num_dynamic_layers_str.empty()) {
    num_dynamic_layers = std::stoi(hybrid_opt_qmoe_num_dynamic_layers_str);
  }

  // allow selecting only first n QMoE layers to be dynamic
  if ((-1 != num_dynamic_layers) && (num_dynamic_layers <= layer_id)) {
    hybrid_opt_qmoe_dynamic_experts_ = 0;
  }

  if (use_external_data_) {
#ifdef _WIN32
    hFile_ = CreateFile(
      external_data_path_.c_str(),  // File name
      GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL, nullptr
    );

    if (hFile_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("Could not open file");
    }

    // always go through mmap
    hMapping_ = CreateFileMapping(
      hFile_, nullptr,
      PAGE_READONLY,  // Read access
      0, 0,
      nullptr
    );  // No name

    if (hMapping_ == nullptr) {
      CloseHandle(hFile_);
      throw std::runtime_error("Could not create file mapping");
    }

    SYSTEM_INFO si;
    GetSystemInfo(&si);

    gran_ = si.dwAllocationGranularity;  // usually 65536

    // should be read from protobuf
    for (size_t fc_index = 0; fc_index < num_fc_; fc_index++) {
      const auto& tensor = header->operators().at(name()).data().at(fc_index);
      auto tensor_offset = tensor.offset();
      auto tensor_size = tensor.size();

      external_fc_info_.push_back(std::make_pair(tensor_offset, tensor_size));
    }

    std::unordered_set<size_t> map_default_expert_ids;

    for (size_t expert_idx = 0; expert_idx < num_experts_; expert_idx++) {
      map_default_expert_ids.insert(expert_idx);
    }

    map_experts(
      num_fc_, num_experts_, map_default_expert_ids,
      external_fc_info_[kFC1Idx].first,
      expert_attributes_[kFC1Idx].packed_expert_sz_fc, gran_, hMapping_, hFile_,
      kFC1Idx, pBufs_, packed_fc_ptrs_
    );
    map_experts(
      num_fc_, num_experts_, map_default_expert_ids,
      external_fc_info_[kFC2Idx].first,
      expert_attributes_[kFC2Idx].packed_expert_sz_fc, gran_, hMapping_, hFile_,
      kFC2Idx, pBufs_, packed_fc_ptrs_
    );

    mapped_fc_ = true;
#else
    abort();
#endif
  } else {
    // read from ORT inputs
    for (size_t fc_index = 0; fc_index < num_fc_; fc_index++) {
      int is_constant = 0;
      Ort::ConstValue packed_const = kernel_info.GetTensorConstantInput(
        input_count - num_fc_ + fc_index, &is_constant
      );
      const uint8_t* packed_weights_ptr = packed_const.GetTensorData<uint8_t>();

      for (size_t expert_idx = 0; expert_idx < num_experts_; expert_idx++) {
        auto fc_expert_idx = std::make_pair(fc_index, expert_idx);

        packed_fc_ptrs_[fc_expert_idx] = packed_weights_ptr;
        packed_weights_ptr += expert_attributes_[fc_index].packed_expert_sz_fc;
      }
    }
  }

  initializeKernels();

  default_expert_ids_.clear();
  if (0 == hybrid_opt_qmoe_dynamic_experts_) {
    // default is bind all the experts weights to xrt::BO
    for (size_t expert_idx = 0; expert_idx < num_experts_; expert_idx++) {
      default_expert_ids_.insert(expert_idx);
    }
  }

  bind_experts(
    num_experts_, default_expert_ids_, gate_up_experts_, packed_fc_ptrs_,
    kFC1Idx, expert_attributes_[kFC1Idx].packed_expert_sz_fc, ss_->gate_up_gemm_
  );
  bind_experts(
    num_experts_, default_expert_ids_, down_experts_, packed_fc_ptrs_, kFC2Idx,
    expert_attributes_[kFC2Idx].packed_expert_sz_fc, ss_->down_gemm_
  );

  UpdateSharedBuffer(getNPUKernelGranularity(
    top_k_ * getInitPromptSize(session_configs), granularity_options
  ));

#ifdef NPU_QMOE_PROFILE_EN
  const Clock::time_point config_end = Clock::now();
  const Duration config_duration = config_end - config_start;
  measurements_.at(EventID::CONFIG_ID) += config_duration;
#endif
}

struct top_k_info {
  float val;
  size_t index;

  top_k_info() : val(0.0f), index(0) {}

  top_k_info(float v, size_t i) : val(v), index(i) {}
};

struct Compare {
  bool operator()(const top_k_info& a, const top_k_info& b) {
    return a.val > b.val;  // min-heap: top has smallest val
  }
};

struct top_k_payload {
  const std::uint16_t* base_router_data_ptr;
  top_k_info* base_out_data_ptr;
  int64_t k;
  int64_t num_experts;
};

static void find_top_k(void* data, size_t index) {
  const top_k_payload* payload = (top_k_payload*)data;

  const auto& base_router_data_ptr = payload->base_router_data_ptr;
  const auto& base_out_data_ptr = payload->base_out_data_ptr;
  const auto& k = payload->k;
  const auto& num_experts = payload->num_experts;

  const auto router_data_ptr = &base_router_data_ptr[index * num_experts];
  auto out_data_ptr = &base_out_data_ptr[index * k];

  std::priority_queue<top_k_info, std::vector<top_k_info>, Compare> min_heap;

  for (int64_t i = 0; i < num_experts; i++) {
    float val = bfloat16_to_float_single(router_data_ptr[i]);
    if (min_heap.size() < k) {
      min_heap.emplace(val, i);
    } else if (val > min_heap.top().val) {
      min_heap.pop();
      min_heap.emplace(val, i);
    }
  }

  while (!min_heap.empty()) {
    *out_data_ptr++ = min_heap.top();
    min_heap.pop();
  }
}

struct normalize_payload {
  top_k_info* base_out_data_ptr;
  int64_t k;
};

static void normalize_weights(void* data, size_t index) {
  const normalize_payload* payload = (normalize_payload*)data;

  const auto& base_out_data_ptr = payload->base_out_data_ptr;
  const auto& k = payload->k;

  auto out_data_ptr = &base_out_data_ptr[index * k];

  // step 1 - find max

  float max_val = out_data_ptr[0].val;

  for (int64_t i = 1; i < k; i++) {
    max_val = std::max(max_val, out_data_ptr[i].val);
  }

  // step 2 - subtract max and exponentiate, have running sum
  float sum_exp = 0.0f;
  for (int64_t i = 0; i < k; i++) {
    out_data_ptr[i].val -= max_val;
    out_data_ptr[i].val = std::expf(out_data_ptr[i].val);
    sum_exp += out_data_ptr[i].val;
  }

  // step 3 - calculate norm scale factor
  float recip_sum_exp = 0.0f;
  __m128 x = _mm_set_ss(sum_exp);
  x = _mm_rcp_ss(x);

  _mm_store_ss(&recip_sum_exp, x);

  // step 4 - apply scale factor
  for (int64_t i = 0; i < k; i++) {
    out_data_ptr[i].val *= recip_sum_exp;
  }
}

struct gather_payload {
  const size_t* token_to_idx_ptr;
  const std::uint16_t* base_input_act_data_ptr;
  std::uint16_t* base_dst_ptr;
  int64_t hidden_dim;
  int64_t k;
  size_t m;
  size_t datum_size;
};

static void gather_expert_tokens(void* data, size_t index) {
  const gather_payload* payload = (gather_payload*)data;

  auto src_idx = index / payload->k;  // which token
  auto dst_idx = payload->token_to_idx_ptr[index];

  const auto& hidden_dim = payload->hidden_dim;

  const auto* src_ptr = &payload->base_input_act_data_ptr[src_idx * hidden_dim];
  auto* dst_ptr = &payload->base_dst_ptr[dst_idx * hidden_dim];

  memcpy(dst_ptr, src_ptr, hidden_dim * payload->datum_size);
}

static void scatter_expert_tokens(void* data, size_t index) {
  const gather_payload* payload = (gather_payload*)data;
  const auto& k = payload->k;
  const auto& m = payload->m;

  auto src_idx = payload->token_to_idx_ptr[index];
  auto dst_idx_k = index % k;  // which expert
  auto dst_idx_m = index / k;  // which token

  const auto& hidden_dim = payload->hidden_dim;

  const auto* src_ptr = &payload->base_input_act_data_ptr[src_idx * hidden_dim];
  // output layout is [k_experts, tokens, hidden_dim]
  auto* dst_ptr =
    &payload->base_dst_ptr[(dst_idx_k * m + dst_idx_m) * hidden_dim];

  memcpy(dst_ptr, src_ptr, hidden_dim * payload->datum_size);
}

struct weighted_sum_payload {
  const std::uint16_t* base_src_ptr;
  const top_k_info* base_top_k_ptr;
  std::uint16_t* base_dst_ptr;
  int64_t hidden_dim;
  int64_t k;
  size_t m;
};

static void weighted_sum(void* data, size_t index) {
  const weighted_sum_payload* payload = (weighted_sum_payload*)data;
  const auto& hidden_dim = payload->hidden_dim;
  const auto& k = payload->k;
  const auto& m = payload->m;

  std::uint16_t* dst_ptr = &payload->base_dst_ptr[index * hidden_dim];
  const top_k_info* top_k_info_ptr = &payload->base_top_k_ptr[index * k];

  // for zero-initializing
  __m256i zeros = _mm256_setzero_si256();

  for (auto i = 0; i < k; i++) {
    // for each token, its experts output will be at [i, index, :]
    const std::uint16_t* src_ptr =
      &payload->base_src_ptr[(i * m + index) * hidden_dim];
    auto expert_weight = top_k_info_ptr[i].val;

    constexpr auto kBlockSize = 16;
    const auto num_blocks = hidden_dim / kBlockSize;

    __m512 scale = _mm512_set1_ps(expert_weight);

    for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
      const std::uint16_t* curr_src_ptr = &src_ptr[kBlockSize * block_idx];
      std::uint16_t* curr_dst_ptr = &dst_ptr[kBlockSize * block_idx];
      // read 16 bfloat16 - 8 in each 128-bit
      __m256i src_packed16 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_src_ptr));

      // cast from bloat16 to float32
      __m512 src = _mm512_cvtpbh_ps((__m256bh)src_packed16);

      // zero out dest to support in place accumulation
      if (0 == i) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(curr_dst_ptr), zeros);
      }

      // reload dest that is in bfloat16
      __m256i dst_packed16 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(curr_dst_ptr));

      // cast from bloat16 to float32
      __m512 dst = _mm512_cvtpbh_ps((__m256bh)dst_packed16);

      // accumulate into dest with fused multiply-add to apply expert weight
      dst = _mm512_fmadd_ps(src, scale, dst);

      // cast back to bfloat16
      __m256bh out_bf16 = _mm512_cvtneps_pbh(dst);

      // store to dest
      _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(curr_dst_ptr), (__m256i)out_bf16
      );
    }

    const auto offset = num_blocks * kBlockSize;

    for (auto j = offset; j < hidden_dim; j++) {
      float p_sum = bfloat16_to_float_single(src_ptr[j]) * expert_weight;

      // zero-initialize dest if this is first partial sum
      //  do this after reading to support in-place
      dst_ptr[j] = (0 == i) ? 0 : dst_ptr[j];

      p_sum += bfloat16_to_float_single(dst_ptr[j]);
      dst_ptr[j] = float_to_bfloat16_2(p_sum);
    }
  }
}

struct swiglu_payload {
  const std::uint16_t* base_src_ptr;
  std::uint16_t* base_dst_ptr;
  int64_t hidden_dim;
  int64_t top_k;
  float alpha;
  float beta;
  float limit;
  std::string activation_type;
  int64_t swiglu_fusion;
};

static inline float sigmoid(float x) {
  // Sigmoid(x) = 1/(1 + e^-x) = e^x / (1 + e^x)
  //                           = 1 - 1 / (1 + e^x)
  // for numerical stability can use following
  //  x >= 0  : 1/(1 + e^-x)
  //  x < 0   : 1 - 1 / (1 + e^x)
  float val = 0.0f;

  if (x >= 0) {
    val = (1.0f / (1.0f + std::expf(-x)));
  } else {
    val = 1 - (1.0f / (1.0f + std::expf(x)));
  }

  return val;
}

static void swiglu(void* data, size_t index) {
  const swiglu_payload* payload = (swiglu_payload*)data;

  const auto& hidden_dim = payload->hidden_dim;

  const std::uint16_t* src_ptr = &payload->base_src_ptr[index * hidden_dim];
  std::uint16_t* dst_ptr = &payload->base_dst_ptr[index * hidden_dim / 2];
  const auto& limit = payload->limit;
  const auto& alpha = payload->alpha;
  const auto& beta = payload->beta;

  // we assume for swiglu linear layers have already been run
  // and columns are interleaved
  // g = xW + b
  // l = xV + c
  // only perform following
  // G = clamp(g, max=limit)
  // L = clamp(l, min=-limit, max=limit)
  // swiglu = G * sigmoid(alpha * G) * (L + beta)

  constexpr auto kBlockSize = 32;
  const auto num_blocks = hidden_dim / kBlockSize;

  __m512 max = _mm512_set1_ps(limit);
  __m512 min = _mm512_set1_ps(-limit);
  __m512 half = _mm512_set1_ps(0.5f);
  __m512 scale = _mm512_set1_ps(alpha);
  __m512 bias = _mm512_set1_ps(beta);

  __m512i odd_mask = _mm512_set_epi16(
    0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0,
    0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0,
    0xFFFF, 0, 0xFFFF, 0
  );

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    const std::uint16_t* curr_src_ptr = &src_ptr[kBlockSize * block_idx];
    std::uint16_t* curr_dst_ptr = &dst_ptr[(kBlockSize / 2) * block_idx];

    // 32 interleaved bfloat16
    __m512i packed32 =
      _mm512_loadu_si512(reinterpret_cast<const __m512i*>(curr_src_ptr));

    // 16 de-interleaved uint32/float32
    __m512i g_i = _mm512_slli_epi32(packed32, 16);
    __m512i l_i = _mm512_and_si512(packed32, odd_mask);

    // cast for compiler
    __m512 g = _mm512_castsi512_ps(g_i);
    __m512 l = _mm512_castsi512_ps(l_i);

    // clamps
    g = _mm512_min_ps(g, max);
    l = _mm512_max_ps(_mm512_min_ps(l, max), min);

    // l + beta
    l = _mm512_add_ps(l, bias);

    // sigmoid(x) = 0.5 *(1 + tanh(0.5 * x))
    // sigmoid(alpha * x) = 0.5 * (1 + tanh(alpha * 0.5 * x))
    // g * sigmoid(alpha * g) = 0.5 * g * (1 + tanh(alpha * 0.5 * g))
    //                        = (0.5 * g) + (0.5 * g) * (tanh(alpha * 0.5 * g))
    g = _mm512_mul_ps(g, half);
#ifdef _WIN32
    g = _mm512_fmadd_ps(g, _mm512_tanh_ps(_mm512_mul_ps(g, scale)), g);
#else
    throw std::runtime_error("implement avx tanh");
#endif

    // output - 16 32-bit floats
    g = _mm512_mul_ps(g, l);

    // cast from float32 to bfloat16
    __m256bh out_bf16 = _mm512_cvtneps_pbh(g);

    // write out to dest
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(curr_dst_ptr), (__m256i)out_bf16
    );
  }

  // number of pairs processed - scalar compute for remainder
  const auto offset = (num_blocks * kBlockSize) / 2;

  for (int64_t i = offset; i < hidden_dim / 2; i++) {
    float x_0 = bfloat16_to_float_single(src_ptr[2 * i]);
    float x_1 = bfloat16_to_float_single(src_ptr[2 * i + 1]);

    x_0 = std::min(x_0, limit);
    x_1 = std::min(std::max(x_1, -limit), limit);

    float y = x_0 * sigmoid(alpha * x_0) * (x_1 + beta);

    dst_ptr[i] = float_to_bfloat16_2(y);
  }
}

void AMDQMoEKernel::UpdateSharedBuffer(size_t kernel_size) {
  /* gate_up_matmul -> general_activation -> down_matmul
   *  initial implementation with only matmul on NPU
   *  TODO: create general_activation DD operator
   */

  constexpr size_t kBatchSize = 1;

  size_t a_bo_size = 0;
  size_t c_bo_size = 0;

  size_t scratch_size = 4096;

  // gate-up matmul
  {
    auto K_fc = expert_attributes_[kFC1Idx].K_fc;
    auto N_fc = expert_attributes_[kFC1Idx].N_fc;
    auto block_size_fc = expert_attributes_[kFC1Idx].block_size_fc;
    std::vector<size_t> a_shape = {
      kBatchSize, kernel_size, static_cast<size_t>(K_fc)
    };
    std::vector<size_t> b_shape = {
      static_cast<size_t>(K_fc), static_cast<size_t>(N_fc)
    };

    std::vector<size_t> c_shape = {
      kBatchSize, kernel_size, static_cast<size_t>(N_fc)
    };

    NPUTensor input_tensor = {nullptr, a_shape, "bfloat16"};
    NPUTensor wts_tensor = {nullptr, b_shape, "bfloat16"};
    NPUTensor out_tensor = {nullptr, c_shape, "bfloat16"};
    NPUTensor placeholder;
    std::vector<NPUTensor> inputs = {input_tensor, wts_tensor,  placeholder,
                                     placeholder,  placeholder, out_tensor};
    std::vector<NPUTensor> outputs;
    std::map<std::string, std::any> attrs;
    std::vector<int> grp_size = {static_cast<int>(block_size_fc)};
    attrs["group_size"] = grp_size;
    attrs["mem_opt"] = 1;
    std::vector<OpArgMap> arg_map =
      ss_->gate_up_gemm_->get_buffer_reqs(inputs, outputs, attrs);
    auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());

    // input for gate_up
    a_bo_size = size_map["in0"];
    // output for gate_up - will be re-used for general activations input
    c_bo_size = size_map["out"];
    if (mladfVersion() == MladfVersion::v2) {
      size_t scratch_bo_size = size_map["scratch"];
      scratch_size = std::max(scratch_size, alignTo4096(scratch_bo_size));
    }
  }

  // general activation will take [M, 2 * hidden_dim] -> [M, hidden_dim]
  // reuse input of gate_up

  // check constraints for down matmul
  size_t down_out_size = 0;
  {
    auto K_fc = expert_attributes_[kFC2Idx].K_fc;
    auto N_fc = expert_attributes_[kFC2Idx].N_fc;
    auto block_size_fc = expert_attributes_[kFC2Idx].block_size_fc;
    std::vector<size_t> a_shape = {
      kBatchSize, kernel_size, static_cast<size_t>(K_fc)
    };
    std::vector<size_t> b_shape = {
      static_cast<size_t>(K_fc), static_cast<size_t>(N_fc)
    };

    std::vector<size_t> c_shape = {
      kBatchSize, kernel_size, static_cast<size_t>(N_fc)
    };

    NPUTensor input_tensor = {nullptr, a_shape, "bfloat16"};
    NPUTensor wts_tensor = {nullptr, b_shape, "bfloat16"};
    NPUTensor out_tensor = {nullptr, c_shape, "bfloat16"};
    NPUTensor placeholder;
    std::vector<NPUTensor> inputs = {input_tensor, wts_tensor,  placeholder,
                                     placeholder,  placeholder, out_tensor};
    std::vector<NPUTensor> outputs;
    std::map<std::string, std::any> attrs;
    std::vector<int> grp_size = {static_cast<int>(block_size_fc)};
    attrs["group_size"] = grp_size;
    attrs["mem_opt"] = 1;
    std::vector<OpArgMap> arg_map =
      ss_->down_gemm_->get_buffer_reqs(inputs, outputs, attrs);
    auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());

    // a_bo -> gate_up -> c_bo -> general_activation -> a_bo -> down -> c_bo
    a_bo_size = std::max(a_bo_size, size_map["in0"]);
    c_bo_size = std::max(c_bo_size, size_map["out"]);

    down_out_size = size_map["out"];
    if (mladfVersion() == MladfVersion::v2) {
      size_t scratch_bo_size = size_map["scratch"];
      scratch_size = std::max(scratch_size, alignTo4096(scratch_bo_size));
    }
  }

  auto a_size = alignTo4096(a_bo_size);
  auto c_size = alignTo4096(c_bo_size);

  /*
  buffer_size_ = 0;

  auto gate_up_in_offset = buffer_size_;
  update_buffer_map("gate_up_in", buffer_size_, a_size);
  buffer_size_ += a_size;

  update_buffer_map("gate_up_out", buffer_size_, c_size);
  update_buffer_map("general_act_in", buffer_size_, c_size);
  update_buffer_map("general_act_out", gate_up_in_offset, a_size);
  update_buffer_map("down_in", gate_up_in_offset, a_size);
  // use actual size here, since we can sync on this
  update_buffer_map("down_out", buffer_size_, down_out_size);
  buffer_size_ += c_size;

  update_buffer_map("scratch", buffer_size_, scratch_size);
  buffer_size_ += scratch_size;
  */

  SharedBuffer::Requirements shared_buffer_reqs = {
    {"gate_up_in+general_act_out+down_in", a_size},
    {"gate_up_out+general_act_in+down_out", c_size},
    {"scratch", scratch_size}
  };

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}

static void execute_batched_matmul(
  xrt::bo& input_bo, xrt::bo& output_bo,
  std::unordered_map<size_t, xrt::bo>& const_bo_map,
  const std::map<size_t, size_t>& expert_to_seq_ids,
  const std::map<size_t, size_t>& expert_to_offset,
  std::unique_ptr<
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>& gemm,
  const int64_t mat_k, const int64_t mat_n, const int64_t block_size
) {
  const auto num_experts_sched = expert_to_seq_ids.size();
  size_t expert_idx = 0;

  for (const auto& [expert_id, num_tokens] : expert_to_seq_ids) {
    ++expert_idx;

    // right now only sync on last matmul
    // TODO: can delay if general_activation can be offloaded to NPU
    //       and without cpu needing to reformat
    //       alternatively can overlap mapping/binding of FC2 with NPU
    //       exectution with dynamic loading of experts
    bool sync = (num_experts_sched == expert_idx);

    const size_t m_offset = expert_to_offset.at(expert_id);
    const size_t input_bo_offset = m_offset * mat_k * sizeof(std::uint16_t);
    const size_t output_bo_offset = m_offset * mat_n * sizeof(std::uint16_t);

    std::vector<size_t> a_shape = {
      static_cast<size_t>(num_tokens), static_cast<size_t>(mat_k)
    };

    // std::vector<size_t> c_shape = {static_cast<size_t>(num_tokens),
    //                                static_cast<size_t>(mat_n)};
    std::vector<size_t> wts_shape_dd = {
      static_cast<size_t>(mat_k), static_cast<size_t>(mat_n)
    };

    gemm->set_shape(a_shape, wts_shape_dd, block_size);

    std::vector<xrt::bo> input_bos = {};
    std::vector<uint64_t> input_addrs = {
      input_bo.address() + input_bo_offset, const_bo_map.at(expert_id).address()
    };
    std::vector<xrt::bo> output_bos = {};
    std::vector<uint64_t> output_addrs = {
      output_bo.address() + output_bo_offset
    };

    gemm->execute(input_bos, input_addrs, output_bos, output_addrs, sync);
  }
}

// Kernel Compute
void AMDQMoEKernel::Compute(OrtKernelContext* context) {
  // std::cout << "---------------Starting compute---------------\n";
#ifdef NPU_QMOE_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif
  // Get ORT Kernel Context
  Ort::KernelContext ctx(context);

  manageDynamicDpmState();

  initializeKernels();

  const auto input_num = ctx.GetInputCount();

  auto input_act = ctx.GetInput(0);
  // e.g [batch_size = 1, seq_len, hidden_dim]
  auto input_act_dims = input_act.GetTensorTypeAndShapeInfo().GetShape();
  auto hidden_dim = input_act_dims[2];
  auto input_act_data = input_act.GetTensorData<std::uint16_t>();
  const std::uint16_t* input_act_data_ptr =
    static_cast<const std::uint16_t*>(input_act_data);

  auto router = ctx.GetInput(1);
  // e.g. [batch*seq_len, num_experts = 32]
  auto router_dims = router.GetTensorTypeAndShapeInfo().GetShape();
  auto router_data = router.GetTensorData<std::uint16_t>();
  const std::uint16_t* router_data_ptr =
    static_cast<const std::uint16_t*>(router_data);

  size_t m = router_dims[0];

  // step 1 - for each token find topk = 4 experts and note their indices
  std::vector<top_k_info> top_k_experts(m * top_k_);

  top_k_payload payload = {
    router_data_ptr, top_k_experts.data(), top_k_, num_experts_
  };

  ctx.ParallelFor(find_top_k, static_cast<size_t>(m), 0, &payload);

  // step 2 - normalize weights: calculate softmax on top k
  if (normalize_routing_weights_) {
    normalize_payload payload_norm = {top_k_experts.data(), top_k_};
    ctx.ParallelFor(
      normalize_weights, static_cast<size_t>(m), 0, &payload_norm
    );
  }

  // step 3 - group tokens that go to same expert so we can batch
  //          create map of expert_idx -> count{seq_indices}
  //          for now implement serial version
  std::map<size_t, size_t> expert_to_seq_ids;

  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < top_k_; j++) {
      const auto expert_index = top_k_experts.at(i * top_k_ + j).index;
      ++expert_to_seq_ids[expert_index];
    }
  }

  // keep ordered to run experts in sequence
  std::map<size_t, size_t> expert_to_offset;
  std::unordered_set<size_t> sched_expert_ids;
  size_t curr_offset = 0;

  for (const auto& [expert_id, num_tokens] : expert_to_seq_ids) {
    expert_to_offset[expert_id] = curr_offset;
    curr_offset += num_tokens;
    sched_expert_ids.insert(expert_id);
  }

  std::map<size_t, size_t> expert_to_offset_temp = expert_to_offset;

  std::vector<size_t> token_to_idx(m * top_k_);

  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < top_k_; j++) {
      const auto expert_index = top_k_experts.at(i * top_k_ + j).index;
      token_to_idx.at(i * top_k_ + j) =
        expert_to_offset_temp.at(expert_index)++;
    }
  }

  // step 4 - gather inputs to kernel buffers using expert_to_seq_ids
  //          want to avoid alignment for running each expert
  //          allocate space for M * top_k_
  //          this also means there will be some over-compute

  // step 4a - allocate NPU kernel buffers
  size_t curr_seq_len = 0;
  size_t alloc_seq_len = 0;  // envelope of packed input for experts

  for (const auto& [expert_id, num_tokens] : expert_to_seq_ids) {
    size_t npu_kernel_size =
      getNPUKernelGranularity(num_tokens, granularity_options);

    alloc_seq_len = std::max(alloc_seq_len, curr_seq_len + npu_kernel_size);
    curr_seq_len += num_tokens;
  }

  // state variables used for debug
  seq_len_ = m;
  num_active_experts_ = expert_to_seq_ids.size();

  UpdateSharedBuffer(alloc_seq_len);

  // buffer layout will be
  //[0, in_size), [in_size, in_size + out_size)
  // the check for rebinding should be
  // curr_in_size < in_size  || curr_out_size < out_size

  if (auto res = shared_buffer_.Validate(
        "gate_up_in+general_act_out+down_in",
        ss_->gate_up_gemm_->get_inputs(kGemmBOsSelector)[0]
      )) {
    ss_->gate_up_gemm_->create_bo(res->ptr, res->len, 0, kGemmBOsSelector);
  }

  if (auto res = shared_buffer_.Validate(
        "gate_up_out+general_act_in+down_out",
        ss_->gate_up_gemm_->get_outputs(kGemmBOsSelector)[0]
      )) {
    ss_->gate_up_gemm_->create_bo(res->ptr, res->len, 1, kGemmBOsSelector);
  }

  if (mladfVersion() == MladfVersion::v2) {
    if (auto res = shared_buffer_.Validate(
          "scratch", ss_->gemm_last_scratch_ptr_, ss_->gemm_last_scratch_len_
        )) {
      // NOTE: current matmul execute will use internal BO for scratch
      //       to reduce memory we will bind to host buffer
      //       need to do this for both gate_up and down matmuls
      ss_->gate_up_gemm_->create_bo(res->ptr, res->len, 2, kGemmBOsSelector);
      ss_->down_gemm_->create_bo(res->ptr, res->len, 2, kGemmBOsSelector);

      ss_->gemm_last_scratch_ptr_ = res->ptr;
      ss_->gemm_last_scratch_len_ = res->len;
    }
  }

  auto gate_input_bo = ss_->gate_up_gemm_->get_inputs(kGemmBOsSelector)[0];
  auto gate_output_bo = ss_->gate_up_gemm_->get_outputs(kGemmBOsSelector)[0];

  std::uint16_t* input_gate_up_ptr = gate_input_bo.map<std::uint16_t*>();

  gather_payload payload_gather = {
    token_to_idx.data(),
    input_act_data_ptr,
    input_gate_up_ptr,
    hidden_dim,
    top_k_,
    m,
    sizeof(std::uint16_t)
  };

  ctx.ParallelFor(
    gather_expert_tokens, static_cast<size_t>(token_to_idx.size()), 0,
    &payload_gather
  );

  // step 5 - run gate_up matmul - plan for NPU offload
  std::uint16_t* output_gate_up_ptr = gate_output_bo.map<std::uint16_t*>();

  const size_t gate_up_in_buf_size =
    m * top_k_ * hidden_dim * sizeof(std::uint16_t);
  gate_input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, gate_up_in_buf_size, 0);

  const auto& load_expert_ids =
    hybrid_opt_qmoe_dynamic_experts_ ? sched_expert_ids : default_expert_ids_;

  unbind_experts(num_experts_, load_expert_ids, gate_up_experts_);
  unbind_experts(num_experts_, load_expert_ids, down_experts_);

#ifdef _WIN32
  unmap_experts(load_expert_ids, pBufs_, packed_fc_ptrs_, kFC1Idx);
  unmap_experts(load_expert_ids, pBufs_, packed_fc_ptrs_, kFC2Idx);

  map_experts(
    num_fc_, num_experts_, load_expert_ids, external_fc_info_[kFC1Idx].first,
    expert_attributes_[kFC1Idx].packed_expert_sz_fc, gran_, hMapping_, hFile_,
    kFC1Idx, pBufs_, packed_fc_ptrs_
  );
#else
  abort();
#endif

  bind_experts(
    num_experts_, load_expert_ids, gate_up_experts_, packed_fc_ptrs_, kFC1Idx,
    expert_attributes_[kFC1Idx].packed_expert_sz_fc, ss_->gate_up_gemm_
  );

  execute_batched_matmul(
    gate_input_bo, gate_output_bo, gate_up_experts_, expert_to_seq_ids,
    expert_to_offset, ss_->gate_up_gemm_, expert_attributes_[kFC1Idx].K_fc,
    expert_attributes_[kFC1Idx].N_fc, expert_attributes_[kFC1Idx].block_size_fc
  );

  // TODO: can remove once general_activation on npu
  const size_t gate_up_out_buf_size =
    m * top_k_ * expert_attributes_[kFC1Idx].N_fc * sizeof(std::uint16_t);
  gate_output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, gate_up_out_buf_size, 0);

  if (1 < hybrid_opt_qmoe_dynamic_experts_) {
    constexpr bool kGateUp = true;
    clearExperts(kGateUp);
#ifdef _WIN32
    unmap_experts({}, pBufs_, packed_fc_ptrs_, kFC1Idx);
#endif
  }

  // step 6 - run general_activation function - plan for NPU offload
  //          will have column interleaved input that will apply
  //
  std::uint16_t* output_swiglu_ptr = input_gate_up_ptr;
  swiglu_payload payload_activation = {
    output_gate_up_ptr,
    output_swiglu_ptr,
    2 * hidden_dim,
    top_k_,
    act_attributes_.activation_alpha,
    act_attributes_.activation_beta,
    act_attributes_.swiglu_limit,
    act_attributes_.activation_type,
    act_attributes_.swiglu_fusion
  };
  ctx.ParallelFor(
    swiglu, static_cast<size_t>(m * top_k_), 0, &payload_activation
  );

  // step 7 - run down matmul - plan for NPU offload
#ifdef _WIN32
  map_experts(
    num_fc_, num_experts_, load_expert_ids, external_fc_info_[kFC2Idx].first,
    expert_attributes_[kFC2Idx].packed_expert_sz_fc, gran_, hMapping_, hFile_,
    kFC2Idx, pBufs_, packed_fc_ptrs_
  );
#endif
  bind_experts(
    num_experts_, load_expert_ids, down_experts_, packed_fc_ptrs_, kFC2Idx,
    expert_attributes_[kFC2Idx].packed_expert_sz_fc, ss_->down_gemm_
  );

  std::uint16_t* output_down_ptr = output_gate_up_ptr;

  // TODO: can remove once general_activation on npu
  const size_t down_in_buf_size =
    m * top_k_ * expert_attributes_[kFC2Idx].K_fc * sizeof(std::uint16_t);
  gate_input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, down_in_buf_size, 0);

  execute_batched_matmul(
    gate_input_bo, gate_output_bo, down_experts_, expert_to_seq_ids,
    expert_to_offset, ss_->down_gemm_, expert_attributes_[kFC2Idx].K_fc,
    expert_attributes_[kFC2Idx].N_fc, expert_attributes_[kFC2Idx].block_size_fc
  );

  // need this with scatter on CPU
  const size_t down_out_buf_size =
    m * top_k_ * expert_attributes_[kFC2Idx].N_fc * sizeof(std::uint16_t);
  gate_output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, down_out_buf_size, 0);

  if (1 < hybrid_opt_qmoe_dynamic_experts_) {
    constexpr bool kDown = false;
    clearExperts(kDown);
#ifdef _WIN32
    unmap_experts({}, pBufs_, packed_fc_ptrs_, kFC2Idx);
#endif
  }

  // step 8 - scatter outputs back to create top_k x M x hidden_dim input for
  //          weighted sum
  std::uint16_t* output_scatter_ptr = input_gate_up_ptr;

  gather_payload payload_scatter = {
    token_to_idx.data(),  output_down_ptr, output_scatter_ptr,
    hidden_dim,           top_k_,          m,
    sizeof(std::uint16_t)
  };

  ctx.ParallelFor(
    scatter_expert_tokens, static_cast<size_t>(token_to_idx.size()), 0,
    &payload_scatter
  );

  // step 9 - run weighted sum with second input being M x top_k weights to
  //          combine experts
  //          output should be M x N - plan for NPU offload
  //          do in place
  std::uint16_t* output_weighted_sum_ptr = output_scatter_ptr;
  weighted_sum_payload payload_ws = {
    output_scatter_ptr,
    top_k_experts.data(),
    output_weighted_sum_ptr,
    hidden_dim,
    top_k_,
    m
  };
  ctx.ParallelFor(weighted_sum, static_cast<size_t>(m), 0, &payload_ws);

  // NOTE: this was added since random output was observed when other custom ops
  //       would use the same shared memory
  const size_t scatter_out_buf_size =
    m * top_k_ * hidden_dim * sizeof(std::uint16_t);
  gate_input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, scatter_out_buf_size, 0);

  // step 10 - copy to ORT output buffer
  // Output activation - single output and same dims as input activation
  auto output_tensor = ctx.GetOutput(0, input_act_dims);
  auto out = output_tensor.GetTensorMutableData<uint16_t>();
  auto output_count =
    output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  uint16_t* out_data_ptr = (uint16_t*)out;

  memcpy(
    out_data_ptr, output_weighted_sum_ptr,
    m * hidden_dim * sizeof(std::uint16_t)
  );

#ifdef NPU_QMOE_PROFILE_EN
  const Clock::time_point execute_end = Clock::now();
#endif

  if (freeAfterPrefill(name())) {
    if (lastNode(name()) && getPrefillBufferRelease(alloc_seq_len)) {
      shared_buffer_.Reset();
    }
  }
}

AMDQMoEKernel::~AMDQMoEKernel() {
  --ss_->instances_;

#ifdef _WIN32
  if (mapped_fc_) {
    for (auto& [fc_expert_idx, p_buf] : pBufs_) {
      UnmapViewOfFile(p_buf);
    }

    CloseHandle(hMapping_);
    CloseHandle(hFile_);

    mapped_fc_ = false;
  }
#endif  // _WIN32

#ifdef NPU_QMOE_PROFILE_EN
  std::ostringstream os;

  int event_id = 0;

  for (const auto& measurement : measurements_) {
    os << name() << ",NPUQmoe," << event_id << ","
       << MillisecondsFp{measurement}.count() << "\n";
    event_id++;
  }

  std::cout << os.str() << std::flush;
#endif
}

}  // namespace ryzenai::onnx_utils
