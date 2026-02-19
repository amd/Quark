// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "external_buffers.hpp"

#include <ryzenai/ryzen_mm.h>

#include <future>
#include <sstream>

#include "external_data.hpp"
#include "lora.hpp"
#include "onnx.hpp"
#include "ops/ops_common/dtype_utils.h"

namespace {
std::vector<OpsFusion::Metadata::TensorInfo> get_external_tensors(
  const OpsFusion::Metadata& meta
) {
  std::vector<OpsFusion::Metadata::TensorInfo> external_buffers;

  std::array<std::string, 3> ext_buf_names = {
    "ext_buf_0", "ext_buf_1", "ext_buf_2"
  };

  for (const auto& ext_buf_name : ext_buf_names) {
    if (meta.fused_tensors.find(ext_buf_name) == meta.fused_tensors.end()) {
      continue;
    }
    external_buffers.push_back(meta.fused_tensors.at(ext_buf_name));
  }

  return external_buffers;
}

void fillSkipExtBufCopy(
  const std::string& session_option, std::vector<bool>& skip_ext_buf_copy
) {
  std::string token;
  std::istringstream tokenStream(session_option);

  while (std::getline(tokenStream, token, ',')) {
    int value = std::stoi(token);
    if (value < skip_ext_buf_copy.size()) {
      skip_ext_buf_copy[value] = true;
    }
  }
}
}  // namespace

namespace ryzenai::onnx_utils {

void ExternalBuffers::construct(
  const Ort::ConstKernelInfo& info, const OpsFusion::Metadata& meta,
  ModelType model_type, const std::string& skip_ext_buf_copy,
  bool legacy_llm_prefill, bool io_bind_kv_cache
) {
  auto external_tensors = get_external_tensors(meta);
  model_type_ = model_type;
  // for fusion prefill, assumption is KV cache is flattened from OGA/ORT
  // and we just need to query underlying bo;
  bool is_llm_prefill_model =
    model_type_ == ModelType::Llm_Prefill || legacy_llm_prefill;
  bool bind_kv_cache = io_bind_kv_cache;
  skip_ext_buf_copy_.resize(external_tensors.size(), bind_kv_cache);
  fillSkipExtBufCopy(skip_ext_buf_copy, skip_ext_buf_copy_);

  auto tensor_index = 0;
  for (const auto& tensor_info : external_tensors) {
    if (skip_ext_buf_copy_[tensor_index] &&
        (kKvCacheTensorIndex == tensor_index)) {
      external_buffers_.emplace_back(nullptr, tensor_info.size);
    } else {
#ifdef _WIN32
      auto* buffer = _aligned_malloc(tensor_info.size, 4096);
#else
      auto* buffer = aligned_alloc(4096, tensor_info.size);
#endif
      external_buffers_.emplace_back(buffer, tensor_info.size);
      // memset(external_buffers_.back().first, 0, tensor_info.size);
    }
    for (const auto& tensor_name : tensor_info.packed_tensors) {
      auto offset = meta.tensor_map.at(tensor_name).offset;
      external_buffer_addresses_.try_emplace(
        tensor_name, external_buffers_.size() - 1, offset
      );

      // assume tensor_index 0 corresponds to KV cache
      if (bind_kv_cache && (kKvCacheTensorIndex == tensor_index) &&
          (0 == offset)) {
        kv_cache_base_tensor_name_ = tensor_name;
      } else if ((tensor_info.packed_tensors.size() > 1) &&
                 (kLoraBufIndex_ == tensor_index) &&
                 (tensor_name.find("lora") != std::string::npos) &&
                 (lora_ext_buf_size_ == 0)) {
        lora_ext_buf_offset_ = meta.tensor_map.at(tensor_name).offset;
        lora_ext_buf_size_ = tensor_info.size - lora_ext_buf_offset_;
      }
    }
    tensor_index++;
  }

  size_t input_num;
  try {
    input_num = info.GetAttribute<int64_t>("input_num");
  } catch (const Ort::Exception&) {
    input_num =
      legacy_llm_prefill ? info.GetInputCount() - 1 : info.GetInputCount();
  }
  for (int i = 0U; i < input_num; ++i) {
    const auto name = info.GetInputName(i);
    if (external_buffer_addresses_.count(name) == 0) {
      input_indices_.push_back(i);
    } else {
      external_input_indices_.try_emplace(i, name);
    }

    if (name.find("attention_mask_const_uint") != std::string::npos) {
      attention_mask_index_ = i;
    }
  }

  const auto output_num = info.GetOutputCount();
  for (int i = 0U; i < output_num; ++i) {
    const auto name = info.GetOutputName(i);
    if (external_buffer_addresses_.count(name) == 0) {
      output_indices_.push_back(i);
    } else {
      external_output_indices_.try_emplace(i, name);
    }
  }
}

void ExternalBuffers::zeroesAllLoraData() const {
  memset(
    (int8_t*)(external_buffers_.at(kLoraBufIndex_).first) +
      lora_ext_buf_offset_,
    0, lora_ext_buf_size_
  );
}

void ExternalBuffers::loadLoraBin(
  const std::string tensor_name, int bin_offset, int bin_size,
  const OpsFusion::Metadata& meta
) {
  if (bin_size != meta.tensor_map.at(tensor_name).size_in_bytes) {
    throw std::runtime_error(
      "Invalid bin size for loading LoRA data into external buffer for "
      "tensor: " +
      tensor_name
    );
  }
  Lora::loadBinData(
    (int8_t*)(external_buffers_.at(kLoraBufIndex_).first) +
      meta.tensor_map.at(tensor_name).offset,
    bin_size, bin_offset,
    model_type_ == ModelType::Llm_Prefill ? LoraDataType::Prefill
                                          : LoraDataType::Token
  );
}

void ExternalBuffers::initialize(
  const Ort::KernelContext& ctx,
  const std::vector<std::vector<int64_t>>& input_shapes,
  const std::vector<std::vector<int64_t>>& output_shapes
) {
  for (const auto& [index, name] : external_input_indices_) {
    if (name.find("lora") != std::string::npos) continue;

    auto external_tensor = getInputTensor(ctx, index, input_shapes.at(index));
    const auto [tensor_index, offset] = external_buffer_addresses_.at(name);

    auto size = std::accumulate(
                  input_shapes.at(index).begin(), input_shapes.at(index).end(),
                  1ULL, std::multiplies<>()
                ) *
                Utils::get_size_of_type(external_tensor.dtype);
    if (skip_ext_buf_copy_[tensor_index] &&
        (kKvCacheTensorIndex == tensor_index)) {
      uintptr_t ptr_val;
      std::memcpy(&ptr_val, external_tensor.data, sizeof(uintptr_t));
      external_buffers_[tensor_index].first = reinterpret_cast<void*>(ptr_val);
    } else {
      std::memcpy(
        static_cast<std::byte*>(external_buffers_[tensor_index].first) + offset,
        external_tensor.data, size
      );
    }
  }

  // initialize the KV cache pointer from the outputs because LLM Prefill does
  // not have KV cache inputs, only outputs. Assuming past and present caches
  // are shared in OGA, this is not a problem. kv_cache_base_tensor_name_ is
  // already gated by checking it's an LLM prefill model from initialize()
  for (const auto& [index, name] : external_output_indices_) {
    if (name == kv_cache_base_tensor_name_) {
      auto external_tensor =
        getOutputTensor(ctx, index, output_shapes.at(index));
      const auto [external_buffer_index, packed_tensor_offset] =
        external_buffer_addresses_.at(name);

      external_buffers_.at(external_buffer_index).first = external_tensor.data;
    }
  }
}

std::vector<size_t> ExternalBuffers::getInputIndices() const {
  return input_indices_;
}

std::vector<size_t> ExternalBuffers::getOutputIndices() const {
  return output_indices_;
}

std::vector<size_t> ExternalBuffers::getExternalOutputIndices() const {
  std::vector<size_t> indices;
  for (const auto& [key, _] : external_output_indices_) {
    indices.push_back(key);
  }
  return indices;
}

std::vector<::Tensor> ExternalBuffers::getExternalBuffers(
  bool include_kv_cache
) const {
  std::vector<::Tensor> external_bufs;
  for (auto i = 0U; i < external_buffers_.size(); i++) {
    const auto& [ptr, size] = external_buffers_[i];
    if ((!include_kv_cache) && (i == kKvCacheTensorIndex)) {
      continue;
    }
    external_bufs.push_back(::Tensor{ptr, {size}, "uint8"});
  }
  return external_bufs;
}

std::vector<xrt::bo> ExternalBuffers::getKvCacheBO() const {
  std::vector<xrt::bo> external_bufs_bo;
  external_bufs_bo.push_back(
    ryzenai::RyzenMM::Platform::XRT::GetUnderlyingBO(
      ryzenai::RyzenMM::UnmanagedBufferPtr(
        external_buffers_[kKvCacheTensorIndex].first
      )
    )
  );

  return external_bufs_bo;
}

bool ExternalBuffers::bindKvCache() const {
  return kv_cache_base_tensor_name_ != "";
}

void ExternalBuffers::copyKvCacheToOrtTensors(
  const Ort::KernelContext& ctx,
  const std::vector<std::vector<int64_t>>& output_shapes, int64_t seq_len_arg,
  bool is_prefill
) {
  int thr = 2;
  std::vector<std::future<void>> futures(thr);
  int cnt = 0;

  // interpret seq_len_arg based on is_prefill
  const int64_t seq_len =
    is_prefill ? *(ctx.GetInput(seq_len_arg).GetTensorData<int64_t>()) : 1;
  const int64_t past_seq_len = is_prefill ? 0 : seq_len_arg;

  for (const auto& [index, name] : external_output_indices_) {
    auto external_tensor = getOutputTensor(ctx, index, output_shapes[index]);
    const auto [tensor_index, offset] = external_buffer_addresses_.at(name);
    if (tensor_index != kKvCacheTensorIndex) {
      continue;
    }
    if (skip_ext_buf_copy_[tensor_index] && !bindKvCache()) {
      auto i =
        reinterpret_cast<std::uintptr_t>(external_buffers_[tensor_index].first);
      std::memcpy(external_tensor.data, &i, sizeof(std::uintptr_t));

    } else {
      const auto kv_num_heads = output_shapes[index][1];
      const void* src = external_buffers_[tensor_index].first;
      void* dst = external_tensor.data;
      auto total_size = std::accumulate(
        output_shapes[index].begin(), output_shapes[index].end(), 1ULL,
        std::multiplies<>()
      );
      if (!bindKvCache() && is_prefill) {
        std::memset(dst, 0, total_size);
      }

      for (auto i = 0; i < kv_num_heads; ++i) {
        int slot = cnt % thr;

        // Wait for previous future in the slot if valid
        // if (futures[slot].valid()) {
        //   futures[slot].get();
        // }

        auto size = seq_len * output_shapes[index][3];
        auto new_offset =
          ((i * output_shapes[index][2] + past_seq_len) *
           output_shapes[index][3] * sizeof(uint16_t));

        // Launch async task
        // futures[slot] = std::async(std::launch::async, [src, offset,
        // new_offset, size, dst]() {
        if (is_prefill && (external_tensor.dtype != "bfloat16")) {
          ryzenai::bfloat16_buffer_to_float16(
            (const uint16_t*)(static_cast<const std::byte*>(src) + offset +
                              new_offset),
            size, (uint16_t*)(static_cast<std::byte*>(dst) + new_offset)
          );
        } else {
          memcpy(
            (static_cast<std::byte*>(dst) + new_offset),
            (static_cast<const std::byte*>(src) + offset + new_offset),
            size * sizeof(uint16_t)
          );
        }
        // });

        ++cnt;
      }
    }
  }
  // for (auto& f : futures) {
  //   if (f.valid()) f.get();
  // }
}

size_t ExternalBuffers::getAttentionMaskIndex() const {
  if (attention_mask_index_ == std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(
      "No input with the name 'attention_mask' found!"
    );
  }
  return attention_mask_index_;
}

bool ExternalBuffers::hasExternalBuffers() const {
  return !external_buffers_.empty();
}

void ExternalBuffers::syncForHost(OpsFusion::FusionRuntime* rt) const {
  for (auto i = 0U; i < external_buffers_.size(); ++i) {
    rt->sync_external_tensor_for_host(i);
  }
}

void ExternalBuffers::syncForDevice(OpsFusion::FusionRuntime* rt) const {
  for (auto i = 0U; i < external_buffers_.size(); ++i) {
    rt->sync_external_tensor_for_device(i);
  }
}

void ExternalBuffers::syncLora(void* rt) {
  static_cast<OpsFusion::FusionRuntime*>(rt)->sync_external_tensor_for_device(
    kLoraBufIndex_
  );
}

ExternalBuffers::~ExternalBuffers() {
  auto tensor_index = 0;

  for (auto skip_copy : skip_ext_buf_copy_) {
    if (!skip_copy) {
#ifdef _WIN32
      _aligned_free(external_buffers_.at(tensor_index).first);
#else
      free(external_buffers_.at(tensor_index).first);
#endif
    }

    tensor_index++;
  }
}

}  // namespace ryzenai::onnx_utils
