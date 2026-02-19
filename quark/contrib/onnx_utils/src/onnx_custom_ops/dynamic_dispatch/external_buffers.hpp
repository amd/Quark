// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_EXTERNAL_BUFFERS
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_EXTERNAL_BUFFERS

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "../lora.hpp"
#include "onnxruntime_cxx_api.h"
#include "op_fuser/fusion_rt.hpp"

namespace ryzenai::onnx_utils {

// this should be kept in sync with the Python version in passes/dd/__init__.py
enum class ModelType {
  Unet,
  Decoder,
  Unet_Bfp,
  Decoder_Bfp,
  Sd15_Unet,
  Sd15_Decoder,
  Sd30_Mmdit,
  Sd30_VAE,
  Llm_Prefill,
  Llm_Token,
  Unknown
};

class ExternalBuffers : public LoraInterface {
 public:
  ~ExternalBuffers();
  void construct(
    const Ort::ConstKernelInfo& info, const OpsFusion::Metadata& meta,
    ModelType model_type, const std::string& skip_ext_buf_copy,
    bool legacy_llm_prefill, bool io_bind_kv_cache
  );
  void initialize(
    const Ort::KernelContext& ctx,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes
  );

  /**
   * @brief Copy the KV Cache from the external buffers to the ORT tensors.
   * For prefill fusion, we have to copy the KV cache to the ORT buffers so
   * token phase can use them. For token fusion, we need to copy in case the
   * context cache is being used.
   *
   * The seq_len_arg is either the seq_len_index (for prefill) or the
   * past_seq_len (for token). In prefill phase, the seq_len is an argument to
   * the custom op so we need the index to read from and past_seq_len is 0. In
   * token phase, the seq_len is always 1 and past_seq_len is needed.
   *
   * @param ctx ORT context
   * @param output_shapes the output shapes of the ORT tensors
   * @param seq_len_arg seq_len_index for prefill, past_seq_len for token
   * @param is_prefill true if prefill phase, false if token phase
   */
  void copyKvCacheToOrtTensors(
    const Ort::KernelContext& ctx,
    const std::vector<std::vector<int64_t>>& output_shapes, int64_t seq_len_arg,
    bool is_prefill
  );
  bool hasExternalBuffers() const;

  std::vector<size_t> getInputIndices() const;
  std::vector<size_t> getOutputIndices() const;
  std::vector<size_t> getExternalOutputIndices() const;
  std::vector<::Tensor> getExternalBuffers(bool include_kv_cache) const;
  std::vector<xrt::bo> getKvCacheBO() const;
  bool bindKvCache() const;
  size_t getAttentionMaskIndex() const;

  void syncForHost(OpsFusion::FusionRuntime* rt) const;
  void syncForDevice(OpsFusion::FusionRuntime* rt) const;

  void zeroesAllLoraData() const;
  void loadLoraBin(
    std::string tensor_name, int bin_offset, int bin_size,
    const OpsFusion::Metadata& meta
  );
  void syncLora(void* rt);

 private:
  std::vector<std::pair<void*, size_t>> external_buffers_;
  // map of the external tensor name to the <external buffer index, offset>
  std::unordered_map<std::string, std::pair<size_t, size_t>>
    external_buffer_addresses_;
  // map of the input index to external tensor name
  std::unordered_map<size_t, std::string> external_input_indices_;
  // map of the output index to external tensor name
  std::unordered_map<size_t, std::string> external_output_indices_;
  std::vector<size_t> input_indices_;
  std::vector<size_t> output_indices_;
  std::vector<bool> skip_ext_buf_copy_;
  std::string kv_cache_base_tensor_name_;
  static constexpr size_t kKvCacheTensorIndex = 0;
  static constexpr size_t kLoraBufIndex_ = 1;
  int lora_ext_buf_offset_ = 0;
  int lora_ext_buf_size_ = 0;
  ModelType model_type_;

  size_t attention_mask_index_ = std::numeric_limits<size_t>::max();
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_EXTERNAL_BUFFERS
