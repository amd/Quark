// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_OPERATOR
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_OPERATOR

#include <string>
#include <unordered_set>

#include "operator.hpp"

namespace ryzenai::onnx_utils {

template <typename Kernel, const char* Name>
class HybridOperator : public Operator<Kernel, Name> {
 public:
  HybridOperator(
    const Ort::ConstSessionOptions& session_options,
    std::unordered_set<std::string> session_config_keys
  )
    : Operator<Kernel, Name>(
        session_options, Merged(
                           {"custom_allocator",
                            "external_data_file",
                            "hybrid_opt_free_after_prefill",
                            "hybrid_opt_gpu_jit",
                            "hybrid_opt_dynamic_jit_factor",
                            "external_data_blob",
                            "external_data_blob_size",
                            "hybrid_opt_enable_dynamic_dpm",
                            "hybrid_opt_continue_on_exception",
                            "hybrid_opt_npu_read_ahead",
                            "hybrid_opt_token_backend",
                            "hybrid_opt_max_seq_length",
                            "hybrid_opt_enable_npu_preemption",
                            "hybrid_opt_enable_npu_qos",
                            "hybrid_opt_disable_npu_ops",
                            "hybrid_opt_npu_pdi_name",
                            "hybrid_opt_enable_npu_lora",
                            "hybrid_opt_init_prompt_size",
                            "hybrid_opt_qmoe_dynamic_experts",
                            "hybrid_opt_qmoe_num_dynamic_layers"},
                           std::move(session_config_keys)
                         )
      ) {}

 private:
  static std::unordered_set<std::string> Merged(
    std::unordered_set<std::string> a, std::unordered_set<std::string> b
  ) {
    a.merge(std::move(b));
    return a;
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_OPERATOR
