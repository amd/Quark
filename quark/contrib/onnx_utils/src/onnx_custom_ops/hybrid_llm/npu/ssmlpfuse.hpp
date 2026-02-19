// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once

#include <onnxruntime_cxx_api.h>
#include <ryzenai/ryzen_mm.h>

#include <algorithm>
#include <filesystem>
#include <mutex>

#include "external_data.hpp"
#include "hybrid_llm/ort/cast.hpp"
#include "hybrid_llm/ort/simplified_layer_norm.hpp"
#include "hybrid_llm/ort/skip_simplified_layer_norm.hpp"
#include "jit_node.hpp"
#include "lora_op_interface.hpp"
#include "npu_op.hpp"
#include "npu_utils.hpp"
#include "ops/mladfadd/mladfadd.hpp"
#include "ops/transformer/ssmlp.hpp"
#include "profile.h"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

// #define NPU_MATMULNBITS_PROFILE

#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && \
  defined(NPU_MATMULNBITS_PROFILE)
#define NPU_MATMULNBITS_PROFILE_EN
#endif

struct AMDSSMlpFuseKernel : public JitNode<AMDSSMlpFuseKernel>, public NpuOp {
  AMDSSMlpFuseKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDSSMlpFuseKernel() override;
  void Compute(OrtKernelContext* context);

  void readDataImpl(int idx) override;
  void loadDataImpl(int idx) override;
  void unloadDataImpl(int idx) override {};

  void initializeKernels() final;
  void UpdateSharedBuffer(size_t kernel_size) override final;

  void execute_ssmlp(
    const uint16_t* input_data, std::vector<int64_t>& input_shape,
    const uint16_t* skip_data, std::vector<int64_t>& skip_shape, int grp_size,
    int run_cnt, bool sync_input
  );
  // void UpdateSharedBufferAie4(size_t kernel_size);

  // ORT cast operators
  OrtCast<Ort::BFloat16_t, Ort::Float16_t> ort_cast_bf16_to_fp16_;
  OrtCast<Ort::Float16_t, Ort::BFloat16_t> ort_cast_fp16_to_bf16_;
  OrtCast<float, Ort::Float16_t> ort_cast_fp32_to_fp16_;

  std::vector<Ort::BFloat16_t> npu_skip_buffer_;

  const char* op_type_ = "SSMLP_FUSE";

#ifdef NPU_SS_MLP_PROFILE_EN
  std::vector<Duration> measurements_;
#endif

 protected:
  // Instance members
  int cnt_;

  // Cast indices
  std::vector<int64_t> input_cast_indices_;
  std::vector<int64_t> output_cast_indices_;

  // fused ssmlp
  int64_t ssmlp_fused_ = 0;
  size_t ssmlp_seq_len_ = 0;
  size_t ssmlp_K_ = 0;
  size_t ssmlp_N_ = 0;
  ExternalTensorInfo jit_tensor_ssmlp_;
  size_t ssmlp_scratch_bo_size_ = 0;

  struct State {
    int instances__ = 0;
    int seq_len_ = 0;
    int run_instances__ = 0;
    std::once_flag initFlag;
    std::unique_ptr<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>> add_;
    // ssmlp for aie4
    std::unique_ptr<ryzenai::dynamic_dispatch::transformer::ssmlp<
      uint16_t, uint8_t, uint16_t>>
      ssmlp_{nullptr};
    int64_t jit_max_bo_size_ssmlp_ = 0;
    std::vector<RyzenMM::BufferRef> ssmlp_bo_data_{};
    xrt::bo ssmlp_inputs_;
    std::vector<xrt::bo> ssmlp_weights_;
    xrt::bo ssmlp_outputs_;
    xrt::bo ssmlp_scratch_;
  };

  SessionState<State> ss_;
};

extern template class JitNode<AMDSSMlpFuseKernel>;

}  // namespace ryzenai::onnx_utils
