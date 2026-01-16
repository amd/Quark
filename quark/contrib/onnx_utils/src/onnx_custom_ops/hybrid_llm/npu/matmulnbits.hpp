// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "../session_state.hpp"
#include "external_data.hpp"
#include "hybrid_llm/ort/cast.hpp"
#include "jit_node.hpp"
#include "lora_op_interface.hpp"
#include "npu_op.hpp"
#include "onnxruntime_cxx_api.h"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "profile.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#include "shared_buffer.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

namespace DML_Ops {
class DMLOps;
}

// #define NPU_MATMULNBITS_PROFILE

#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && \
  defined(NPU_MATMULNBITS_PROFILE)
#define NPU_MATMULNBITS_PROFILE_EN
#endif

struct MatMulNBitsAttributes {
  int64_t n = 0;
  int64_t k = 0;
  int64_t bits = 0;
  int64_t block_size = 0;
};

struct AMDMatMulNBitsKernel : public JitNode<AMDMatMulNBitsKernel>,
                              public NpuOp,
                              public LoraOpInterface {
  AMDMatMulNBitsKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDMatMulNBitsKernel() override;
  void Compute(OrtKernelContext* context, bool with_custom_allocator);
  void initialize(
    std::vector<int8_t>& b, std::vector<int8_t>& zeros,
    std::vector<float>& scales, std::vector<float>& bias, std::string shape_l
  );
  void initialize(
    const uint8_t* preformat_consts, size_t size, bool use_shared_buffer = false
  );
  void initializeLora();
  void execute(
    const uint16_t* input_data, std::vector<int64_t> input_shape,
    uint16_t* output_data, std::vector<int64_t> output_shape,
    std::vector<int> wts_shape, int grp_size, int run_cnt, bool input_sync,
    int dst_row_index
  ) const;
  // Currently float32 flow is unsupported
  // need to handle input/output buffer allocation
  // void execute(const float* input_data, float* out,
  //              std::vector<int64_t> input_shape, std::vector<int> wts_shape,
  //              int grp_size, int run_cnt) const;
  void loadDataImpl(int idx) override;
  void readDataImpl(int idx) override;
  void unloadDataImpl(int idx) override {};
  void UpdateSharedBuffer(size_t kernel_size) final;

  void loadLoraData();
  void LoadLora() override;

 private:
  void initializeKernels() final;

  struct State {
    std::unique_ptr<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>
      gemm{nullptr};

    xrt::bo gemm_input;
    xrt::bo gemm_output;
    void* gemm_last_scratch_ptr{nullptr};
    size_t gemm_last_scratch_len{0};
    int64_t jit_max_bo_size{0};
    std::vector<RyzenMM::BufferRef> bo_data{};

    std::vector<int> grp_sizes;
    std::vector<std::vector<int>> n_sizes;

    int instances{0};
    int run_instances{0};

    int seq_len{0};
    int past_n{0};
    int past_k{0};
    int past_grp_size{0};

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
    bool init_cast_op = true;
    size_t vocab_size = 0;
    std::future<void> async_cast_exe{};
#endif
  };

  SessionState<State> ss_;

  LoraBuffer lora_buffers_;

  int cnt;
  int64_t m_N, m_K, m_bits, m_block_size, m_acc_level, npu_k, npu_n;
  bool m_asymmetric, m_biased;
  ExternalTensorInfo jit_tensor_info_;
  bool prune_logits_ = false;

  Ort::ConstValue m_weights, m_scales, m_zeros, m_bias, m_packed_consts;
  Ort::Logger m_logger{nullptr};
  mutable uint16_t* input_data_mladf_ = nullptr;
  mutable uint16_t* output_data_ = nullptr;

  OrtCast<Ort::BFloat16_t, Ort::Float16_t> ort_cast_bf16_to_fp16_;
  OrtCast<Ort::Float16_t, Ort::BFloat16_t> ort_cast_fp16_to_bf16_;

  std::vector<int64_t> input_cast_indices_;
  std::vector<int64_t> output_cast_indices_;
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  // Async Cast initialization.
  std::shared_ptr<DML_Ops::DMLOps> dml_instance_;
  static constexpr bool gpu_cast_en_ = true;
#else
  static constexpr bool gpu_cast_en_ = false;
#endif

#ifdef NPU_MATMULNBITS_PROFILE_EN
  std::vector<Duration> measurements_;
#endif

  RyzenMM::NPUAllocator<'MMNB'> allocator_;
};
extern template class JitNode<AMDMatMulNBitsKernel>;

}  // namespace ryzenai::onnx_utils
