// Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "ops/elwmul/elwmul.hpp"
#include "ops/mladfadd/mladfadd.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "ops/mladfrmsnorm/mladfrmsnorm.hpp"
#include "ops/silu/silu.hpp"
#include "ops/transformer/general_activation.hpp"
#include "profile.h"

#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && \
  defined(NPU_SS_MLP_PROFILE)
#define NPU_SS_MLP_PROFILE_EN
#endif

namespace ryzenai::onnx_utils {

namespace fs = std::filesystem;

/**
 * @brief Base class for SSMLP implementations containing common functionality
 *
 * This class extracts common code between AMDSSMLPKernel and AMDSSGMLPKernel
 * to reduce duplication and improve maintainability.
 */
template <typename T>
class SSMLPBase : public NpuOp, public JitNode<T>, public LoraOpInterface {
  using MatmulKernelPtr = std::shared_ptr<
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>;

 public:
  SSMLPBase(
    JitNode<T>* op_inter, const OrtKernelInfo* k_info,
    const std::unordered_map<std::string, std::string>& session_configs,
    bool is_ssgmlp
  );
  virtual ~SSMLPBase();

  void Compute(OrtKernelContext* context);

  void initializeLora();
  void loadOneLoraData(int idx);

  void readDataImpl(int idx) override;
  void loadDataImpl(int idx) override;
  void unloadDataImpl(int idx) override {};
  void loadLoraData();
  void LoadLora() override;

  void activation_execute(
    size_t gp_M, std::vector<xrt::bo> gate_outputs,
    std::vector<xrt::bo> ewmul_outputs, bool wait
  );
  void get_fused_size(size_t kernel_size);

 protected:
  /**
   * @brief Initialize matmul projection with preformatted constants
   */
  void initialize(
    const uint8_t* preformat_consts, size_t size, int block_size,
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* ptr, int k,
    int n, int bo_size, bool skip_input, bool use_shared_buffer = false
  );

  void initializeKernels() final;

  /**
   * @brief Extract and transpose quantized weights from packed format
   */
  void extractAndTransposeWeights(
    const int8_t* src_weights, std::vector<int8_t>& dst_weights, int64_t k,
    int64_t n, uint8_t type_offset
  );

  /**
   * @brief Transpose scales from NxK format to KxN format
   */
  void transposeScales(
    const Ort::Float16_t* src_scales, std::vector<float>& dst_scales, int n,
    int kblks
  );

  /**
   * @brief Extract zero points from packed format
   */
  void extractZeroPoints(
    const int8_t* src_zps, std::vector<int8_t>& dst_zpoints, int n,
    int kblks_pad, uint8_t type_offset
  );

  /**
   * @brief Initialize projection (gate/up/down) with unpacked constants
   */
  void initializeProjection(
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* ptr,
    const std::vector<int8_t>& weights, const std::vector<float>& scales,
    const std::vector<int8_t>& zpoints, const std::vector<float>& bias, int k,
    int n, int block_size, bool skip_input = true, bool skip_output = true
  );

  /**
   * @brief Calculate buffer sizes for all operations
   */
  void UpdateSharedBuffer(size_t kernel_size) override final;

  // Instance members
  bool sslrn_cpu_out = false;
  int cnt_;
  size_t buffer_size_ = 0;
  bool is_ssgmlp_;

  ExternalTensorInfo jit_tensor_gate_;
  ExternalTensorInfo jit_tensor_up_;
  ExternalTensorInfo jit_tensor_down_;
  std::vector<size_t> supported_lengths{4096, 3072, 2048, 1920, 1792, 1664,
                                        1536, 1408, 1280, 1152, 1024, 768,
                                        640,  512,  384,  256,  128};

  // Projection parameters (gate, up, down)
  int64_t gp_k, gp_n, gp_bits, gp_block_size;
  int64_t up_k, up_n, up_bits, up_block_size;
  int64_t dp_k, dp_n, dp_bits, dp_block_size;

  // Weight data
  size_t num_el, num_el2, num_el3, num_el4;
  size_t num_el_bo, num_el_bo2, num_el_bo3, num_el_bo4;
  bool gate_up_fused_ = false;

  // wts3_data_ is for ssgmlp first additional SLRN
  // wts4_data_ is for ssgmlp second additional SLRN
  std::vector<float> wts_data_, wts2_data_, wts3_data_, wts4_data_;
  std::vector<uint8_t> wts_, wts2_, wts3_, wts4_;
  // m3_weights is for ssgmlp first additional SLRN
  // m4_weights is for ssgmlp second additional SLRN
  Ort::ConstValue m_weights, m2_weights, m3_weights, m4_weights;

  // Epsilon for RMS norm
  float epsilon_;

  // Cast indices
  std::vector<int64_t> input_cast_indices_;
  std::vector<int64_t> output_cast_indices_;

  // ORT cast operators
  OrtCast<Ort::BFloat16_t, Ort::Float16_t> ort_cast_bf16_to_fp16_;
  OrtCast<Ort::Float16_t, Ort::BFloat16_t> ort_cast_fp16_to_bf16_;
  OrtCast<float, Ort::Float16_t> ort_cast_fp32_to_fp16_;

  OrtSimplifiedLayerNorm ort_slrn_;
  OrtSkipSimplifiedLayerNorm ort_sslrn_;

  LoraBuffer lora_buffers_;
  std::vector<Ort::BFloat16_t> npu_skip_buffer_;

  // Allocator
  RyzenMM::NPUAllocator<'NPSS'> allocator_;

  inline static const char* const ssgmlp_op_type_ = "SSGMLP";
  inline static const char* const ssmlp_op_type_ = "SSMLP";
  const char* op_type_;

#ifdef NPU_SS_MLP_PROFILE_EN
  std::vector<Duration> measurements_;
#endif

  struct State {
    int instances__ = 0;
    int seq_len_ = 0;
    int run_instances__ = 0;
    std::shared_ptr<void> ewmul_ = nullptr;
    std::shared_ptr<void> silu_ = nullptr;
    MatmulKernelPtr gate_proj_{nullptr};
    MatmulKernelPtr up_proj_{nullptr};
    MatmulKernelPtr down_proj_{nullptr};
    std::once_flag initFlag;

    void* gemm_last_scratch_ptr_{nullptr};
    size_t gemm_last_scratch_len_{0};

    // RMS norm and add operators
    std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rms_norm_;
    std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rms_norm2_;
    std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rms_norm3_{
      nullptr
    };
    std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rms_norm4_{
      nullptr
    };
    std::unique_ptr<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>> add_;

    int64_t jit_max_bo_size_up_ = 0;
    int64_t jit_max_bo_size_gate_ = 0;
    int64_t jit_max_bo_size_down_ = 0;

    std::vector<RyzenMM::BufferRef> bo_data_gate_{};
    std::vector<RyzenMM::BufferRef> bo_data_up_{};
    std::vector<RyzenMM::BufferRef> bo_data_down_{};
  };

  SessionState<State> ss_;
};

}  // namespace ryzenai::onnx_utils
