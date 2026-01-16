// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "external_data.hpp"
// #include "jit_node.hpp"
#include "npu_op.hpp"
#include "onnxruntime_cxx_api.h"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "profile.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

namespace DML_Ops {
class DMLOps;
}

// #define NPU_QMOE_PROFILE

#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && defined(NPU_QMOE_PROFILE)
#define NPU_QMOE_PROFILE_EN
#endif

struct AMDQMoEKernel : public NpuOp {
  AMDQMoEKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDQMoEKernel() override;
  void Compute(OrtKernelContext* context);

 private:
  void initializeKernels() final;
  void UpdateSharedBuffer(size_t kernel_size) final;
  void clearExperts(bool is_gate_up);

  // standard attributes

  struct ActivationAttributes {
    float activation_alpha = 0;
    float activation_beta = 0;
    int64_t swiglu_fusion = 0;
    float swiglu_limit = 0;
    std::string activation_type;
  };

  ActivationAttributes act_attributes_;

  int64_t expert_weight_bits_ = 0;
  int64_t top_k_ = 0;
  int64_t normalize_routing_weights_ = 0;

  int64_t use_sparse_mixer_ = 0;

  // attributes from post-processing experts
  int64_t num_experts_ = 0;
  int64_t num_fc_ = 0;

  struct QmoeExpertAttributes {
    int64_t K_fc;
    int64_t N_fc;
    int64_t block_size_fc;
    int64_t packed_expert_sz_fc;
  };

  static inline constexpr size_t kFC1Idx = 0;
  static inline constexpr size_t kFC2Idx = 1;
  static inline constexpr size_t kFC3Idx = 2;
  static inline constexpr size_t kMaxFC = 3;

  // TODO: get this from graph info?
  int layer_id = -1;

  QmoeExpertAttributes expert_attributes_[kMaxFC];

  // whether pre-formatted weights have been pulled out
  // layout in memory will be
  //[FC1 expert0| FC1 expert1| FC1 expert2|...] [FC2 expert0| FC2 expert1|...]
  bool use_external_data_ = false;
  std::string external_data_path_;
  std::vector<std::pair<size_t, size_t>> external_fc_info_;
  std::map<std::pair<size_t, size_t>, const std::uint8_t*> packed_fc_ptrs_;

  // file IO
#ifdef _WIN32
  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  HANDLE hMapping_ = INVALID_HANDLE_VALUE;
  DWORD gran_ = 0;
  std::map<std::pair<size_t, size_t>, LPVOID> pBufs_;
#endif

  bool mapped_fc_ = false;

  int hybrid_opt_qmoe_dynamic_experts_ = 0;

  inline static const char* const op_type_ = "QMoEBf";
  inline static const std::vector<size_t> granularity_options = {
    128, 256, 512, 1024, 2048, 3072, 4096
  };

  std::unordered_set<size_t> default_expert_ids_;
  std::unordered_map<size_t, xrt::bo> gate_up_experts_;
  std::unordered_map<size_t, xrt::bo> down_experts_;

  size_t seq_len_ = 0;
  size_t num_active_experts_ = 4;

  struct State {
    int instances_ = 0;
    std::unique_ptr<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>
      gate_up_gemm_{nullptr};
    std::unique_ptr<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>
      down_gemm_{nullptr};

    void* gemm_last_scratch_ptr_{nullptr};
    size_t gemm_last_scratch_len_{0};
  };

  SessionState<State> ss_;
};

}  // namespace ryzenai::onnx_utils
