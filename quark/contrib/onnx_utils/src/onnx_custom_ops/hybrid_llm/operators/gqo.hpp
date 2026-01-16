// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_GQO
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_GQO

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

// #define ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
// #define ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

class AMDGQOKernel;
// forward-declaring these classes gave linker errors for some reason
// class OnnxTensorInfo;
// namespace DML_Ops {
//   class DMLOps;
// } // namespace DML_Ops

class GQOKernel : public HybridKernel {
 public:
  GQOKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~GQOKernel();

  void Compute(OrtKernelContext* context);
  void CreateSharedKVCache(size_t resourceSize);
  std::pair<void*, void*> GetCpuMappedMemSharedKVCache() const;
  std::pair<void*, void*> GetD3DResourceSharedKVCache() const;
  bool RebuildSharedKVCache(const Ort::KernelContext& ctx);

 private:
  bool isKVShapeChanged_ = false;
  bool isFirstRun_ = true;

  std::vector<std::pair<void*, size_t>> mapped_output_tensors_;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  RyzenMM::BufferRef shared_k_cache_;
  RyzenMM::BufferRef shared_v_cache_;
#endif

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  // added here since current GQO doesnt follow regular donwload flow
  Duration download_duration_{};
#endif

  RyzenMM::GPUAllocator<'GQO1'> gpuAllocator_;
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDGQOKernel> npu_instance_;
#endif
};

static const char kGQO[] = "GQO";

struct GQO : HybridOperator<GQOKernel, kGQO> {
  explicit GQO(const Ort::ConstSessionOptions& session_options)
    : HybridOperator(session_options, GetSessionConfigKeys()) {}

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    switch (index) {
      case 0:   // query
      case 3:   // past key
      case 4:   // past value
      case 5:   // total_seq_len
      case 6:   // seq_len_k
      case 7:   // cos_cache
      case 8:   // sin_cache
      case 12:  // matmul weights
      case 13:  // matmul scales
      case 14:  // matmul zeros
      case 15:  // matmul bias
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
      case 1:   // key
      case 2:   // value
      case 9:   // position_ids
      case 10:  // attention_bias
      case 11:  // head_sink
      case 16:  // matmul packed
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
      default:
        throw std::out_of_range("GQO operator has up to 14 inputs.");
    }
  }

  virtual size_t GetInputTypeCount() const noexcept { return 17; }

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {
      "hybrid_dbg_use_aie_rope",  "hybrid_dbg_use_aie_gqa",
      "hybrid_dbg_use_flash_mha", "hybrid_opt_execution_mode",
      "hybrid_opt_chunk_context", "hybrid_opt_chunk_context_threshold",
    };
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS
