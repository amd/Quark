// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_KERNEL
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_KERNEL

#include <optional>
#include <string>
#include <unordered_map>

#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/gpu_utils.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "ort.hpp"

namespace ryzenai::onnx_utils {

namespace DML_Ops {
class DMLOps;
}  // namespace DML_Ops

enum class Backend {
  Cpu,
  Npu,
  Gpu,
};

class HybridKernel {
 public:
  HybridKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

 protected:
  template <typename R>
  R getAttribute(
    const char* name, std::optional<R> default_value = std::nullopt
  ) const {
    return ryzenai::onnx_utils::getAttribute<R>(
      kernel_info_, name, default_value
    );
  }

  template <typename R>
  std::vector<R> getAttributes(
    const char* name, std::optional<std::vector<R>> default_value = std::nullopt
  ) const {
    return ryzenai::onnx_utils::getAttributes<R>(
      kernel_info_, name, default_value
    );
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  void createGpu(
    const std::unordered_map<std::string, std::string>& session_configs,
    size_t gpu_input_num, int op_chaining_input_idx, int op_chaining_output_idx
  );
  void initializeGpu();
  bool computeGpu(const Ort::KernelContext& ctx);
  inline bool usingGpu() const {
    return prefill_backend_ == Backend::Gpu || token_backend_ == Backend::Gpu;
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

  void setTokenBackend(
    const std::unordered_map<std::string, std::string>& session_configs
  );

  void setDisableNpuOps(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  bool disableNpuOps() const;

  const std::string& nodeName() const;
  bool withCustomAllocator() const;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  DML_Ops::DMLOps* dmlInstance() const;
  std::pair<std::vector<OnnxTensorInfo>&, std::vector<OnnxTensorInfo>&>
  gpuTensors();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

  void releaseGpuJitWeights();
  Backend getBackend(const Ort::KernelContext& ctx);

 private:
  Ort::ConstKernelInfo kernel_info_;
  std::string node_name_;

  bool with_custom_allocator_ = false;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Backend prefill_backend_ = Backend::Cpu;
  Backend token_backend_ = Backend::Cpu;
#else
// by default, use prefill on NPU and token on GPU, if available
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  Backend prefill_backend_ = Backend::Npu;
#else
  Backend prefill_backend_ = Backend::Gpu;
#endif
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  Backend token_backend_ = Backend::Gpu;
#else
  Backend token_backend_ = Backend::Npu;
#endif
#endif

  bool disable_npu_ops_ = false;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  std::shared_ptr<DML_Ops::DMLOps> dml_instance_;
  std::vector<OnnxTensorInfo> tensor_inputs_;
  std::vector<OnnxTensorInfo> tensor_outputs_;
  RyzenMM::GPUAllocator<'HYBK'> gpuAllocator_;
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_HYBRID_KERNEL
