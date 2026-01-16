// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_GATHER
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_GATHER

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

// class AMDMatMulNBitsKernel;
// forward-declaring these classes gave linker errors for some reason
// class OnnxTensorInfo;
// namespace DML_Ops {
//   class DMLOps;
// } // namespace DML_Ops

class GatherKernel : public HybridKernel {
 public:
  GatherKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~GatherKernel();

  void Compute(OrtKernelContext* context);

 private:
  void RegisterKernel(
    const ONNXTensorElementDataType& type, const OrtKernelInfo* info
  );

  OrtApi ort_{};
  Ort::KernelInfo info_{nullptr};
  std::string node_name_;
  int64_t axis_attr_;
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  using OpMap = std::unordered_map<ONNXTensorElementDataType, Ort::Op>;
  OpMap cpu_op_map_{};
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // std::unique_ptr<AMDGatherKernel> npu_instance_;
#endif
};

static const char kGather[] = "Gather";

struct Gather : Operator<GatherKernel, kGather> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_GATHER
