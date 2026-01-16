// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "qmoe.hpp"

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <queue>
#include <type_traits>

#include "external_data.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "opUtils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/qmoe.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

QmoeKernel::QmoeKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  // printLog("Constructing QmoeKernel custom op...");

  Ort::ConstKernelInfo kernel_info{info};

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  if (!disableNpuOps()) {
    printLog("NPU Path: QMoE custom op\n");
    npu_instance_ = std::make_unique<AMDQMoEKernel>(info, session_configs);
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  printLog("Constructed ", nodeName(), " of type QmoeKernel Custom op\n ");
}
QmoeKernel::~QmoeKernel() {}

void QmoeKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
      throw std::runtime_error("CPU is unsupported backend for qmoe op");
      break;
    }
    case Backend::Gpu: {
      throw std::runtime_error("GPU is unsupported backend for qmoe op");
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      releaseGpuJitWeights();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      npu_instance_->Compute(context);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }
}

}  // namespace ryzenai::onnx_utils
