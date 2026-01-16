// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "ssgmlp.hpp"

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "opUtils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

SSGMlpKernel::SSGMlpKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  printLog("Constructing SSMlp Kernel custom op...");

  Ort::ConstKernelInfo kernel_info{info};

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    try {
      AttributeForSSMLP attrForSSMLP;
      attrForSSMLP.epsilon = getAttribute<float>("epsilon");

      attrForSSMLP.gateBits = getAttribute<int64_t>("gate_bits");
      attrForSSMLP.gateBlockSize = getAttribute<int64_t>("gate_block_size");
      attrForSSMLP.gate_K = getAttribute<int64_t>("gate_K");
      attrForSSMLP.gate_N = getAttribute<int64_t>("gate_N");

      attrForSSMLP.upBits = getAttribute<int64_t>("up_bits");
      attrForSSMLP.upBlockSize = getAttribute<int64_t>("up_block_size");
      attrForSSMLP.up_K = getAttribute<int64_t>("up_K");
      attrForSSMLP.up_N = getAttribute<int64_t>("up_N");

      attrForSSMLP.downBits = getAttribute<int64_t>("down_bits");
      attrForSSMLP.downBlockSize = getAttribute<int64_t>("down_block_size");
      attrForSSMLP.down_K = getAttribute<int64_t>("down_K");
      attrForSSMLP.down_N = getAttribute<int64_t>("down_N");

      attrForSSMLP.has_gelu = getAttribute<int64_t>("has_gelu", 0);
      auto iCount = (attrForSSMLP.has_gelu == 1) ? 15 : 13;
      auto gpu_input_count = input_count > iCount ? iCount : input_count;

      createGpu(session_configs, gpu_input_count, 1, 0);

      printLog(
        " GPU Path: Constructing ", nodeName(), " of type SSMlp Custom op "
      );

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      dml_instance->CreateSSMLPOp(
        nodeName(), tensor_inputs, tensor_outputs, attrForSSMLP
      );

      initializeGpu();

      if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
        // set D3D resources and upload consts / weights
        dml_instance->InitializeSSMLP(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in SSGMLP constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  printLog("NPU Path: SSGMLP custom op");
  npu_instance_ = std::make_unique<AMDSSGMLPKernel>(info, session_configs);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  printLog("Constructed ", nodeName(), " of type SSGMlp Custom op\n ");
}

SSGMlpKernel::~SSGMlpKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#if defined(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU) && \
  defined(ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING)
  std::ostringstream os;

  for (const auto& [event_id, duration] :
       dml_instance->GetPerfData(nodeName())) {
    os << nodeName() << ",GPUSSMLP," << event_id << ","
       << MillisecondsFp{duration}.count() << "\n";
  }

  std::cout << os.str() << std::flush;
#endif
}

void SSGMlpKernel::Compute(OrtKernelContext* context) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  printLog("Executing ", nodeName(), " of type SSMlp Custom op\n\n");
#endif

  Ort::KernelContext ctx(context);
  auto dimensions = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();
  std::vector<const OrtValue*> inputs;
  std::vector<OrtValue*> outputs;

  for (int i = 0; i < input_num; i++) {
    auto input = ctx.GetInput(i);
    inputs.push_back(input);
  }

  for (int i = 0; i < output_num; i++) {
    auto output = ctx.GetOutput(i, dimensions);
    outputs.push_back(output);
  }
#endif

  switch (auto backend = getBackend(ctx); backend) {
    case Backend::Cpu: {
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      // GPU stuff
      printLog("GPU Path for: ", nodeName(), " of type SSMlp Custom op\n\n");

      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      if (withCustomAllocator() || dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeSSMLP(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
      // PROFILE_THIS(dml_instance->ComputeSSMLPGPU(nodeName(), tensor_inputs,
      //                                             tensor_outputs));
      dml_instance->ComputeSSMLPGPU(
        nodeName(), tensor_inputs, tensor_outputs, reBindD3DResc
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      releaseGpuJitWeights();
      // std::cout << "NPU Path: Compute function\n";
      npu_instance_->Compute(context);
      // PROFILE_THIS(npu_instance_->Compute(context));
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  printLog(
    "\nExecution done for: ", nodeName(), " of type SSMlp Custom op\n\n"
  );
#endif
}

}  // namespace ryzenai::onnx_utils
