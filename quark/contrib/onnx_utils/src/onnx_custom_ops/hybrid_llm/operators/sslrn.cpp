// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "sslrn.hpp"

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#include "opUtils.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
// #include "../npu/ .hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

SkipSimplifiedLayerNormKernel::SkipSimplifiedLayerNormKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  std::cout << "Constructing SkipSimplifiedLayerNorm Kernel custom op...\n";

  Ort::ConstKernelInfo kernel_info{info};

  epsilon_ = getAttribute<float>("epsilon", 0.0f);

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  const std::string domain = "com.microsoft";
  constexpr auto version = 1;
  constexpr auto type_constraint_count = 4;

  std::array<const char*, type_constraint_count> type_constraint_names = {"T1"};
  std::array<ONNXTensorElementDataType, type_constraint_count>
    type_constraint_values = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};

  cpu_op_ = Ort::Op::Create(
    info, "SkipSimplifiedLayerNormalization", domain.c_str(), version,
    type_constraint_names.data(), type_constraint_values.data(),
    type_constraint_count, nullptr, 0, input_count, output_count
  );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  // GPU stuff below
  if (usingGpu()) {
    try {
      createGpu(session_configs, input_count, 0, 0);

      std::cout << " GPU Path: Constructing " << nodeName()
                << " of type SkipSimplifiedLayerNormKernel Custom op\n ";

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      dml_instance->CreateSkipSimplifiedLayerNormOp(
        nodeName(), tensor_inputs, tensor_outputs, epsilon_
      );

      initializeGpu();

      if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
        // set D3D resources and upload consts / weights
        dml_instance->InitializeSkipSimplifiedLayerNorm(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in SSLRM constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::cout << "NPU Path: SkipSimplifiedLayerNorm custom op\n";
  // npu_instance_ = std::make_unique<npu sslrn here>(info);
  throw std::invalid_argument("NPU need to add SkipSimplifiedLayerNorm custom");
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  std::cout << "Constructed " << nodeName()
            << " of type SkipSimplifiedLayerNorm Custom op\n\n ";
}

SkipSimplifiedLayerNormKernel::~SkipSimplifiedLayerNormKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
}

void SkipSimplifiedLayerNormKernel::Compute(OrtKernelContext* context) {
  std::cout << "Executing " << nodeName()
            << " of type SkipSimplifiedLayerNormKernel Custom op\n\n";

  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();
  std::vector<const OrtValue*> inputs;
  std::vector<OrtValue*> outputs;

  for (int i = 0; i < input_num; i++) {
    auto input = ctx.GetInput(i);
    inputs.push_back(input);
  }

  auto dimensions = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();

  for (int i = 0; i < output_num; i++) {
    auto output = ctx.GetOutput(i, dimensions);
    outputs.push_back(output);
  }

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      // cpu_op_.Invoke(context, inputs.data(), input_num, outputs.data(),
      // output_num);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      std::cout << "GPU Path for: " << nodeName()
                << " of type SkipSimplifiedLayerNormKernel Custom op\n";

      computeGpu(ctx);
      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      dml_instance->ComputeSkipSimplifiedLayerNormGPU(
        nodeName(), tensor_inputs, tensor_outputs
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      std::cout << "NPU Path: Compute function\n";
      // npu_instance_->Compute(context);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

  std::cout << "\nExecution done for: " << nodeName()
            << " of type SkipSimplifiedLayerNormKernel Custom op\n\n";
}

}  // namespace ryzenai::onnx_utils
