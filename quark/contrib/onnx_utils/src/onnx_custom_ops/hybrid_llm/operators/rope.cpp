// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "rope.hpp"

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

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

RotaryEmbeddingKernel::RotaryEmbeddingKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  std::cout << "Constructing RotaryEmbeddingKernel custom op...\n";

  Ort::ConstKernelInfo kernel_info{info};

  interleaved_ = getAttribute<int64_t>("interleaved");

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    try {
      createGpu(session_configs, input_count, 0, 0);
      std::cout << " GPU Path: Constructing " << nodeName()
                << " of type RotaryEmbeddingKernel Custom op\n ";

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      dml_instance->CreateRotaryEmbeddingOperator(
        nodeName(), tensor_inputs, tensor_outputs, interleaved_
      );

      initializeGpu();
      if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
        // Set D3D resources and upload consts / weights
        dml_instance->InitializeRotaryEmbedding(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in ROPE constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

  std::cout << "Constructed " << nodeName()
            << " of type RotaryEmbeddingKernel Custom op\n\n ";
}

RotaryEmbeddingKernel::~RotaryEmbeddingKernel() {}

void RotaryEmbeddingKernel::Compute(OrtKernelContext* context) {
  std::cout << "Executing " << nodeName()
            << " of type RotaryEmbeddingKernel Custom op\n\n";

  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();

  auto input_0 = ctx.GetInput(0);  // input tensor
  auto input_1 = ctx.GetInput(1);  // position ids
  auto input_2 = ctx.GetInput(2);  // cos cache
  auto input_3 = ctx.GetInput(3);  // sin cache

  // this will be [batch_size, sequence_length, num_heads, head_size]
  auto dimensions = input_0.GetTensorTypeAndShapeInfo().GetShape();

  // output dim should be [batch_size, sequence_length, num_heads, head_size]
  auto output_0 = ctx.GetOutput(0, dimensions);

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      if (withCustomAllocator() || dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeRotaryEmbedding(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }

      // TODO(agurmukh): update to use rebind
      dml_instance->ComputeRotaryEmbeddingGPU(
        nodeName(), tensor_inputs, tensor_outputs
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
      break;
    }
  }
}

}  // namespace ryzenai::onnx_utils
