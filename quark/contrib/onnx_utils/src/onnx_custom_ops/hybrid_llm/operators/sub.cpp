// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "sub.hpp"

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
// #include "../npu/ .hpp" (NPU includes if available)
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

SubKernel::SubKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  std::cout << "Constructing Sub Kernel custom op...\n";

  Ort::ConstKernelInfo kernel_info{info};

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();
  int attr_count = 0;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  std::cout << "CPU Path: SubKernel custom op\n";
  const char* add_type_constraint_names[3] = {"T", "T1", "T3"};
  const ONNXTensorElementDataType add_type_constraint_values[3] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
  };

  cpu_op_ = Ort::Op::Create(
    info, "Sub", "", 13 /*kernel version*/, add_type_constraint_names,
    add_type_constraint_values, 1 /*constraint count*/,
    nullptr /*no attributes*/, attr_count, input_count, output_count
  );
  std::cout << "constructed SubKernel custom op\n";
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  // GPU initialization
  if (usingGpu()) {
    try {
      createGpu(session_configs, input_count, 0, 0);

      std::cout << " GPU Path: Constructing " << nodeName()
                << " of type Sub Custom op\n";

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      dml_instance->CreateSubOp(nodeName(), tensor_inputs, tensor_outputs);

      initializeGpu();

      // Initialize the operator
      if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeSub(nodeName(), tensor_inputs, tensor_outputs);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in SUB constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // NPU initialization if available
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  std::cout << "Constructed " << nodeName() << " of type Sub Custom op\n";
}

SubKernel::~SubKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
}

void SubKernel::Compute(OrtKernelContext* context) {
  std::cout << "Executing " << nodeName() << " of type SubKernel Custom op\n";

  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();
  std::vector<const OrtValue*> inputs;
  std::vector<OrtValue*> outputs;

  for (int i = 0; i < input_num; i++) {
    auto input = ctx.GetInput(i);
    inputs.push_back(input);
  }

  auto output_shape = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();

  for (int i = 0; i < output_num; i++) {
    auto output = ctx.GetOutput(i, output_shape);
    outputs.push_back(output);
  }

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      // Recalculate output shapes for CPU using getOutputDimsELWOps
      auto input_0 = ctx.GetInput(0);
      auto input_1 = ctx.GetInput(1);
      std::vector<int64_t> input_0_dim({1});
      std::vector<int64_t> input_1_dim({1});
      if (input_0.IsTensor()) {
        input_0_dim = input_0.GetTensorTypeAndShapeInfo().GetShape();
      }
      if (input_1.IsTensor()) {
        input_1_dim = input_1.GetTensorTypeAndShapeInfo().GetShape();
      }
      std::vector<int64_t> out_dims_cpu = input_0_dim;
      getOutputDimsELWOps(input_0_dim, input_1_dim, out_dims_cpu);

      // Get the single output with the calculated dimensions
      OrtValue* output_cpu = ctx.GetOutput(0, out_dims_cpu);

      // Invoke the operation for CPU
      cpu_op_.Invoke(context, inputs.data(), input_num, &output_cpu, 1);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      // GPU execution
      std::cout << "GPU Path for: " << nodeName()
                << " of type SubKernel Custom op\n";

      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      if (withCustomAllocator() || dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeSub(nodeName(), tensor_inputs, tensor_outputs);
      }

      // Execute the operator
      dml_instance->ComputeSubGPU(nodeName(), tensor_inputs, tensor_outputs);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }

    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      // NPU execution if available
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

  std::cout << "Execution done for: " << nodeName()
            << " of type SubKernel Custom op\n";
}

}  // namespace ryzenai::onnx_utils
