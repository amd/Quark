// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "reducesum.hpp"

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

ReduceSumKernel::ReduceSumKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  [[maybe_unused]] const std::unordered_map<std::string, std::string>&
    session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  std::cout << "Constructing ReduceSum Kernel custom op...\n";

  Ort::ConstKernelInfo kernel_info{info};

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

  keepdims_attr_ = getAttribute<int64_t>("keepdims", 1);
  noop_with_empty_axes_attr_ = getAttribute<int64_t>("noop_with_empty_axes", 0);

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  std::cout << "CPU Path: ReduceSumKernel custom op\n";
  const char* add_type_constraint_names[2] = {"T", "I"};
  const ONNXTensorElementDataType add_type_constraint_values[1] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
  };

  auto keepdims =
    Ort::OpAttr("keepdims", &keepdims_attr_, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  auto noop_empty_axis = Ort::OpAttr(
    "noop_with_empty_axes", &noop_with_empty_axes_attr_, 1,
    OrtOpAttrType::ORT_OP_ATTR_INT
  );
  constexpr auto attr_count = 2;
  Ort::OpAttr red_sum_attrs[attr_count] = {
    std::move(keepdims), std::move(noop_empty_axis)
  };
  // TODO(mohmohit): check llama3
  cpu_op_ = Ort::Op::Create(
    info, "ReduceSum", "ai.onnx", 10, add_type_constraint_names,
    add_type_constraint_values, 1, red_sum_attrs, attr_count, input_count,
    output_count
  );
  std::cout << "constructed ReduceSumKernel custom op\n";
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    try {
      createGpu(session_configs, input_count, 0, 0);
      // GPU stuff below

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      std::cout << " GPU Path: Constructing " << nodeName()
                << " of type ReduceSumKernel Custom op\n ";

      dml_instance->CreateReduceOp(nodeName(), tensor_inputs, tensor_outputs);

      initializeGpu();

      // set D3D resources and upload consts / weights
      dml_instance->InitializeReduce(nodeName(), tensor_inputs, tensor_outputs);
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in ReduceSum constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::cout << "NPU Path: Reduce custom op\n";
  // npu_instance_ = std::make_unique<npu sslrn here>(info);
  throw std::invalid_argument("NPU need to add Reduce custom");
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  std::cout << "Constructed " << nodeName() << " of type Reduce Custom op\n\n ";
}

ReduceSumKernel::~ReduceSumKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
}

void ReduceSumKernel::Compute(OrtKernelContext* context) {
  std::cout << "Executing " << nodeName()
            << " of type ReduceSumKernel Custom op\n\n";

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

  std::vector<int64_t> axes;
  if (inputs[1] != nullptr /*optional*/) {
    axes = ctx.GetInput(1).GetTensorTypeAndShapeInfo().GetShape();
  }
  HandleNegativeAxis(axes, axes.size());
  std::vector<int64_t> out_dims;
  for (int i = 0; i < dimensions.size(); ++i) {
    if ((noop_with_empty_axes_attr_ && axes.empty() /*reduce all dim*/) ||
        std::find(axes.begin(), axes.end(), i) == axes.end())
      out_dims.push_back(dimensions[i]);
    else {
      if (keepdims_attr_ == 1) {
        out_dims.push_back(1);
      }
    }
  }
  auto output_0 = ctx.GetOutput(0, out_dims);

  for (int i = 0; i < output_num; i++) {
    auto output = ctx.GetOutput(i, out_dims);
    outputs.push_back(output);
  }

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      cpu_op_.Invoke(
        context, inputs.data(), input_num, outputs.data(), output_num
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      if (withCustomAllocator() || dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeReduce(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
      // TODO(agurmukh): update to use rebind
      dml_instance->ComputeReduceGPU(nodeName(), tensor_inputs, tensor_outputs);
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
            << " of type ReduceSumKernel Custom op\n\n";
}

}  // namespace ryzenai::onnx_utils
