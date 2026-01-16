// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "slrn.hpp"

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

SimplifiedLayerNormKernel::SimplifiedLayerNormKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  printLog("Constructing SimplifiedLayerNorm Kernel custom op...\n");

  api_ = &ort_api;

  Ort::ConstKernelInfo kernel_info{info};

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

  try {
    shape_in_ = kernel_info.GetAttributes<int64_t>("shape_in");
  } catch (const Ort::Exception&) {
    shape_in_ = {};
  }

  try {
    shape_out_ = kernel_info.GetAttributes<int64_t>("shape_out");
  } catch (const Ort::Exception&) {
    shape_out_ = {};
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  /*if (!shape_in_.empty()) {
    int64_t shape_shape[1] = {shape_in_.size()};
    Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto shape_type =
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    api_->CreateTensorWithDataAsOrtValue(mem_info, shape_in_.data(),
                                        shape_in_.size() * sizeof(int64_t),
                                        shape_shape, 1, shape_type,
  &shape_in_tensor_);
  }

  if (!shape_in_.empty()) {
    const std::string domain = "ai.onnx";//"com.microsoft";
    constexpr auto type_constraint_count = 1;
    std::array<const char*, type_constraint_count> type_constraint_names =
  {"T"}; std::array<ONNXTensorElementDataType, type_constraint_count>
      type_constraint_values = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};
    //Ort::OpAttrList attrs;
    //attrs.values["shape"].p_s = shape_tensor_;
    reshape1_ = Ort::Op::Create(
        info, "Reshape", domain.c_str(), 19,
        type_constraint_names.data(), type_constraint_values.data(),
        type_constraint_count, nullptr, 0, 2, 1);
  }

  const std::string domain = "com.microsoft";
  constexpr auto type_constraint_count = 1;
  std::array<const char*, type_constraint_count> type_constraint_names = {"T1"};
  std::array<ONNXTensorElementDataType, type_constraint_count>
    type_constraint_values = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};
  cpu_op_ = Ort::Op::Create(
    info, "SimplifiedLayerNormalization", domain.c_str(), 17,
    type_constraint_names.data(), type_constraint_values.data(),
    type_constraint_count, nullptr, 0, input_count, output_count);

  if (!shape_out_.empty()) {
    int64_t shape_shape[1] = {shape_out_.size()};
    Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto shape_type =
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    api_->CreateTensorWithDataAsOrtValue(mem_info, shape_out_.data(),
                                        shape_out_.size() * sizeof(int64_t),
                                        shape_shape, 1, shape_type,
  &shape_out_tensor_);


  }*/

#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  // GPU stuff below
  if (usingGpu()) {
    AttributeForSlrn attrForSlrn;
    attrForSlrn.epsilon = getAttribute<float>("epsilon", 0.0f);
    attrForSlrn.shapeIn = shape_in_;
    attrForSlrn.shapeOut = shape_out_;

    createGpu(session_configs, input_count, 0, 0);

    printLog(
      " GPU Path: Constructing ", nodeName(),
      " of type SimplifiedLayerNormKernel Custom op\n "
    );

    auto* dml_instance = dmlInstance();
    const auto& [tensor_inputs, tensor_outputs] = gpuTensors();

    dml_instance->CreateSlrnOp(
      nodeName(), tensor_inputs, tensor_outputs, attrForSlrn
    );

    initializeGpu();

    if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
      // set D3D resources and upload consts / weights
      dml_instance->InitializeSlrn(nodeName(), tensor_inputs, tensor_outputs);
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  if (!disableNpuOps()) {
    printLog("NPU Path: SimplifiedLayerNorm custom op\n");
    npu_instance_ = std::make_unique<AMDSLRNKernel>(info, session_configs);
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  printLog(
    "Constructed ", nodeName(), " of type SimplifiedLayerNorm Custom op\n\n "
  );
}

SimplifiedLayerNormKernel::~SimplifiedLayerNormKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  // reshape1_.release();
  // cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
}

void SimplifiedLayerNormKernel::run_cpu_reshape(
  OrtKernelContext* context, Ort::ConstValue& input, OrtValue* shape_tensor
) {
  std::vector<const OrtValue*> reshape1_inputs;
  reshape1_inputs.push_back(input);
  reshape1_inputs.push_back(shape_tensor);
  std::vector<const OrtValue*> reshape1_outputs;
  reshape1_outputs.push_back(input);
  reshape1_.Invoke(
    context, (OrtValue* const*)reshape1_inputs.data(), 2,
    (OrtValue* const*)reshape1_outputs.data(), 1
  );
  auto out_shape = input.GetTensorTypeAndShapeInfo().GetShape();
  std::cout << "out shape after run_cpu_reshape:" << out_shape[0] << ","
            << out_shape[1] << "," << out_shape[2] << std::endl;
}

void SimplifiedLayerNormKernel::Compute(OrtKernelContext* context) {
  printLog(
    "Executing ", nodeName(), " of type SimplifiedLayerNormKernel Custom op\n\n"
  );

  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();

  auto input = ctx.GetInput(0);
  auto dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  auto input_elem_cnt = input.GetTensorTypeAndShapeInfo().GetElementCount();

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      if (!shape_in_.empty()) {
        run_cpu_reshape(context, input, shape_in_tensor_);
      }
      auto out_shape = input.GetTensorTypeAndShapeInfo().GetShape();
      // std::cout << "out shape after reshape:" << out_shape[0] << "," <<
      // out_shape[1] << "," << out_shape[2] << std::endl;

      std::vector<const OrtValue*> inputs;
      inputs.push_back(input);
      std::vector<OrtValue*> outputs;
      for (int i = 1; i < input_num; i++) {
        auto input = ctx.GetInput(i);
        inputs.push_back(input);
      }

      for (int i = 0; i < output_num; i++) {
        auto output = ctx.GetOutput(i, dimensions);
        outputs.push_back(output);
      }

      cpu_op_.Invoke(
        context, inputs.data(), input_num, outputs.data(), output_num
      );
      if (!shape_out_.empty()) {
        // run_cpu_reshape(context, ctx.GetOutput(0, dimensions),
        // shape_out_tensor_);
        std::vector<const OrtValue*> reshape_inputs;
        reshape_inputs.push_back(ctx.GetOutput(0, dimensions));
        reshape_inputs.push_back(shape_out_tensor_);
        std::vector<const OrtValue*> reshape_outputs;
        reshape_outputs.push_back(ctx.GetOutput(0, dimensions));
        reshape2_.Invoke(
          context, (OrtValue* const*)reshape_inputs.data(), 2,
          (OrtValue* const*)reshape_outputs.data(), 1
        );
        auto out_shape =
          ctx.GetOutput(0, dimensions).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "out shape after run_cpu_reshape:" << out_shape[0] << ","
                  << out_shape[1] << "," << out_shape[2] << std::endl;
      }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      printLog(
        "GPU Path for: ", nodeName(),
        " of type SimplifiedLayerNormKernel Custom op\n"
      );

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      auto reBindD3DResc = computeGpu(ctx);
      dml_instance->InitializeSlrn(nodeName(), tensor_inputs, tensor_outputs);
      dml_instance->ComputeSlrn(
        nodeName(), tensor_inputs, tensor_outputs, reBindD3DResc
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      releaseGpuJitWeights();
      printLog("NPU Path: Compute function\n");

      npu_instance_->Compute(context);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

  printLog(
    "\nExecution done for: ", nodeName(),
    " of type SimplifiedLayerNormKernel Custom op\n\n"
  );
  // std::cout << "SimplifiedLayerNormKernel::Compute end" << std::endl;
}

}  // namespace ryzenai::onnx_utils
