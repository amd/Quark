// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "gather.hpp"

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

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
// #include "../npu/gather.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

GatherKernel::GatherKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  std::cout << "Constructing GatherKernel custom op...\n";

  Ort::ConstKernelInfo kernel_info{info};
  axis_attr_ = getAttribute<int64_t>("axis", 0);
  RegisterKernel(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, info);
  RegisterKernel(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, info);
  RegisterKernel(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, info);

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

  // DX resource allocator for gpu and cpu handle
  std::vector<void*> cpuInputMappedMemory;
  std::vector<void*> cpuOutputMappedMemory;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  auto* dml_instance = dmlInstance();
  const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
  createGpu(session_configs, input_count, 0, 0);
  dml_instance->CreateGatherOperator(nodeName(), tensor_inputs, tensor_outputs);

  initializeGpu();

  if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
    // set D3D resources and upload consts / weights
    dml_instance->Initialize(nodeName(), tensor_inputs, tensor_outputs);
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::cout << "NPU Path: Gather custom op\n";
  throw std::invalid_argument("NPU need to add Gather custom");
  // npu_instance_ = std::make_unique<AMDGatherKernel>(info);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  std::cout << "Constructed " << nodeName() << " of type Gather Custom op\n\n ";
}

void GatherKernel::RegisterKernel(
  const ONNXTensorElementDataType& type, const OrtKernelInfo* info
) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  int input_count = 2;
  int attr_count = 1;
  int out_count = 1;
  std::cout << "CPU Path: GatherKernel custom op\n";
  const char* add_type_constraint_names[2] = {"T", "Tind"};
  const ONNXTensorElementDataType add_type_constraint_values[2] = {
    type, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
  };

  auto axis = Ort::OpAttr(
    "axis", &axis_attr_, 1 /*size*/, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  Ort::OpAttr gather_attrs[1] = {std::move(axis)};
  cpu_op_map_.insert(
    OpMap::value_type(
      type,
      Ort::Op::Create(
        info, "Gather", "", 13 /*kernel version*/, add_type_constraint_names,
        add_type_constraint_values, 2 /*constraint count*/, gather_attrs,
        attr_count, input_count, out_count
      )
    )
  );
#endif
}

GatherKernel::~GatherKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  for (auto& ops : cpu_op_map_) {
    ops.second.release();
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
}

void GatherKernel::Compute(OrtKernelContext* context) {
  std::cout << "Executing " << nodeName() << " of type Gather Custom op\n\n";

  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();

  auto input_0 = ctx.GetInput(0);  // data (float16)
  auto input_1 = ctx.GetInput(1);  // indices (int64)

  auto input_0_ty_info = input_0.GetTensorTypeAndShapeInfo();
  auto input_0_dim = input_0_ty_info.GetShape();
  auto indices_dim = input_1.GetTensorTypeAndShapeInfo().GetShape();
  HandleNegativeAxis(axis_attr_, input_0_dim.size());
  std::vector<int64_t> out_dims = input_0_dim;

  if (indices_dim.size() > 0) {
    out_dims[axis_attr_] = indices_dim[0];
  } else {
    const int64_t* data = input_1.GetTensorData<int64_t>();
    if (data != nullptr) {
      out_dims[axis_attr_] = *data;
    } else {
      out_dims = std::vector<int64_t>();
    }
  }

  for (int i = 1; i < indices_dim.size(); i++) {
    out_dims.insert(out_dims.begin() + axis_attr_ + i, indices_dim[i]);
  }

  auto output_0 = ctx.GetOutput(0, out_dims);

  std::vector<const OrtValue*> inputs = {input_0, input_1};
  OrtValue* outputs[1] = {output_0};

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      auto type = input_0_ty_info.GetElementType();
      auto type_op = cpu_op_map_.find(type);
      if (type_op == cpu_op_map_.end()) {
        throw std::runtime_error("Unsupported type for inputs\n");
      }
      type_op->second.Invoke(
        context, inputs.data(), input_num, outputs, output_num
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
        dml_instance->Initialize(nodeName(), tensor_inputs, tensor_outputs);
      }
      dml_instance->ComputeGatherGPU(nodeName(), tensor_inputs, tensor_outputs);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }

    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      std::cout << "NPU Path: Compute function\n";
      // npu_instance_->Compute(context);
      throw std::invalid_argument("NPU path for gather not implemented");
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

  std::cout << "\nExecution done for: " << nodeName()
            << " of type GatherKernel Custom op\n\n";
}

}  // namespace ryzenai::onnx_utils
