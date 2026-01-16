// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "cast.hpp"

#include <iostream>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"

namespace ryzenai::onnx_utils {

CastKernel::CastKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : ort_(ort_api) {
  std::cout << "Constructing CastKernel custom op...\n";
  Ort::ConstKernelInfo kernel_info{info};
  node_name_ = kernel_info.GetNodeName();
  int input_count = 1;
  int out_count = 1;
  int attr_count = 2;

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  std::cout << "CPU Path: CastKernel custom op\n";
  to_attr_ = kernel_info.GetAttribute<int64_t>("to");
  try {
    saturate_attr_ = kernel_info.GetAttribute<int64_t>("saturate");
  } catch (const Ort::Exception&) {
    saturate_attr_ = 1;
  }
  const char* add_type_constraint_names[2] = {"T1", "T2"};
  auto to =
    Ort::OpAttr("to", &to_attr_, 1 /*size*/, OrtOpAttrType::ORT_OP_ATTR_INT);
  auto saturate = Ort::OpAttr(
    "saturate", &saturate_attr_, 1 /*size*/, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  Ort::OpAttr cast_attrs[2] = {std::move(to), std::move(saturate)};
  const ONNXTensorElementDataType add_type_constraint_values[2] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
  };
  cpu_op_ = Ort::Op::Create(
    info, "Cast", "", 11 /*kernel version*/, add_type_constraint_names,
    add_type_constraint_values, 2 /*constraint count*/, cast_attrs, attr_count,
    input_count, out_count
  );
  std::cout << "constructed CastKernel custom op\n";
#endif
}

CastKernel::~CastKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif
}

void CastKernel::Compute(OrtKernelContext* context) {
  // std::cout << "Running CastKernel custom op...\n";
  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();
  auto input_0 = ctx.GetInput(0);
  auto input_0_dim = input_0.GetTensorTypeAndShapeInfo().GetShape();
  auto output_0 = ctx.GetOutput(0, input_0_dim);
  const OrtValue* inputs[1] = {input_0};
  OrtValue* outputs[1] = {output_0};
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.Invoke(context, inputs, input_num, outputs, output_num);
#endif
}

}  // namespace ryzenai::onnx_utils
