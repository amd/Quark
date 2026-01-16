// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "onnx.hpp"

namespace ryzenai::onnx_utils {

const char* type_to_str(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "bfp16ebs8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    default:
      throw std::invalid_argument(
        "Invalid onnx type to convert to string: " + std::to_string(type)
      );
  }
}

::Tensor getInputTensor(
  const Ort::KernelContext& ctx, int index, const std::vector<int64_t>& shape
) {
  const auto tensor = ctx.GetInput(index);
  auto type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
  auto* data = const_cast<void*>(tensor.GetTensorRawData());
  std::vector<size_t> shape_dd{shape.begin(), shape.end()};
  const std::string dtype = type_to_str(type);
  return {data, shape_dd, dtype};
}

::Tensor getInputTensor(const Ort::KernelContext& ctx, int index) {
  const auto tensor = ctx.GetInput(index);
  auto type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
  auto* data = const_cast<void*>(tensor.GetTensorRawData());
  auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<size_t> shape_dd{shape.begin(), shape.end()};
  const std::string dtype = type_to_str(type);
  return {data, shape_dd, dtype};
}

::Tensor getOutputTensor(
  const Ort::KernelContext& ctx, int index, const std::vector<int64_t>& shape
) {
  auto tensor = ctx.GetOutput(index, shape);
  auto type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
  auto* data = tensor.GetTensorMutableRawData();
  std::vector<size_t> dd_shape{shape.begin(), shape.end()};
  const std::string dtype = type_to_str(type);
  return {data, dd_shape, dtype};
}

}  // namespace ryzenai::onnx_utils
