// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <sstream>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif

namespace ryzenai::onnx_utils {

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
inline hstring ElementFormat(ONNXTensorElementDataType dataType) {
  hstring format = L"Unknown";

  switch (dataType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      format = L"Float";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      format = L"Uint32";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      format = L"Int32";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      format = L"Float16";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      format = L"Uint16";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      format = L"Int16";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      format = L"Uint4";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      format = L"Float64";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      format = L"Uint4";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      format = L"Int4";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      format = L"Int64";
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      format = L"Int8";
      break;
    default:
      throw std::runtime_error("Unknown tensor data type.");
      break;
  }

  return format;
}

inline uint64_t GetPackedTensorSize(
  ONNXTensorElementDataType dataType, uint64_t elementCount
) {
  uint64_t sizeInBytes = 0;

  switch (dataType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      sizeInBytes = elementCount * sizeof(float_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      sizeInBytes = elementCount * sizeof(uint32_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      sizeInBytes = elementCount * sizeof(int32_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      sizeInBytes = elementCount * sizeof(Ort::Float16_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      sizeInBytes = elementCount * sizeof(uint16_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      sizeInBytes = elementCount * sizeof(int16_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      sizeInBytes = elementCount * sizeof(uint8_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      sizeInBytes = elementCount * sizeof(double_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      sizeInBytes = (elementCount + 1) / 2;
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      sizeInBytes = (elementCount + 1) / 2;
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      sizeInBytes = elementCount * sizeof(int64_t);
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      sizeInBytes = elementCount * sizeof(int8_t);
      break;
    default:
      throw std::runtime_error("Unknown tensor data type.");
      break;
  }

  return sizeInBytes;
}
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

// Prints the logs
template <typename... Args>
void printLog(Args... args) {
#ifdef ENABLE_PRINT
  std::ostringstream oss;
  (oss << ... << args);
  std::cout << oss.str() << std::endl;
#endif
}

template <typename T>
typename std::enable_if<
  std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>>::
  type static HandleNegativeAxis(T& value, int64_t size) {
  if (value < 0) {
    value = value + size;
  }
}

template <typename T>
static void HandleNegativeAxis(std::vector<T>& vec, size_t size) {
  for (auto& elem : vec) {
    HandleNegativeAxis(elem, size);
  }
}

template <typename T>
typename std::enable_if<
  std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t>>::
  type static getOutputDimsELWOps(
    const std::vector<T>& in_dims0, const std::vector<T>& in_dims1,
    std::vector<T>& out_dims
  ) {
  // ONNX follows numpy broadcasting rules for elements wise operators.
  // https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
  auto errorMag = [&]() -> std::string {
    std::ostringstream oss;
    oss << " [ERROR] Can't broadcast shapes : ( ";
    for (auto it = in_dims0.begin(); it != in_dims0.end(); ++it) {
      oss << *it << ",";
    }
    oss << " ) and  ( ";
    for (auto it = in_dims1.begin(); it != in_dims1.end(); ++it) {
      oss << *it << ",";
    }
    oss << " ) .\n";
    return oss.str();
  };
  auto dim_size = std::max(in_dims0.size(), in_dims1.size());
  auto in0_offset = dim_size - in_dims0.size();
  auto in1_offset = dim_size - in_dims1.size();
  out_dims.resize(dim_size);

  for (int i = 0; i < dim_size; i++) {
    if (i < in0_offset) {
      out_dims[i] = in_dims1[i];
    } else if (i < in1_offset) {
      out_dims[i] = in_dims0[i];
    } else if (in_dims0[i - in0_offset] == -1 ||
               in_dims1[i - in1_offset] == -1) {
      out_dims[i] = -1;
    } else if (in_dims0[i - in0_offset] == 1 || in_dims1[i - in1_offset] == 1 ||
               in_dims0[i - in0_offset] == in_dims1[i - in1_offset]) {
      out_dims[i] =
        std::max(in_dims0[i - in0_offset], in_dims1[i - in1_offset]);
    } else {
      throw std::runtime_error(errorMag());
    }
  }
}

}  // namespace ryzenai::onnx_utils
