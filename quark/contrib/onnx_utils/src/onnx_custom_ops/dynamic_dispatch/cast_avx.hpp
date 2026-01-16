// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_AVX
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_AVX

#include <any>
#include <memory>
#include <type_traits>

#include "dynamic_dispatch.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "operator.hpp"
#include "ops/ops_common/dtype_utils.h"

namespace ryzenai::onnx_utils {

struct CastAvxKernel : ExecutionProviderExtensions {
  CastAvxKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  )
    : ort_(ort_api) {
    const auto info_obj = Ort::ConstKernelInfo(info);
    to_ = static_cast<ONNXTensorElementDataType>(
      info_obj.GetAttribute<int64_t>("to")
    );
  }

  void Compute(OrtKernelContext* context) {
    auto ctx = Ort::KernelContext(context);

    const auto input_tensor = ctx.GetInput(0);
    const auto* input_data = input_tensor.GetTensorRawData();
    ONNXTensorElementDataType input_dtype =
      input_tensor.GetTensorTypeAndShapeInfo().GetElementType();
    auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto elements = input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    auto output_tensor = ctx.GetOutput(0, input_shape);
    auto* output_data = output_tensor.GetTensorMutableRawData();
    if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      RecordDuration(Metric::Casting, [&]() {
        ryzenai::float_buffer_to_bfloat16(
          (float*)input_data, elements, (uint16_t*)output_data
        );
      });
    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      RecordDuration(Metric::Casting, [&]() {
        ryzenai::bfloat16_buffer_to_float(
          (uint16_t*)input_data, elements, (float*)output_data
        );
      });
    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      RecordDuration(Metric::Casting, [&]() {
        ryzenai::float16_buffer_to_bfloat16(
          (uint16_t*)input_data, elements, (uint16_t*)output_data
        );
      });
    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      RecordDuration(Metric::Casting, [&]() {
        ryzenai::bfloat16_buffer_to_float16(
          (uint16_t*)input_data, elements, (uint16_t*)output_data
        );
      });
    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      RecordDuration(Metric::Casting, [&]() {
        ryzenai::float16_buffer_to_float(
          (uint16_t*)input_data, elements, (float*)output_data
        );
      });
    } else {
      throw std::invalid_argument(
        "Unsupported CastAvx type from " +
        std::string(type_to_str(input_dtype)) + " to " +
        std::string(type_to_str(to_))
      );
    }
  }

 private:
  OrtApi ort_{};
  ONNXTensorElementDataType to_;
};

static const char kCastAvx[] = "CastAvx";
struct CastAvx : Operator<CastAvxKernel, kCastAvx> {
  using Operator::Operator;
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_AVX
