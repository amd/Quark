// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_MATMUL
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_MATMUL

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#include "../ort/cast.hpp"

namespace ryzenai::onnx_utils {

class OrtMatMul;

class MatMulKernel : public HybridKernel {
 public:
  MatMulKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~MatMulKernel();

  void Compute(OrtKernelContext* context);

 private:
  std::unique_ptr<OrtMatMul> ort_matmul_;
  std::unique_ptr<OrtCast<Ort::Float16_t, float>> ort_cast_fp16_to_fp32_;
  std::unique_ptr<OrtCast<float, Ort::Float16_t>> ort_cast_fp32_to_fp16_;
  int64_t prune_en_{0};
};

template <const char* kName>
struct MatMulBase : public HybridOperator<MatMulKernel, kName> {
  explicit MatMulBase(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<MatMulKernel, kName>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const { return {}; }

  size_t GetInputTypeCount() const noexcept override { return 2; }

  size_t GetOutputTypeCount() const noexcept override { return 1; }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    return INPUT_OUTPUT_REQUIRED;
  }

  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept override {
    return INPUT_OUTPUT_REQUIRED;
  }
};

static const char kMatMul[] = "MatMul";

struct MatMul : MatMulBase<kMatMul> {
  using MatMulBase::MatMulBase;

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept override {
    switch (index) {
      case 0:
        // want to allow bfloat16/float16/float32
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      case 1:
        // currently weights are float32 since we use CPU ort
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }

  ONNXTensorElementDataType GetOutputType(
    size_t index
  ) const noexcept override {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_OPERATORS_MATMUL
