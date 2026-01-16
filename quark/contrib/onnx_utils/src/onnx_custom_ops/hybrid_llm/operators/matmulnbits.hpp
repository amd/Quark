// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

class AMDMatMulNBitsKernel;
class CoreLibKernel;
// forward-declaring these classes gave linker errors for some reason
// class OnnxTensorInfo;
// namespace DML_Ops {
//   class DMLOps;
// } // namespace DML_Ops

class MatMulNBitsKernel : public HybridKernel {
 public:
  MatMulNBitsKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~MatMulNBitsKernel();

  void Compute(OrtKernelContext* context);

 private:
  int64_t accuracy_level_ = 0;
  int64_t bits_;
  int64_t block_size_;
  int64_t k_;
  int64_t n_;
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  Ort::Op cpu_op_{nullptr};
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#if defined(ONNX_UTILS_ENABLE_CORELIB)
  std::unique_ptr<CoreLibKernel> npu_instance_;
#else
  std::unique_ptr<AMDMatMulNBitsKernel> npu_instance_;
#endif
#endif
};

template <const char* kName>
struct MatMulNBitsBase : public HybridOperator<MatMulNBitsKernel, kName> {
  explicit MatMulNBitsBase(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<MatMulNBitsKernel, kName>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"hybrid_opt_execution_mode"};
  }

  size_t GetInputTypeCount() const noexcept override { return 6; }

  size_t GetOutputTypeCount() const noexcept override { return 1; }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    if (index >= 4)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    else
      return INPUT_OUTPUT_REQUIRED;
  }

  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept override {
    return INPUT_OUTPUT_REQUIRED;
  }
};

static const char kMatMulNBits[] = "MatMulNBits";

struct MatMulNBits : MatMulNBitsBase<kMatMulNBits> {
  using MatMulNBitsBase::MatMulNBitsBase;

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept override {
    switch (index) {
      case 0:
      case 2:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      case 1:
      case 3:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }

  ONNXTensorElementDataType GetOutputType(
    size_t index
  ) const noexcept override {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
};

static const char kMatMulNBitsBf[] = "MatMulNBitsBf";
struct MatMulNBitsBf : MatMulNBitsBase<kMatMulNBitsBf> {
  using MatMulNBitsBase::MatMulNBitsBase;

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept override {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }

  ONNXTensorElementDataType GetOutputType(
    size_t index
  ) const noexcept override {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_MATMULNBITS
