// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_QMOE
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_QMOE

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace ryzenai::onnx_utils {

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
class AMDQMoEKernel;
#endif

class QmoeKernel : public HybridKernel {
 public:
  QmoeKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~QmoeKernel();

  void Compute(OrtKernelContext* context);

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
 private:
  std::unique_ptr<AMDQMoEKernel> npu_instance_;
#endif
};

template <const char* kName>
struct QmoeBase : public HybridOperator<QmoeKernel, kName> {
  explicit QmoeBase(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<QmoeKernel, kName>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"hybrid_opt_qmoe_bind_all"};
  }

  size_t GetInputTypeCount() const noexcept override { return 16; }

  size_t GetOutputTypeCount() const noexcept override { return 1; }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
    size_t index
  ) const noexcept override {
    switch (index) {
      case 0:   // input
      case 1:   // router probs
      case 2:   // fc1 experts weights
      case 3:   // fc1 experts scales
      case 4:   // fc1 experts bias
      case 5:   // fc2 experts weights
      case 6:   // fc2 experts scales
      case 7:   // fc2 experts bias
      case 14:  // packed fc1 expert
      case 15:  // packed fc2 expert
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
      case 8:   // fc3 expert weights
      case 9:   // fc3 expert scales
      case 10:  // fc3 expert bias
      case 11:  // fc1 zero point
      case 12:  // fc2 zero point
      case 13:  // fc3 zero point
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
      default:
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    }
  }

  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept override {
    return INPUT_OUTPUT_REQUIRED;
  }
};

static const char kQmoe[] = "QMoEBf";

struct QmoeBf : QmoeBase<kQmoe> {
  using QmoeBase::QmoeBase;

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept override {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      default:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }

  ONNXTensorElementDataType GetOutputType(
    size_t index
  ) const noexcept override {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_QMOE
