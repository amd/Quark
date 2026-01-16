// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_LUT
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_LUT

#include <ryzenai/ryzen_mm.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace ryzenai::onnx_utils {

class LutKernel : public HybridKernel {
 public:
  LutKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~LutKernel();

  void Compute(OrtKernelContext* context);

 private:
  int64_t axis_ = 0;
  int64_t vocab_size_ = 0;
  int64_t embedding_dim_ = 0;
  bool use_external_data_ = false;
  std::string external_data_path_;

  // file IO
#ifdef _WIN32
  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  HANDLE hMapping_ = INVALID_HANDLE_VALUE;
  LPVOID pBuf_;
#endif

  // in memory case
  RyzenMM::BufferRef embedding_;

  bool mapped_lut_ = false;
  const std::uint8_t* external_lut_ptr_ = nullptr;
};

template <const char* kName>
struct LutBase : public HybridOperator<LutKernel, kName> {
  explicit LutBase(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<LutKernel, kName>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"hybrid_opt_embedding_mmap"};
  }

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

static const char kLut[] = "LUT";

struct Lut : LutBase<kLut> {
  using LutBase::LutBase;

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept override {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
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

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_LUT
