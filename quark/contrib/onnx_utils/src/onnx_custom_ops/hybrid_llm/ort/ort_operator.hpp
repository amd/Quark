// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <map>

#include "execution_provider.hpp"
#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

class OrtOperator : ExecutionProviderExtensions {
 public:
  OrtOperator();
  ~OrtOperator();

 protected:
  void createOp(
    const OrtKernelInfo* info, const char* op_name, const char* domain,
    int version,
    std::map<std::string, ONNXTensorElementDataType> type_constraints,
    std::vector<Ort::OpAttr> attrs, size_t input_count, size_t output_count
  );

  bool isInitialized() const;

  void invokeOp(
    const OrtKernelContext* context, std::vector<const OrtValue*> input,
    std::vector<OrtValue*> output
  );

  Ort::MemoryInfo memory_info_;

 private:
  Ort::Op op_{nullptr};
  std::shared_ptr<CPUGate::Op> ep_op_;
};

}  // namespace ryzenai::onnx_utils
