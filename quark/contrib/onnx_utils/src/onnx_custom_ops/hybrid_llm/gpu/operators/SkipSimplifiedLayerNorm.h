// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract norm operation.
struct SkipSimplifiedLayerNormParams {
  std::vector<int32> inputShape;  // Input batch, channels, height, and width.
  float epsilon = 0.0f;           // Epsilon value
  bool hasScale = true;           // The optional scale input tensor is present.
  bool hasBias = false;           // The optional bias input tensor is present.
  bool UseMean = false;           // false for SimplifiedLayerNorm
  bool UseVariance = true;        // default is true
  bool hasNonMVNBias = false;     // Bias for Add op in the graph
  int32 outputCount = 2;          // Number of output tensors.
  hstring dataType = L"Float";    // Which TensorProto DataType to use for our
                                  // tensors (e.g.,// Float).
  bool isWtsFp32 = false;         // if weights are in fp32
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class SkipSimplifiedLayerNormOperator : public DmlOperator {
 public:
  explicit SkipSimplifiedLayerNormOperator(
    const Context& context, bool disableMetacmds,
    const SkipSimplifiedLayerNormParams& params
  );
  virtual ~SkipSimplifiedLayerNormOperator() {}

 private:
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  SkipSimplifiedLayerNormParams m_SkipSimplifiedNorm;
};

}  // namespace ryzenai::onnx_utils
