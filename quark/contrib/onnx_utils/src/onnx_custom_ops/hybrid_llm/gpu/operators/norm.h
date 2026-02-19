// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract norm operation.
struct NormParams {
  std::vector<int32> inputShape;  // batch, sequence,hiddenSize.
  int64 scaleShape;
  uint32 onnxDimCount = 4;
  float epsilon = 0.0f;     // Epsilon value
  bool hasScale = true;     // The optional scale input tensor is present.
  bool hasBias = false;     // The optional bias input tensor is present.
  bool UseMean = false;     // false for SimplifiedLayerNorm
  bool UseVariance = true;  // default is true
  int64 onnxAxis = -1;
  hstring dataType = L"Float";  // Which TensorProto DataType to use for our
  std::vector<int32> shapeIn;
  std::vector<int32> shapeOut;
  bool isWtsFp32 = false;  // whether scale and bias are in fp32
  // tensors (e.g.,// Float).
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class NormOperator : public DmlOperator {
 public:
  explicit NormOperator(const Context& context, const NormParams& params);
  virtual ~NormOperator() {}

 private:
  // All tensors must use this data type.
  const DML_TENSOR_DATA_TYPE m_dataType;
  NormParams m_norm;
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
