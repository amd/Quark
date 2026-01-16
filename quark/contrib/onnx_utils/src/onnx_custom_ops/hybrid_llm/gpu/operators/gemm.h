// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract Gemm operation {Y = alpha*A*B + beta*C}.
struct GemmParams {
  int32 m = 1;  // Y's and C's height.
  int32 n = 1;  // Y's and C's width.
  int32 k = 1;  // Dot product dimension (A's width and B's height).
  int32 batches =
    1;  // The outer dimension of the matrices. Gemm is executed once per batch.
  float alpha = 1.0f;   // Scalar multiplier for tensor A*B.
  float beta = 0.0f;    // Scalar multiplier for tensor C.
  bool transA = false;  // A is transposed in memory.
  bool transB = false;  // B is transposed in memory.
  bool hasC = false;    // If the optional C matrix is present.

  // Tensor properties.
  hstring orderA =
    L"DHW";  // The dimension packing order for A. For example, "HW" or "DWH".
  hstring orderB =
    L"DHW";  // The dimension packing order for B. For example, "HW" or "DWH".
  hstring orderC =
    L"DHW";  // The dimension packing order for C. For example, "HW" or "DWH".
  hstring orderY =
    L"DHW";  // The dimension packing order for Y. For example, "HW" or "DWH".
  int32 stridesA[3] = {-1, -1, -1};  // Strides for A's dimensions, from outer
                                     // to inner, in elements, -1 means packed.
  int32 stridesB[3] = {-1, -1, -1};  // Strides for B's dimensions, from outer
                                     // to inner, in elements, -1 means packed.
  int32 stridesC[3] = {-1, -1, -1};  // Strides for C's dimensions, from outer
                                     // to inner, in elements, -1 means packed.
  int32 stridesY[3] = {-1, -1, -1};  // Strides for Y's dimensions, from outer
                                     // to inner, in elements, -1 means packed.

  hstring dataType = L"Float";  // Which TensorProto DataType to use for our
                                // tensors (e.g., Float).
  hstring activationFunc = L"None";  // The activation function or None if there
                                     // is no activation function.
  float activationParam1 = 0.0f;     // The function's first constant parameter.
  float activationParam2 = 0.0f;  // The function's second constant parameter.
};

// =====================================================================================================================
// This class abstracts a DirectML GEMM operator.
class GemmOperator : public DmlOperator {
 public:
  explicit GemmOperator(
    const Context& context, bool disableMetacmds, const GemmParams& params
  );
  virtual ~GemmOperator() {}

  // Uses a simple reference function to compute the outputs of this operator.
  // Intended for validation only. virtual void ComputeOutputs(const
  // TensorVector& inputs, TensorVector* pOutputs) const override;

 private:
  const GemmParams m_gemm;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
