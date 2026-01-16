// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract Gemm operation {Y = alpha*A*B + beta*C}.
struct MatMulNBitsParams {
  int32 m = 1;          // Y's and C's height.
  int32 n = 1;          // Y's and C's width.
  int32 k = 1;          // Dot product dimension (A's width and B's height).
  int32 batches = 1;    // The outer dimension of the matrices. Gemm is executed
                        // once per batch.
  bool transA = false;  // A is transposed in memory.
  bool transB = true;   // B is transposed in memory.
  bool hasC = false;    // if bias is used.
  bool hasZeroPoint = false;  // if zero point is used

  // Tensor properties.
  hstring orderA =
    L"DHW";  // The dimension packing order for A. For example, "HW" or "DWH".
  hstring orderB =
    L"DHW";  // The dimension packing order for B. For example, "HW" or "DWH".
  hstring orderY =
    L"DHW";  // The dimension packing order for Y. For example, "HW" or "DWH".
  int32 stridesA[3] = {-1, -1, -1};  // Strides for A's dimensions, from outer
                                     // to inner, in elements, -1 means packed.
  int32 stridesB[3] = {-1, -1, -1};  // Strides for B's dimensions, from outer
                                     // to inner, in elements, -1 means packed.
  int32 stridesY[3] = {-1, -1, -1};  // Strides for Y's dimensions, from outer
                                     // to inner, in elements, -1 means packed.

  hstring dataType = L"Float16";  // Which TensorProto DataType to use for our
                                  // tensors (e.g., Float).
  hstring quantizedA = L"None";
  hstring quantizedB = L"Uint4";
  int32 quantizedBlockA = 1;
  int32 quantizedBlockB = 32;
};

// =====================================================================================================================
// This class abstracts a DirectML GEMM operator.
class MatMulNBitsOperator : public DmlOperator {
 public:
  explicit MatMulNBitsOperator(
    const Context& context, const MatMulNBitsParams& params,
    bool disableCompile = false
  );
  virtual ~MatMulNBitsOperator() {}
  void CreateMatmulNBitOperator(
    const Context& pContext, std::vector<CComPtr<IDMLOperator>>& dmlOperators,
    std::shared_ptr<DmlTensorDesc>& dmlDescInFromGQA
  );

 private:
  void SharedInit();
  const MatMulNBitsParams m_gemm;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  const DML_TENSOR_DATA_TYPE m_quantDataType;  // Quantized datatype
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
