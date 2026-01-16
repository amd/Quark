// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes a [m,k] x [k,n] = [m,n] matrix multiplication done batches*channels
// times. In short, the tensors are 4D but both outer dimensions are just layers
// of batching. The inner two dimensions are the ones that do the multiply.
struct MatMulIntParams {
  int32 m;         // Y's height.
  int32 n;         // Y's width.
  int32 k;         // Dot product dimension (A's width and B's height).
  int32 batches;   // The outer batch dimension of the tensors.
  int32 channels;  // The inner batch dimension of the tensors.
  bool hasScales;  // If the values in A and B should be scaled on dequantize
                   // and the values in Y on quantize.
  bool hasAZero;   // If there's a zero point for A.
  bool hasBZero;   // If there's a zero point for B.
  bool
    hasYZero;  // If there's a zero point for Y. Ignored if hasScales is false.
  bool quantizeARows;  // If A should have unique dequantize constants per row.
  bool
    quantizeBCols;  // If B should have unique dequantize constants per column.
  bool quantizeYRows;  // If Y should have unique quantize constants per row.

  // Tensor properties.
  hstring
    orderA;  // The dimension packing order for A. For example, "HW" or "NCWH".
  hstring
    orderB;  // The dimension packing order for B. For example, "HW" or "NCWH".
  hstring
    orderY;  // The dimension packing order for Y. For example, "HW" or "NCWH".
  int32 stridesA[4];  // Strides for A's dimensions, from outer to inner, in
                      // elements, -1 means packed.
  int32 stridesB[4];  // Strides for B's dimensions, from outer to inner, in
                      // elements, -1 means packed.
  int32 stridesY[4];  // Strides for Y's dimensions, from outer to inner, in
                      // elements, -1 means packed.
  hstring dataType;   // Which TensorProto DataType to use for A, B, and Y if
                      // hasScales is true (Int8 or Uint8).
};

// There are a lot of inputs to this operator, we keep track of their
// TensorVector offsets using this enum.
enum MatMulIntInputs : int32 {
  MatMulIntInputA = 0,   // TensorA
  MatMulIntInputAScale,  // TensorAScale, present if hasScales is true
  MatMulIntInputAZp,     // TensorAZeroPoint, optional
  MatMulIntInputB,       // TensorB
  MatMulIntInputBScale,  // TensorBScale, present if hasScales is true
  MatMulIntInputBZp,     // TensorBZeroPoint, optional
  MatMulIntInputYScale,  // TensorYScale, present if hasScales is true
  MatMulIntInputYZp,     // TensorYZeroPoint, optional, can only be used if
                         // hasScales is true
  MatMulIntInputCount
};

// =====================================================================================================================
// This class abstracts the DirectML Matrix Multiply Integer operator.
class MatMulIntOperator : public DmlOperator {
 public:
  explicit MatMulIntOperator(
    const Context& context, bool disableMetacmds, const MatMulIntParams& params
  );
  virtual ~MatMulIntOperator() {}

  // We use a custom input tensor ordering because we abstract two DML operators
  // using one class.
  virtual void BindInputs(
    CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
  ) const;

 private:
  const MatMulIntParams m_params;
  const DML_TENSOR_DATA_TYPE
    m_inputType;  // The A, AZp, B, BZp, and YZp tensors must use this type.
  const DML_TENSOR_DATA_TYPE m_outputType;  // The Y tensor must use this type.
};

}  // namespace ryzenai::onnx_utils
