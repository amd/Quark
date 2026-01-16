// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract intger convolution operation. It is only as abstract as
// required to fit all DirectML integer convolution layers. Note that it lacks
// some features supported by the non-integer convolution path.
struct ConvIntParams {
  int32 c;       // Number of channels.
  int32 n;       // Number of batches.
  int32 k;       // Number of features.
  int32 groups;  // Split c and k into this many filter groups.
  bool bias;  // The optional bias input tensor is present. Requires hasScales.
  bool hasScales;   // If the inputs, filter, and outputs should be scaled on
                    // quantize, dequantize and permits bias.
  bool hasXZero;    // If there's a zero point for the input tensor.
  bool hasWZero;    // If there's a zero point for the filter tensor.
  bool hasYZero;    // If there's a zero point for the input tensor.
  bool quantWPerK;  // If the filter should have unique dequantize constants per
                    // feature.

  // 2D convolution area parameters.
  int32 inputSize[2];   // Input height and width.
  int32 filterSize[2];  // Filter height and width.
  int32 startPad[2];    // Zero padding added to the beginning of the input in
                        // each dimension.
  int32
    endPad[2];  // Zero padding added to the end of the input in each dimension.
  int32 stride[2];    // Step size between filter evaluations in each dimension.
  int32 dilation[2];  // Step size between filter elements in each dimension.

  // Tensor properties.
  hstring orderX;     // The dimension packing order for X. For example, "NCHW".
  hstring orderW;     // The dimension packing order for W. For example, "NCHW".
  hstring orderY;     // The dimension packing order for Y. For example, "NCHW".
  int32 stridesX[4];  // Strides for the input dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 stridesW[4];  // Strides for the weight dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 stridesY[4];  // Strides for the output dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 strideB;      // Stride for the 1D bias tensor.
  hstring dataType;  // Which TensorProto DataType to use for our tensors (e.g.,
                     // Float).
};

// There are a lot of inputs to this operator, we keep track of their
// TensorVector offsets using this enum. Yes, there are some inputs with
// "Output" in their name.
enum ConvIntInputs : int32 {
  ConvIntInputX = 0,   // TensorX
  ConvIntInputXScale,  // TensorXScale, present if hasScales is true
  ConvIntInputXZp,     // TensorXZeroPoint, optional
  ConvIntInputW,       // TensorW
  ConvIntInputWScale,  // TensorWScale, present if hasScales is true
  ConvIntInputWZp,     // TensorWZeroPoint, optional
  ConvIntInputB,  // TensorB, optional, can only be used if hasScales is true
  ConvIntInputYScale,  // TensorYScale, present if hasScales is true
  ConvIntInputYZp,  // TensorYZeroPoint, optional, can only be used if hasScales
                    // is true
  ConvIntInputCount
};

// =====================================================================================================================
// This class abstracts the two DirectML integer convolution operators.
class ConvIntOperator : public DmlOperator {
 public:
  explicit ConvIntOperator(
    const Context& context, bool disableMetacmds, const ConvIntParams& params
  );
  virtual ~ConvIntOperator() {}

  // We use a custom input tensor ordering because we abstract two DML operators
  // using one class.
  virtual void BindInputs(
    CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
  ) const;

 private:
  const ConvIntParams m_params;
  const DML_TENSOR_DATA_TYPE m_inputType;  // All inputs except the scale and
                                           // bias tensors must use this type.
  const DML_TENSOR_DATA_TYPE
    m_outputType;  // The Output tensor must use this type.
};

}  // namespace ryzenai::onnx_utils
