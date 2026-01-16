// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract convolution operation.
struct ConvParams {
  int32 c;        // Number of channels.
  int32 n;        // Number of batches.
  int32 k;        // Number of features.
  int32 groups;   // Split c and k into this many filter groups.
  int32 dims;     // Number of convolution volume dimensions, must be 2 or 3.
  bool bias;      // The optional bias input tensor is present.
  bool backward;  // If these parameters describe an operator that acts like a
                  // backward convolution.
  bool
    crossCorrelation;  // If this operator computes a cross correlation instead
                       // of a convolution. Almost all convolutions defined by
                       // CNN frameworks are actually cross correlations.

  // 2D or 3D convolution volume parameters.
  int32 inputSize[3];   // Input depth, height, and width.
  int32 filterSize[3];  // Filter depth, height, and width.
  int32 startPad[3];    // Zero padding added to the beginning of the input in
                        // each dimension.
  int32
    endPad[3];  // Zero padding added to the end of the input in each dimension.
  int32 outPad[3];  // Zero padding at the end of our output in each dimension.
                    // (also called a/adj)
  int32 stride[3];  // Step size between filter evaluations in each dimension.
  int32 dilation[3];  // Step size between filter elements in each dimension.

  // Tensor properties.
  hstring orderX;  // The dimension packing order for X. For example, "NCHW" or
                   // "NCDHW".
  hstring orderW;  // The dimension packing order for W. For example, "NCHW" or
                   // "NCDHW".
  hstring orderY;  // The dimension packing order for Y. For example, "NCHW" or
                   // "NCDHW".
  int32 stridesX[5];  // Strides for the input dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 stridesW[5];  // Strides for the weight dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 stridesY[5];  // Strides for the output dimensions, from outer to inner,
                      // in elements, -1 means packed.
  int32 strideB;      // Stride for the 1D bias tensor.

  hstring dataType;  // Which TensorProto DataType to use for our tensors (e.g.,
                     // Float).
  hstring activationFunc;  // The activation function or None if there is no
                           // activation function.
  float activationParam1;  // The function's first constant parameter.
  float activationParam2;  // The function's second constant parameter.
};

// =====================================================================================================================
// Defines the full set of parameters to a generic convolution. Used by
// ConvOperator::ComputeOutputs.
struct GenericConvParams {
  int32 c;       // Number of channels.
  int32 n;       // Number of batches.
  int32 k;       // Number of features.
  int32 groups;  // Split c and k into this many filter groups.
  int32 dims;    // Number of convolution volume dimensions, must be 2 or 3.
  int32 inputSize[3];   // Input depth, height, and width.
  int32 filterSize[3];  // Filter depth, height, and width.
  int32 outputSize[3];  // Output depth, height, and width.
  int32 startPad[3];    // Zero padding added to the beginning of the input
                        // dimensions.
  int32 endPad[3];  // Zero padding added to the end of the input dimensions.
  int32 outPad[3];  // Zero padding added to the end of the output dimensions.
  int32 convStride[3];    // Step size between convolutions in input dimensions.
  int32 inputStride[3];   // Step size between dot products in input dimensions.
                          // Adds zero padding in the input.
  int32 filterStride[3];  // Step size between dot products in filter
                          // dimensions. Adds zero padding in the filter.
  bool bias;              // The optional bias input tensor is present.
  bool swapCK;  // The filter c/k dimensions are swapped (the second one is
                // still split into groups).
  bool reverseFilter;  // The filter x/y/z dimensions are accessed in reverse
                       // order.
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class ConvOperator : public DmlOperator {
 public:
  explicit ConvOperator(
    const Context& context, bool disableMetacmds, const ConvParams& params
  );
  virtual ~ConvOperator() {}

  //// Uses a simple reference function to compute the outputs of this operator.
  /// Intended for validation only.
  // virtual void ComputeOutputs(const TensorVector& inputs, TensorVector*
  // pOutputs) const override;

 private:
  void ValidateConvParams(const ConvParams& params) const;

  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  GenericConvParams m_conv;
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
