// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract Gather operation.
struct GatherParams {
  uint32 axis = 2;  // The axis dimension of InputTensor to gather on, ranging
                    // [0, *InputTensor.DimensionCount*)
  uint32 indexDimensions =
    2;  // The number of actual index dimensions within the IndicesTensor
  int32 batch = 1;  // Batches
  int32 strides[3] = {
    -1, -1, -1
  };  // Strides, from outer to inner, in elements, -1 means packed.
  // Tensor properties.
  std::vector<int64_t> inputShape;    // The shape of the input tensor.
  std::vector<int64_t> indicesShape;  // The shape of the indices tensor.
  std::vector<int64_t> outputShape;   // The shape of the output tensor.
  hstring order =
    L"DHW";  // The dimension packing order for A. For example, "HW" or "DWH".
  hstring dataType = L"Float";  // Which TensorProto DataType to use for our
                                // tensors (e.g., Float).
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class GatherOperator : public DmlOperator {
 public:
  explicit GatherOperator(
    const Context& context, bool disableMetacmds, const GatherParams& params
  );
  virtual ~GatherOperator() {}

 private:
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  GatherParams m_gather;
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
