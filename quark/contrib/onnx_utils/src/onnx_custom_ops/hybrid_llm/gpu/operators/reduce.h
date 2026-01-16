// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract Reduce operation.
struct ReduceParams {
  int32 batch = 1;  // Batches
  // Strides, from outer to inner, in elements, -1 means packed Tensor
  // properties.
  int32 strides[3] = {-1, -1, -1};
  std::vector<uint32_t> axes;       // axes along which to apply reduce operator
  std::vector<int64_t> inputShape;  // The shape of the input tensor.
  DML_REDUCE_FUNCTION reduceFunction = DML_REDUCE_FUNCTION_SUM;
  // The dimension packing order for A. For example, "HW" or "DWH".
  hstring order = L"DHW";
  // Which TensorProto DataType to use for our  tensors (e.g., Float).
  hstring dataType = L"Float";
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class ReduceOperator : public DmlOperator {
 public:
  explicit ReduceOperator(
    const Context& context, bool disableMetacmds, const ReduceParams& params
  );
  virtual ~ReduceOperator() {}

 private:
  // All tensors must use this data type.
  const DML_TENSOR_DATA_TYPE m_dataType;
  ReduceParams m_Reduce;
};

}  // namespace ryzenai::onnx_utils
