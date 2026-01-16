// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes the parameters for the Subtraction operation.
struct SubParams {
  std::vector<int64_t> inputShapeA;
  std::vector<int64_t> inputShapeB;
  std::vector<int64_t> outputShape;
  hstring order = L"DHW";
  hstring dataType = L"Int64";
};

// =====================================================================================================================
// This class abstracts a DirectML subtraction operator.
class SubOperator : public DmlOperator {
 public:
  explicit SubOperator(
    const Context& context, bool disableMetacmds, const SubParams& params
  );
  virtual ~SubOperator() {}

 private:
  const SubParams m_params;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
};

}  // namespace ryzenai::onnx_utils
