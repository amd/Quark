// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/reduce.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
ReduceOperator::ReduceOperator(
  const Context& context,
  bool disableMetacmds,  // If metacommands should be disabled for this operator
  const ReduceParams& params
)
  : m_Reduce(params), m_dataType(StringToDataType(params.dataType)) {
  DmlTensorDesc dmlInputTensor = {};
  DmlTensorDesc dmlOutputTensor = {};

  const std::vector<int64> strides{-1, -1, -1};

  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"InputTensor", params.order, m_dataType, m_Reduce.inputShape, strides
  ));
  m_outputTensorDescVec.emplace_back(CreateTensorDesc(
    L"OutputTensor", params.order, m_dataType, m_Reduce.inputShape, strides
  ));

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlInputTensor);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlOutputTensor);

  DML_REDUCE_OPERATOR_DESC dmlDesc = {};
  dmlDesc.Function = DML_REDUCE_FUNCTION_SUM;
  dmlDesc.InputTensor = &dmlInputTensor.desc;
  dmlDesc.OutputTensor = &dmlOutputTensor.desc;
  dmlDesc.Axes = m_Reduce.axes.data();
  dmlDesc.AxisCount = m_Reduce.axes.size();

  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_REDUCE, &dmlDesc};
  m_operator = context.CompileOperator(
    &opDesc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
