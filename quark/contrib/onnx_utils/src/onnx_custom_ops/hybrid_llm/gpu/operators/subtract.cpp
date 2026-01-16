// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/subtract.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Helper function to pad a shape with 1s to match the dimensions of a reference
// shape
std::vector<int64_t> PadShapeWithOnesForBroadCast(
  const std::vector<int64_t>& referenceShape,
  const std::vector<int64_t>& targetShape
) {
  std::vector<int64_t> paddedShape = targetShape;
  while (paddedShape.size() < referenceShape.size()) {
    paddedShape.insert(paddedShape.begin(), 1);
  }
  return paddedShape;
}

// =====================================================================================================================
SubOperator::SubOperator(
  const Context& context, bool disableMetacmds, const SubParams& params
)
  : m_params(params), m_dataType(StringToDataType(params.dataType)) {
  std::vector<int64_t> broadcastedInputShapeB =
    PadShapeWithOnesForBroadCast(m_params.inputShapeA, m_params.inputShapeB);

  DmlTensorDesc dmlInputTensorA = {};
  DmlTensorDesc dmlInputTensorB = {};
  DmlTensorDesc dmlOutputTensor = {};

  std::vector<int64_t> stridesA =
    std::vector<int64_t>(m_params.inputShapeA.size(), -1);
  std::vector<int64_t> stridesB = ComputeStrides(broadcastedInputShapeB);
  std::vector<int64_t> outputStrides =
    std::vector<int64_t>(m_params.outputShape.size(), -1);

  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"InputTensorA", m_params.order, m_dataType, m_params.inputShapeA, stridesA
  ));

  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"InputTensorB", m_params.order, m_dataType, m_params.inputShapeA, stridesB
  ));

  m_outputTensorDescVec.emplace_back(CreateTensorDesc(
    L"OutputTensor", m_params.order, m_dataType, m_params.outputShape,
    outputStrides
  ));

  // Convert the tensor descriptions to DirectML tensor descriptions
  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlInputTensorA);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlInputTensorB);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlOutputTensor);

  // Setup the DirectML subtraction operation
  DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC dmlDesc = {};
  dmlDesc.ATensor = &dmlInputTensorA.desc;
  dmlDesc.BTensor = &dmlInputTensorB.desc;
  dmlDesc.OutputTensor = &dmlOutputTensor.desc;

  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_ELEMENT_WISE_SUBTRACT, &dmlDesc};

  // Compile the operator
  m_operator = context.CompileOperator(
    &opDesc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
