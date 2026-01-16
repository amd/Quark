// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/gather.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

//// Define defaults and parameter layout metadata for GatherParams
// static const GatherParams GatherDefaults = {
//   2,           // batch
//   2,           // qSeq
//   L"DHW",      // order
//   L"Float16",  // dataType
// };

// =====================================================================================================================
GatherOperator::GatherOperator(
  const Context& context,
  bool disableMetacmds,  // If metacommands should be
                         // disabled for this operator.
  const GatherParams& params
)
  : m_gather(params), m_dataType(StringToDataType(params.dataType)) {
  m_gather.indexDimensions = params.indicesShape.size();

  std::vector<int64_t> dataDimensions = params.inputShape;
  std::vector<int64_t> indicesDimensions = params.indicesShape;
  std::vector<int64_t> outputDimensions = params.outputShape;

  DmlTensorDesc dmlInputTensor = {};
  DmlTensorDesc dmlIndicesTensor = {};
  DmlTensorDesc dmlOutputTensor = {};

  const std::vector<int64> size{params.batch, 2, 2};
  const std::vector<int64> strides{-1, -1, -1};

  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"InputTensor", params.order, m_dataType, size, strides)
  );
  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"IndicesTensor", params.order, m_dataType, size, strides)
  );
  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"OutputTensor", params.order, m_dataType, size, strides)
  );

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlInputTensor);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlIndicesTensor);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlOutputTensor);

  // uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelCreationContext,
  // m_inputTensorDescs.front().GetDimensionCount());

  DML_GATHER_OPERATOR_DESC dmlDesc = {};
  dmlDesc.InputTensor = &dmlInputTensor.desc;
  dmlDesc.IndicesTensor = &dmlIndicesTensor.desc;
  dmlDesc.OutputTensor = &dmlOutputTensor.desc;
  dmlDesc.Axis = m_gather.axis;
  dmlDesc.IndexDimensions = m_gather.indexDimensions;

  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_GATHER, &dmlDesc};

  m_operator = context.CompileOperator(
    &opDesc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
