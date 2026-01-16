// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/norm.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
NormOperator::NormOperator(const Context& pContext, const NormParams& params)
  : m_norm(params), m_dataType(StringToDataType(params.dataType)) {
  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::wstringstream ss;
    ss << "Norm only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  const uint32_t batchSize =
    m_norm.inputShape[0] == -1 ? 1 : m_norm.inputShape[0];
  const uint32_t sequenceLength =
    m_norm.inputShape.size() == 3
      ? (m_norm.inputShape[1] == -1 ? 1 : m_norm.inputShape[1])
      : 1;

  const uint32_t hiddenSize = m_norm.inputShape.back();

  const std::vector<int64> inputTensorShape = {
    batchSize, sequenceLength, hiddenSize
  };
  const std::vector<int64> vectorShape = {1, 1, m_norm.scaleShape};
  const std::vector<int64> scalarShape = {1, 1, 1};
  const std::vector<int64> strides{-1, -1, -1};
  const std::vector<int64> biasStrides = {0, 0, 1};

  DmlTensorDesc dmlTensorReshape1 = {};
  DmlTensorDesc dmlTmpSlrnOutput = {};
  DmlTensorDesc dmlTensorX = {};
  DmlTensorDesc dmlTensorS = {};
  DmlTensorDesc dmlTensorB = {};
  DmlTensorDesc dmlTensorY = {};

  auto seqLenAfterReshape = hiddenSize / params.scaleShape;
  const std::vector<int64> reshapeTensor = {
    batchSize, sequenceLength * seqLenAfterReshape, params.scaleShape
  };

  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"X", L"DHW", m_dataType, reshapeTensor, strides)
  );

  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"Y", L"DHW", m_dataType, reshapeTensor, strides)
  );

  if (m_norm.hasScale) {
    m_inputTensorDescVec.emplace_back(
      CreateTensorDesc(L"scale", L"DHW", m_dataType, vectorShape, strides)
    );
  }

  if (m_norm.hasBias) {
    m_inputTensorDescVec.emplace_back(
      CreateTensorDesc(L"bias", L"DHW", m_dataType, vectorShape, strides)
    );
  }

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorX);

  if (m_norm.hasScale) {
    ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlTensorS);
  }
  if (m_norm.hasBias) {
    ConvertTensorDesc(*m_inputTensorDescVec[2], &dmlTensorB);
  }
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorY);

  std::vector<const DML_OPERATOR_DESC*> opDescs;

  uint32_t dmlDimCount = m_inputTensorDescVec[0]->dims.size();
  if (m_norm.onnxAxis < 0) {
    m_norm.onnxAxis += m_norm.onnxDimCount;
  }
  uint32_t absoluteAxis = static_cast<uint32_t>(m_norm.onnxAxis);
  assert(absoluteAxis < m_norm.onnxDimCount);
  std::vector<uint32_t> onnxAxes(
    static_cast<size_t>(dmlDimCount) - static_cast<size_t>(absoluteAxis)
  );

  std::array<uint32_t, 1> axes = {3};

  // Build the main DirectML operator descriptor.
  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC dmlDesc = {};
  dmlDesc.InputTensor = &dmlTensorX.desc;
  dmlDesc.ScaleTensor = m_norm.hasScale ? &dmlTensorS.desc : nullptr;
  dmlDesc.BiasTensor = nullptr;
  dmlDesc.OutputTensor = &dmlTensorY.desc;
  dmlDesc.UseMean = m_norm.UseMean;  // false for simplified Layer Norm
  dmlDesc.UseVariance = m_norm.UseVariance;
  dmlDesc.Axes = axes.data();
  dmlDesc.AxisCount = axes.size();
  dmlDesc.Epsilon = params.epsilon;

  // Finally, compile the operator.
  DML_OPERATOR_DESC slrnDesc = {};
  slrnDesc.Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2;
  slrnDesc.Desc = &dmlDesc;

  opDescs.push_back(&slrnDesc);

  // Construct the graph
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

  std::vector<CComPtr<IDMLOperator>> dmlOperators(
    static_cast<uint32_t>(opDescs.size())
  );

  DML_INPUT_GRAPH_EDGE_DESC inputGraphEdge = {};
  inputGraphEdge.GraphInputIndex = 0;
  inputGraphEdge.ToNodeIndex = 0;
  inputGraphEdge.ToNodeInputIndex = 0;
  inputEdges.push_back(inputGraphEdge);

  DML_INPUT_GRAPH_EDGE_DESC inputScaleEdge = {};
  inputGraphEdge.GraphInputIndex = 1;
  inputGraphEdge.ToNodeIndex = 0;
  inputGraphEdge.ToNodeInputIndex = 1;
  inputEdges.push_back(inputGraphEdge);

  DML_OUTPUT_GRAPH_EDGE_DESC outputGraphEdge = {};
  outputGraphEdge.FromNodeIndex = 0;
  outputGraphEdge.FromNodeOutputIndex = 0;
  outputGraphEdge.GraphOutputIndex = 0;
  outputEdges.push_back(outputGraphEdge);

  HRESULT status = S_OK;
  for (size_t i = 0; i < opDescs.size(); ++i) {
    status = pContext.DmlDevice()->CreateOperator(
      opDescs[i], IID_PPV_ARGS(&dmlOperators[i])
    );

    if (FAILED(status)) {
      throw std::runtime_error("Failed to compile a DirectML operator.");
    }
  }

  CompileGraph(
    pContext, dmlOperators, inputEdges, intermediateEdges, outputEdges
  );
}

}  // namespace ryzenai::onnx_utils
