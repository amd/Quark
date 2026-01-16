// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/SkipSimplifiedLayerNorm.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
SkipSimplifiedLayerNormOperator::SkipSimplifiedLayerNormOperator(
  const Context& context,
  bool
    disableMetacmds,  // If metacommands should be disabled for this operator.
  const SkipSimplifiedLayerNormParams& params
)
  : m_SkipSimplifiedNorm(params),
    m_dataType(StringToDataType(params.dataType)) {
  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::wstringstream ss;
    ss << "SkipSimplifiedLayerNorm only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  assert(
    m_SkipSimplifiedNorm.inputShape.size() == 2 ||
    m_SkipSimplifiedNorm.inputShape.size() == 3
  );

  const uint32_t batchSize = m_SkipSimplifiedNorm.inputShape[0] == -1
                               ? 1
                               : m_SkipSimplifiedNorm.inputShape[0];
  const uint32_t sequenceLength = m_SkipSimplifiedNorm.inputShape.size() == 3
                                    ? (m_SkipSimplifiedNorm.inputShape[1] == -1
                                         ? 1
                                         : m_SkipSimplifiedNorm.inputShape[1])
                                    : 1;
  const uint32_t hiddenSize = m_SkipSimplifiedNorm.inputShape.back();

  const std::vector<int64> inputTensorShape = {
    batchSize, sequenceLength, hiddenSize, 1
  };
  const std::vector<int64> vectorShape = {1, 1, hiddenSize, 1};
  const std::vector<int64> scalarShape = {1, 1, 1, 1};
  const std::vector<int64> strides{-1, -1, -1, -1};
  const std::vector<int64> biasStrides = {0, 0, 1, 0};

  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"InputTensor", L"DHW", m_dataType, inputTensorShape, strides
  ));
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"SkipTensor", L"DHW", m_dataType, inputTensorShape, strides
  ));

  if (m_SkipSimplifiedNorm.hasScale)
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"mvnScaleTensor", L"DHW", m_dataType, vectorShape, strides
    ));

  if (m_SkipSimplifiedNorm.hasBias)
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"mvnBiasTensor", L"DHW", m_dataType, vectorShape, strides
    ));

  if (m_SkipSimplifiedNorm.hasNonMVNBias)
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"NonMVNBiasTensor", L"DHW", m_dataType, vectorShape, biasStrides
    ));

  m_outputTensorDescVec.emplace_back(CreateTensorDesc(
    L"OutputTensor", L"DHW", m_dataType, inputTensorShape, strides
  ));
  m_outputTensorDescVec.emplace_back(CreateTensorDesc(
    L"SkipBiasSumOutputTensor", L"DHW", m_dataType, inputTensorShape, strides
  ));

  const float epsilon = m_SkipSimplifiedNorm.epsilon;
  std::array<uint32_t, 2> axes = {2, 3};

  DmlTensorDesc inputDesc = {};
  DmlTensorDesc skipDesc = {};
  DmlTensorDesc mvnScaleDesc = {};
  DmlTensorDesc mvnBiasDesc = {};
  DmlTensorDesc nonMVNBiasDesc = {};

  ConvertTensorDesc(*m_inputTensorDescVec[0], &inputDesc);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &skipDesc);

  if (m_SkipSimplifiedNorm.hasScale)
    ConvertTensorDesc(*m_inputTensorDescVec[2], &mvnScaleDesc);

  if (m_SkipSimplifiedNorm.hasBias)
    ConvertTensorDesc(*m_inputTensorDescVec[3], &mvnBiasDesc);

  if (m_SkipSimplifiedNorm.hasNonMVNBias)
    ConvertTensorDesc(*m_inputTensorDescVec[4], &nonMVNBiasDesc);

  DmlTensorDesc outputDesc = {};
  DmlTensorDesc inputSkipBiasSum = {};

  ConvertTensorDesc(*m_outputTensorDescVec[0], &outputDesc);
  ConvertTensorDesc(*m_outputTensorDescVec[1], &inputSkipBiasSum);

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC inputSkipAddDesc = {};
  inputSkipAddDesc.ATensor = &inputDesc.desc;
  inputSkipAddDesc.BTensor = &skipDesc.desc;
  inputSkipAddDesc.OutputTensor = &inputDesc.desc;
  DML_OPERATOR_DESC inputSkipAddOpDesc = {
    DML_OPERATOR_ELEMENT_WISE_ADD, &inputSkipAddDesc
  };

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC inputSkipBiasAddDesc = {};
  if (m_SkipSimplifiedNorm.hasNonMVNBias) {
    inputSkipBiasAddDesc.ATensor = &inputDesc.desc;
    inputSkipBiasAddDesc.BTensor = &nonMVNBiasDesc.desc;
    inputSkipBiasAddDesc.OutputTensor = &inputDesc.desc;
  }
  DML_OPERATOR_DESC inputSkipBiasAddOpDesc = {
    DML_OPERATOR_ELEMENT_WISE_ADD, &inputSkipBiasAddDesc
  };

  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC mvnDesc = {};
  mvnDesc.InputTensor = &inputDesc.desc;
  mvnDesc.ScaleTensor =
    m_SkipSimplifiedNorm.hasScale ? &mvnScaleDesc.desc : nullptr;
  mvnDesc.BiasTensor =
    m_SkipSimplifiedNorm.hasBias ? &mvnBiasDesc.desc : nullptr;
  mvnDesc.OutputTensor = &outputDesc.desc;
  mvnDesc.Axes = axes.data();
  mvnDesc.AxisCount = axes.size();
  mvnDesc.UseMean = !m_SkipSimplifiedNorm.UseMean;
  mvnDesc.UseVariance = m_SkipSimplifiedNorm.UseVariance;
  mvnDesc.Epsilon = epsilon;
  mvnDesc.FusedActivation = nullptr;

  DML_OPERATOR_DESC mvnOpDesc = {
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2, &mvnDesc
  };

  // Construct the graph
  std::vector<const DML_OPERATOR_DESC*> opDescs;
  opDescs.reserve(3);

  std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
  inputEdges.reserve(5);

  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
  intermediateEdges.reserve(2);

  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
  outputEdges.reserve(1);

  // Insert the Input + Skip operation into the graph
  opDescs.push_back(&inputSkipAddOpDesc);

  DML_INPUT_GRAPH_EDGE_DESC dataInputEdge = {};
  dataInputEdge.GraphInputIndex = 0;
  dataInputEdge.ToNodeIndex = 0;
  dataInputEdge.ToNodeInputIndex = 0;
  inputEdges.push_back(std::move(dataInputEdge));

  DML_INPUT_GRAPH_EDGE_DESC skipInputEdge = {};
  skipInputEdge.GraphInputIndex = 1;
  skipInputEdge.ToNodeIndex = 0;
  skipInputEdge.ToNodeInputIndex = 1;
  inputEdges.push_back(std::move(skipInputEdge));

  // Insert the InputSkip + Bias operation into the graph
  if (m_SkipSimplifiedNorm.hasNonMVNBias) {
    opDescs.push_back(&inputSkipBiasAddOpDesc);

    DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
    intermediateEdge.FromNodeIndex = 0;
    intermediateEdge.FromNodeOutputIndex = 0;
    intermediateEdge.ToNodeIndex = 1;
    intermediateEdge.ToNodeInputIndex = 0;
    intermediateEdges.push_back(std::move(intermediateEdge));

    DML_INPUT_GRAPH_EDGE_DESC biasInputEdge = {};
    biasInputEdge.GraphInputIndex = 4;
    biasInputEdge.ToNodeIndex = 1;
    biasInputEdge.ToNodeInputIndex = 1;
    inputEdges.push_back(std::move(biasInputEdge));

    if (inputSkipBiasSum.sizes.size() > 0) {
      DML_OUTPUT_GRAPH_EDGE_DESC inputSkipBiasSumEdge = {};
      inputSkipBiasSumEdge.FromNodeIndex = 1;
      inputSkipBiasSumEdge.FromNodeOutputIndex = 0;
      inputSkipBiasSumEdge.GraphOutputIndex = m_outputTensorDescVec.size() - 1;
      outputEdges.push_back(std::move(inputSkipBiasSumEdge));
    }
  } else if (inputSkipBiasSum.sizes.size() > 0) {
    DML_OUTPUT_GRAPH_EDGE_DESC inputSkipBiasSumEdge = {};
    inputSkipBiasSumEdge.FromNodeIndex = 0;
    inputSkipBiasSumEdge.FromNodeOutputIndex = 0;
    inputSkipBiasSumEdge.GraphOutputIndex = m_outputTensorDescVec.size() - 1;
    outputEdges.push_back(std::move(inputSkipBiasSumEdge));
  }

  // Insert the MVN operation into the graph
  opDescs.push_back(&mvnOpDesc);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
  intermediateEdge.FromNodeIndex = m_SkipSimplifiedNorm.hasNonMVNBias ? 1 : 0;
  intermediateEdge.FromNodeOutputIndex = 0;
  intermediateEdge.ToNodeIndex = m_SkipSimplifiedNorm.hasNonMVNBias ? 2 : 1;
  intermediateEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(std::move(intermediateEdge));

  DML_INPUT_GRAPH_EDGE_DESC gammaInputEdge = {};
  gammaInputEdge.GraphInputIndex = 2;
  gammaInputEdge.ToNodeIndex = m_SkipSimplifiedNorm.hasNonMVNBias ? 2 : 1;
  gammaInputEdge.ToNodeInputIndex = 1;
  inputEdges.push_back(std::move(gammaInputEdge));

  if (m_SkipSimplifiedNorm.hasBias != false) {
    DML_INPUT_GRAPH_EDGE_DESC betaInputEdge = {};
    betaInputEdge.GraphInputIndex = 3;
    betaInputEdge.ToNodeIndex = m_SkipSimplifiedNorm.hasBias ? 2 : 1;
    betaInputEdge.ToNodeInputIndex = 2;
    inputEdges.push_back(std::move(betaInputEdge));
  }

  DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
  outputEdge.GraphOutputIndex = 0;
  outputEdge.FromNodeIndex = m_SkipSimplifiedNorm.hasBias ? 2 : 1;
  outputEdge.FromNodeOutputIndex = 0;
  outputEdges.push_back(std::move(outputEdge));

  // need code from matmulnbits
  DML_GRAPH_DESC graphDesc = {};
  std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(
    static_cast<uint32_t>(opDescs.size())
  );
  std::vector<CComPtr<IDMLOperator>> dmlOperators(
    static_cast<uint32_t>(opDescs.size())
  );
  std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(
    static_cast<uint32_t>(opDescs.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(
    static_cast<uint32_t>(inputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(
    static_cast<uint32_t>(outputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(
    static_cast<uint32_t>(intermediateEdges.size())
  );

  // build the graph description from the above edges
  graphDesc.InputCount = static_cast<uint32_t>(inputEdges.size());
  graphDesc.OutputCount = static_cast<uint32_t>(outputEdges.size());
  graphDesc.NodeCount = static_cast<uint32_t>(opDescs.size());
  HRESULT status = S_OK;

  for (size_t i = 0; i < graphDesc.NodeCount; ++i) {
    // Create the operator.
    status = context.DmlDevice()->CreateOperator(
      opDescs[i], IID_PPV_ARGS(&dmlOperators[i])
    );
    dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{dmlOperators[i]};
    dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{
      DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]
    };
    if (FAILED(status)) {
      throw std::runtime_error("Failed to compile a DirectML operator.");
    }
  }

  graphDesc.Nodes = dmlGraphNodes.data();

  // set the input edges
  graphDesc.InputEdgeCount = static_cast<uint32_t>(inputEdges.size());
  for (size_t i = 0; i < graphDesc.InputEdgeCount; ++i) {
    dmlInputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &inputEdges[i]};
  }
  graphDesc.InputEdges = dmlInputEdges.data();

  // set the output edges
  graphDesc.OutputEdgeCount = graphDesc.OutputCount;
  for (size_t i = 0; i < graphDesc.OutputEdgeCount; ++i) {
    dmlOutputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &outputEdges[i]};
  }
  graphDesc.OutputEdges = dmlOutputEdges.data();

  // set the intermediate edges
  graphDesc.IntermediateEdgeCount =
    static_cast<uint32_t>(intermediateEdges.size());
  for (size_t i = 0; i < graphDesc.IntermediateEdgeCount; ++i) {
    dmlIntermediateEdges[i] = DML_GRAPH_EDGE_DESC{
      DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediateEdges[i]
    };
  }
  graphDesc.IntermediateEdges = dmlIntermediateEdges.data();

  CComPtr<IDMLDevice1> dmlDevice1;
  status = context.DmlDevice()->QueryInterface(IID_PPV_ARGS(&dmlDevice1));
  if (FAILED(status)) {
    throw std::runtime_error("Failed to acquire dml graph interface.");
  }

  DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_NONE;
  if (disableMetacmds) {
    executionFlags = DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
  }
  executionFlags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;

  status = dmlDevice1->CompileGraph(
    &graphDesc, executionFlags, IID_PPV_ARGS(&m_operator)
  );
  if (FAILED(status)) {
    throw std::runtime_error("Failed to compile a DirectML operator.");
  }
}

}  // namespace ryzenai::onnx_utils
