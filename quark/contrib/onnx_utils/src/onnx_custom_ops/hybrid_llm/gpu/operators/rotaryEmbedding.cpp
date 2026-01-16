// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/rotaryEmbedding.h"

#include <comdef.h>
#include <windows.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "context.h"
#include "tensor.h"

std::string GetErrorMessage(HRESULT hr) {
  _com_error err(hr);
  return std::string(err.ErrorMessage());
}

namespace ryzenai::onnx_utils {

void PrintTensorDesc(
  const DmlTensorDesc& tensorDesc, const std::string& tensorName
) {
  std::cout << "Tensor: " << tensorName << std::endl;
  std::cout << "Data Type: " << tensorDesc.desc.Type << std::endl;
  std::cout << "Dimensions: ";
  for (size_t i = 0; i < tensorDesc.sizes.size(); ++i) {
    std::cout << tensorDesc.sizes[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Strides: ";
  for (size_t i = 0; i < tensorDesc.strides.size(); ++i) {
    std::cout << tensorDesc.strides[i] << " ";
  }
  std::cout << std::endl;
}

// =====================================================================================================================
void RotaryEmbeddingOperator::CreateRotaryEmbeddingOperator(
  const Context& pContext, std::vector<CComPtr<IDMLOperator>>& dmlOperators,
  std::shared_ptr<DmlTensorDesc>& dmlDescInputOutput
) {
  // Now convert them to the DirectML representation.
  DmlTensorDesc dmlTensorInput = {};
  DmlTensorDesc dmlTensorPositionIds = {};
  DmlTensorDesc dmlTensorCosCache = {};
  DmlTensorDesc dmlTensorSinCache = {};

  // Output
  DmlTensorDesc dmlTensorOutput = {};

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorInput);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlTensorPositionIds);
  ConvertTensorDesc(*m_inputTensorDescVec[2], &dmlTensorCosCache);
  ConvertTensorDesc(*m_inputTensorDescVec[3], &dmlTensorSinCache);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorOutput);

  const bool interleaved = m_ropeParams.interleaved;

  std::vector<int64> inputOutputShape = {
    m_ropeParams.batchSize, m_ropeParams.sequenceLength, m_ropeParams.numHeads,
    m_ropeParams.headSize
  };

  std::vector<int64> splitInputOutputShape1 = {
    m_ropeParams.batchSize, m_ropeParams.sequenceLength, m_ropeParams.numHeads,
    m_ropeParams.rotaryEmbeddingDim
  };

  std::vector<int64> splitInputOutputShape2 = {
    m_ropeParams.batchSize, m_ropeParams.sequenceLength, m_ropeParams.numHeads,
    m_ropeParams.headSize - m_ropeParams.rotaryEmbeddingDim
  };

  const std::vector<int64> autoStrides{-1, -1, -1, -1};

  DmlTensorDesc inputOutputTensorDesc = {};
  std::shared_ptr<TensorDesc> tmpio = CreateTensorDesc(
    L"IOTensor", m_ropeParams.order, m_dataType, inputOutputShape, autoStrides
  );
  ConvertTensorDesc(*tmpio, &inputOutputTensorDesc);

  DmlTensorDesc dmlSplitInputOutput1 = {};
  std::shared_ptr<TensorDesc> splitInputOutputTensorDesc1 = CreateTensorDesc(
    L"splitInputOutputTensor1", m_ropeParams.order, m_dataType,
    splitInputOutputShape1, autoStrides
  );
  ConvertTensorDesc(*splitInputOutputTensorDesc1, &dmlSplitInputOutput1);

  DmlTensorDesc dmlSplitInputOutput2 = {};
  std::shared_ptr<TensorDesc> splitInputOutputTensorDesc2 = CreateTensorDesc(
    L"splitInputOutputTensor2", m_ropeParams.order, m_dataType,
    splitInputOutputShape2, autoStrides
  );
  ConvertTensorDesc(*splitInputOutputTensorDesc2, &dmlSplitInputOutput2);

  // if we want to use an incoming DmlTensorDesc instead of local one
  // this is needed if output of Rope is used as a input to another op like GQA
  // while constructing a complete graph

  std::shared_ptr<DmlTensorDesc> stridedInputOutputTensorDesc;
  std::shared_ptr<TensorDesc> stdtmpio = CreateTensorDesc(
    L"IOTensor", m_ropeParams.order, m_dataType, inputOutputShape, autoStrides
  );

  // dmlDescInputOutput : Check for null
  if (dmlDescInputOutput) {
    stridedInputOutputTensorDesc = dmlDescInputOutput;
  } else {
    stridedInputOutputTensorDesc = std::make_shared<DmlTensorDesc>();
  }

  ConvertTensorDesc(*stdtmpio, stridedInputOutputTensorDesc.get());

  std::array<DML_TENSOR_DESC, 2> splitTensorDescs = {
    dmlSplitInputOutput1.desc, dmlSplitInputOutput2.desc
  };

  // Check if partial rotary embedding is required
  m_isPartialRotaryEmbedding =
    (m_ropeParams.headSize > m_ropeParams.rotaryEmbeddingDim);

  DML_SPLIT_OPERATOR_DESC splitInputDesc{};
  DML_OPERATOR_DESC splitInputDmlDesc{};

  // Handle partial rotary embedding
  if (m_isPartialRotaryEmbedding) {
    splitInputDesc.InputTensor = &inputOutputTensorDesc.desc;
    splitInputDesc.OutputCount = splitTensorDescs.size();
    splitInputDesc.OutputTensors = splitTensorDescs.data();
    splitInputDesc.Axis = inputOutputShape.size() - 1;
    splitInputDmlDesc.Type = DML_OPERATOR_SPLIT;
    splitInputDmlDesc.Desc = &splitInputDesc;
  }

  // Copy the input to preserve its real input shape in the graph without
  // reshaping it. This will disappear during DML's graph compilation phase.
  DML_SCALE_BIAS scaleBias = {1.0f, 0.0f};

  std::vector<int64> partialInputOutputShape = {
    m_ropeParams.batchSize, m_ropeParams.sequenceLength, m_ropeParams.numHeads,
    m_ropeParams.rotaryEmbeddingDim
  };

  DmlTensorDesc dmlPartialStridedInputOutput = {};
  std::shared_ptr<TensorDesc> partialStridedInputOutputTensorDesc =
    CreateTensorDesc(
      L"partialStridedInputOutputTensor", m_ropeParams.order, m_dataType,
      partialInputOutputShape, autoStrides
    );
  ConvertTensorDesc(
    *partialStridedInputOutputTensorDesc, &dmlPartialStridedInputOutput
  );

  DmlTensorDesc dmlPartialInputOutput = {};
  std::shared_ptr<TensorDesc> partialInputOutputTensorDesc = CreateTensorDesc(
    L"partialInputOutputTensor", m_ropeParams.order, m_dataType,
    partialInputOutputShape, autoStrides
  );
  ConvertTensorDesc(*partialInputOutputTensorDesc, &dmlPartialInputOutput);

  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC copyInputDesc{};
  copyInputDesc.InputTensor = &dmlPartialStridedInputOutput.desc;
  copyInputDesc.OutputTensor = &dmlPartialInputOutput.desc;
  copyInputDesc.ScaleBias = &scaleBias;
  const DML_OPERATOR_DESC copyInputDmlDesc = {
    DML_OPERATOR_ELEMENT_WISE_IDENTITY, &copyInputDesc
  };

  const uint32_t halfRoraryEmbeddingDim = m_ropeParams.rotaryEmbeddingDim / 2;

  // Split the input data into 2 equal parts
  DmlTensorDesc dmlTensorSplitInputData[2] = {};
  DmlTensorDesc dmlTensorJoinedData = {};
  DmlTensorDesc dmlTensorPartialInputData = {};

  ConvertTensorDesc(*m_interTensorDescVec[0], &dmlTensorSplitInputData[0]);
  ConvertTensorDesc(*m_interTensorDescVec[0], &dmlTensorSplitInputData[1]);
  ConvertTensorDesc(*m_interTensorDescVec[1], &dmlTensorJoinedData);
  ConvertTensorDesc(*m_interTensorDescVec[1], &dmlTensorPartialInputData);

  const DML_TENSOR_DESC splitInputTensors[2] = {
    dmlTensorSplitInputData[0].desc, dmlTensorSplitInputData[1].desc
  };

  DML_SPLIT_OPERATOR_DESC splitPartialInputDesc{};
  splitPartialInputDesc.InputTensor = &dmlTensorPartialInputData.desc;
  splitPartialInputDesc.OutputTensors = splitInputTensors;
  splitPartialInputDesc.OutputCount = 2;
  splitPartialInputDesc.Axis =
    interleaved
      ? static_cast<uint32_t>(m_interTensorDescVec[0]->dims.size()) - 1
      : static_cast<uint32_t>(m_interTensorDescVec[0]->dims.size()) - 2;

  const DML_OPERATOR_DESC splitPartialInputDmlDesc = {
    DML_OPERATOR_SPLIT, &splitPartialInputDesc
  };

  // Swap the 2 halves and join them together
  DML_JOIN_OPERATOR_DESC joinPartialInputDesc{};
  joinPartialInputDesc.InputTensors = splitInputTensors;
  joinPartialInputDesc.OutputTensor = &dmlTensorJoinedData.desc;
  joinPartialInputDesc.Axis = splitPartialInputDesc.Axis;
  joinPartialInputDesc.InputCount = 2;
  const DML_OPERATOR_DESC joinPartialInputDmlDesc = {
    DML_OPERATOR_JOIN, &joinPartialInputDesc
  };

  // Gather the cos/sin values based on the position ids
  DmlTensorDesc dmlTensorGatheredCosSin = {};
  ConvertTensorDesc(*m_interTensorDescVec[2], &dmlTensorGatheredCosSin);

  DML_GATHER_OPERATOR_DESC gatherCosSinDesc{};
  gatherCosSinDesc.InputTensor = &dmlTensorCosCache.desc;
  gatherCosSinDesc.IndicesTensor = &dmlTensorPositionIds.desc;
  gatherCosSinDesc.OutputTensor = &dmlTensorGatheredCosSin.desc;
  gatherCosSinDesc.Axis = 2;
  gatherCosSinDesc.IndexDimensions = 2;
  const DML_OPERATOR_DESC gatherCosSinDmlDesc{
    DML_OPERATOR_GATHER, &gatherCosSinDesc
  };

  // After gathering cos/sin, reshape and broadcast them to match the number of
  // heads of the input data
  DmlTensorDesc dmlTensorBroadcastedCosSin = {};
  ConvertTensorDesc(*m_interTensorDescVec[3], &dmlTensorBroadcastedCosSin);

  // Create a vector that contains the sign values {-1, 1}
  DmlTensorDesc dmlTensorSign = {};
  ConvertTensorDesc(*m_interTensorDescVec[4], &dmlTensorSign);

  DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC signRange{};
  signRange.OutputTensor = &dmlTensorSign.desc;
  if (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16) {
    const uint16_t valueStart = Float16(-1.0f).u16;  // -1.0 in float16!
    const uint16_t valueDelta = Float16(2.0f).u16;   // 2.0 in float16!
    // memcpy(signRange.ValueStart.Bytes, &valueStart, sizeof(valueStart));
    // memcpy(signRange.ValueDelta.Bytes, &valueDelta, sizeof(valueDelta));

    signRange.ValueStart.UInt16 = valueStart;
    signRange.ValueDelta.UInt16 = valueDelta;

    signRange.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else {
    if (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) {
      throw std::runtime_error(
        "RotaryEmbedding only supports Float16 and Float32"
      );
    }
    signRange.ValueStart.Float32 = -1.0f;
    signRange.ValueDelta.Float32 = 2.0f;
    signRange.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  }
  const DML_OPERATOR_DESC signRangeDmlDesc = {
    DML_OPERATOR_FILL_VALUE_SEQUENCE, &signRange
  };

  // tile Sign tensor to match rotated input
  DmlTensorDesc dmlTensorBroadcastedSign = {};
  ConvertTensorDesc(*m_interTensorDescVec[5], &dmlTensorBroadcastedSign);

  // Multiply the broadcasted sign values with the rotated input
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulSignDesc{};
  mulSignDesc.ATensor = &dmlTensorJoinedData.desc;
  mulSignDesc.BTensor = &dmlTensorBroadcastedSign.desc;
  // mulSignDesc.BTensor = &dmlTensorJoinedData.desc;
  mulSignDesc.OutputTensor = &dmlTensorJoinedData.desc;
  const DML_OPERATOR_DESC mulSignDmlDesc = {
    DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulSignDesc
  };

  // Multiply the non-rotated data with the cos and the rotated data with the
  // sin
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulCosSinDesc{};
  mulCosSinDesc.ATensor = &dmlTensorJoinedData.desc;
  mulCosSinDesc.BTensor = &dmlTensorBroadcastedCosSin.desc;
  mulCosSinDesc.OutputTensor = &dmlTensorJoinedData.desc;
  const DML_OPERATOR_DESC mulCosSinDmlDesc = {
    DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulCosSinDesc
  };

  // Add the multiplied cos and sin values together
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc{};
  addDesc.ATensor = &dmlPartialInputOutput.desc;
  addDesc.BTensor = &dmlPartialInputOutput.desc;
  addDesc.OutputTensor = &dmlPartialStridedInputOutput.desc;
  const DML_OPERATOR_DESC addDmlDesc = {
    DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc
  };

  // Handle partial rotary embedding
  DML_JOIN_OPERATOR_DESC joinOutputDesc{};
  DML_OPERATOR_DESC joinOutputDmlDesc{};
  if (m_isPartialRotaryEmbedding) {
    joinOutputDesc.InputCount = splitTensorDescs.size();
    joinOutputDesc.InputTensors = splitTensorDescs.data();
    joinOutputDesc.OutputTensor = &inputOutputTensorDesc.desc;
    joinOutputDesc.Axis = inputOutputShape.size() - 1;
    joinOutputDmlDesc.Type = DML_OPERATOR_JOIN;
    joinOutputDmlDesc.Desc = &joinOutputDesc;
  }

  std::vector<const DML_OPERATOR_DESC*> opDescs = {
    &copyInputDmlDesc,  // Copy the input data to preserve the real input shape
    &splitPartialInputDmlDesc,  // Split the input data
    &gatherCosSinDmlDesc,       // Gather cos
    &gatherCosSinDmlDesc,       // Gather sin
    &signRangeDmlDesc,          // Generate the signs

    &joinPartialInputDmlDesc,  // Join the split data
    &mulCosSinDmlDesc,         // Multiply cos with the non-rotated data
    &mulCosSinDmlDesc,         // Multiply sin with the rotated data
    &mulSignDmlDesc,           // Multiply the sign with the rotated data
    &addDmlDesc,  // Add the rotated cos and non-rotated sin parts together
  };

  // Handle partial rotary embedding
  if (m_isPartialRotaryEmbedding) {
    opDescs.push_back(&splitInputDmlDesc);
    opDescs.push_back(&joinOutputDmlDesc);
  }

  // offset is used to define the location of the node in the GQO graph
  size_t offset = dmlOperators.size();
  dmlOperators.resize(offset + opDescs.size());

  HRESULT status = S_OK;
  for (size_t i = 0; i < opDescs.size(); ++i) {
    // Create the operator.
    status = pContext.DmlDevice()->CreateOperator(
      opDescs[i], IID_PPV_ARGS(&dmlOperators[i + offset])
    );

    if (FAILED(status)) {
      throw std::runtime_error("Failed to compile RoPE DirectML operator.");
    }
  }
}

// =====================================================================================================================
// Constructs a DirectML-based RotaryEmbeddingOperator which evaluates a rotary
// embedding operation with the given parameters.
RotaryEmbeddingOperator::RotaryEmbeddingOperator(
  const Context& pContext, const RotaryEmbeddingParams& params,
  bool disableCompile
)
  : m_ropeParams(params), m_dataType(StringToDataType(params.dataType)) {
  SharedInit();

  if (disableCompile) return;

  std::vector<CComPtr<IDMLOperator>> dmlOperators(0);
  std::shared_ptr<DmlTensorDesc> nullDmlTensor = nullptr;
  CreateRotaryEmbeddingOperator(pContext, dmlOperators, nullDmlTensor);

  std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );
  std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );

  for (size_t i = 0; i < dmlOperators.size(); ++i) {
    // Create the operator.
    dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{dmlOperators[i]};
    dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{
      DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]
    };
  }

  // Construct the graph
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

  enum NodeIndex : uint32_t {
    copyInputOpIndex,
    splitPartialInputOpIndex,
    gatherCosOpIndex,
    gatherSinOpIndex,
    signRangeOpIndex,

    joinPartialInputOpIndex,
    mulCosOpIndex,
    mulSinOpIndex,
    mulSignOpIndex,
    addOpIndex,
    splitInputOpIndex,
    joinOutputOpIndex
  };

  enum InputIndex : uint32_t {
    inputDataIndex,
    positionIdsIndex,
    cosCacheIndex,
    sinCacheIndex,
  };

  // Handle partial rotary embedding
  if (m_isPartialRotaryEmbedding) {
    DML_INPUT_GRAPH_EDGE_DESC inputToSplitInputEdge = {};
    inputToSplitInputEdge.GraphInputIndex = inputDataIndex;
    inputToSplitInputEdge.ToNodeIndex = splitInputOpIndex;
    inputToSplitInputEdge.ToNodeInputIndex = 0;
    inputEdges.push_back(inputToSplitInputEdge);

    DML_INTERMEDIATE_GRAPH_EDGE_DESC partialInputToCopyInputEdge = {};
    partialInputToCopyInputEdge.FromNodeIndex = splitInputOpIndex;
    partialInputToCopyInputEdge.FromNodeOutputIndex = 0;
    partialInputToCopyInputEdge.ToNodeIndex = copyInputOpIndex;
    partialInputToCopyInputEdge.ToNodeInputIndex = 0;
    intermediateEdges.push_back(partialInputToCopyInputEdge);
  } else {
    DML_INPUT_GRAPH_EDGE_DESC inputToCopyInputEdge = {};
    inputToCopyInputEdge.GraphInputIndex = inputDataIndex;
    inputToCopyInputEdge.ToNodeIndex = copyInputOpIndex;
    inputToCopyInputEdge.ToNodeInputIndex = 0;
    inputEdges.push_back(inputToCopyInputEdge);
  }

  DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherCosEdge = {};
  positionIdsToGatherCosEdge.GraphInputIndex = positionIdsIndex;
  positionIdsToGatherCosEdge.ToNodeIndex = gatherCosOpIndex;
  positionIdsToGatherCosEdge.ToNodeInputIndex = 1;
  inputEdges.push_back(positionIdsToGatherCosEdge);

  DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherSinEdge = {};
  positionIdsToGatherSinEdge.GraphInputIndex = positionIdsIndex;
  positionIdsToGatherSinEdge.ToNodeIndex = gatherSinOpIndex;
  positionIdsToGatherSinEdge.ToNodeInputIndex = 1;
  inputEdges.push_back(positionIdsToGatherSinEdge);

  DML_INPUT_GRAPH_EDGE_DESC cosToGatherEdge = {};
  cosToGatherEdge.GraphInputIndex = cosCacheIndex;
  cosToGatherEdge.ToNodeIndex = gatherCosOpIndex;
  cosToGatherEdge.ToNodeInputIndex = 0;
  inputEdges.push_back(cosToGatherEdge);

  DML_INPUT_GRAPH_EDGE_DESC sinToGatherEdge = {};
  sinToGatherEdge.GraphInputIndex = sinCacheIndex;
  sinToGatherEdge.ToNodeIndex = gatherSinOpIndex;
  sinToGatherEdge.ToNodeInputIndex = 0;
  inputEdges.push_back(sinToGatherEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC inputToSplitEdge = {};
  inputToSplitEdge.FromNodeIndex = copyInputOpIndex;
  inputToSplitEdge.FromNodeOutputIndex = 0;
  inputToSplitEdge.ToNodeIndex = splitPartialInputOpIndex;
  inputToSplitEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(inputToSplitEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC nonRotatedDataToMulEdge = {};
  nonRotatedDataToMulEdge.FromNodeIndex = copyInputOpIndex;
  nonRotatedDataToMulEdge.FromNodeOutputIndex = 0;
  nonRotatedDataToMulEdge.ToNodeIndex = mulCosOpIndex;
  nonRotatedDataToMulEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(nonRotatedDataToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfDataToJoinEdge = {};
  secondHalfDataToJoinEdge.FromNodeIndex = splitPartialInputOpIndex;
  secondHalfDataToJoinEdge.FromNodeOutputIndex = 1;
  secondHalfDataToJoinEdge.ToNodeIndex = joinPartialInputOpIndex;
  secondHalfDataToJoinEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(secondHalfDataToJoinEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfDataToJoinEdge = {};
  firstHalfDataToJoinEdge.FromNodeIndex = splitPartialInputOpIndex;
  firstHalfDataToJoinEdge.FromNodeOutputIndex = 0;
  firstHalfDataToJoinEdge.ToNodeIndex = joinPartialInputOpIndex;
  firstHalfDataToJoinEdge.ToNodeInputIndex = 1;
  intermediateEdges.push_back(firstHalfDataToJoinEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC cosToMulEdge = {};
  cosToMulEdge.FromNodeIndex = gatherCosOpIndex;
  cosToMulEdge.FromNodeOutputIndex = 0;
  cosToMulEdge.ToNodeIndex = mulCosOpIndex;
  cosToMulEdge.ToNodeInputIndex = 1;
  intermediateEdges.push_back(cosToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedDataToMulEdge = {};
  rotatedDataToMulEdge.FromNodeIndex = joinPartialInputOpIndex;
  rotatedDataToMulEdge.FromNodeOutputIndex = 0;
  rotatedDataToMulEdge.ToNodeIndex = mulSinOpIndex;
  rotatedDataToMulEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(rotatedDataToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC sinToMulEdge = {};
  sinToMulEdge.FromNodeIndex = gatherSinOpIndex;
  sinToMulEdge.FromNodeOutputIndex = 0;
  sinToMulEdge.ToNodeIndex = mulSinOpIndex;
  sinToMulEdge.ToNodeInputIndex = 1;
  intermediateEdges.push_back(sinToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedSinToMulEdge = {};
  rotatedSinToMulEdge.FromNodeIndex = mulSinOpIndex;
  rotatedSinToMulEdge.FromNodeOutputIndex = 0;
  rotatedSinToMulEdge.ToNodeIndex = mulSignOpIndex;
  rotatedSinToMulEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(rotatedSinToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC signToMulEdge = {};
  signToMulEdge.FromNodeIndex = signRangeOpIndex;
  signToMulEdge.FromNodeOutputIndex = 0;
  signToMulEdge.ToNodeIndex = mulSignOpIndex;
  signToMulEdge.ToNodeInputIndex = 1;
  intermediateEdges.push_back(signToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC nonRotatedCosToAddEdge = {};
  nonRotatedCosToAddEdge.FromNodeIndex = mulCosOpIndex;
  nonRotatedCosToAddEdge.FromNodeOutputIndex = 0;
  nonRotatedCosToAddEdge.ToNodeIndex = addOpIndex;
  nonRotatedCosToAddEdge.ToNodeInputIndex = 0;
  intermediateEdges.push_back(nonRotatedCosToAddEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedSinToAddEdge = {};
  rotatedSinToAddEdge.FromNodeIndex = mulSignOpIndex;
  rotatedSinToAddEdge.FromNodeOutputIndex = 0;
  rotatedSinToAddEdge.ToNodeIndex = addOpIndex;
  rotatedSinToAddEdge.ToNodeInputIndex = 1;
  intermediateEdges.push_back(rotatedSinToAddEdge);

  // Handle partial rotary embedding
  if (m_isPartialRotaryEmbedding) {
    DML_INTERMEDIATE_GRAPH_EDGE_DESC addToJoinOutputEdge = {};
    addToJoinOutputEdge.FromNodeIndex = addOpIndex;
    addToJoinOutputEdge.FromNodeOutputIndex = 0;
    addToJoinOutputEdge.ToNodeIndex = joinOutputOpIndex;
    addToJoinOutputEdge.ToNodeInputIndex = 0;
    intermediateEdges.push_back(addToJoinOutputEdge);

    DML_INTERMEDIATE_GRAPH_EDGE_DESC remainingInputToJoinOutputEdge = {};
    remainingInputToJoinOutputEdge.FromNodeIndex = splitInputOpIndex;
    remainingInputToJoinOutputEdge.FromNodeOutputIndex = 1;
    remainingInputToJoinOutputEdge.ToNodeIndex = joinOutputOpIndex;
    remainingInputToJoinOutputEdge.ToNodeInputIndex = 1;
    intermediateEdges.push_back(remainingInputToJoinOutputEdge);

    DML_OUTPUT_GRAPH_EDGE_DESC joinOutputToOutputEdge = {};
    joinOutputToOutputEdge.FromNodeIndex = joinOutputOpIndex;
    joinOutputToOutputEdge.FromNodeOutputIndex = 0;
    joinOutputToOutputEdge.GraphOutputIndex = 0;
    outputEdges.push_back(joinOutputToOutputEdge);
  } else {
    DML_OUTPUT_GRAPH_EDGE_DESC addToOutputEdge = {};
    addToOutputEdge.FromNodeIndex = addOpIndex;
    addToOutputEdge.FromNodeOutputIndex = 0;
    addToOutputEdge.GraphOutputIndex = 0;
    outputEdges.push_back(addToOutputEdge);
  }

  DML_GRAPH_DESC graphDesc = {};

  std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(
    static_cast<uint32_t>(inputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(
    static_cast<uint32_t>(outputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(
    static_cast<uint32_t>(intermediateEdges.size())
  );

  // Build the graph description from the above edges
  graphDesc.InputCount = static_cast<uint32_t>(m_inputTensorDescVec.size());
  graphDesc.OutputCount = static_cast<uint32_t>(m_outputTensorDescVec.size());
  graphDesc.NodeCount = static_cast<uint32_t>(dmlOperators.size());
  HRESULT status = S_OK;

  graphDesc.Nodes = dmlGraphNodes.data();

  // Set the input edges
  graphDesc.InputEdgeCount = static_cast<uint32_t>(inputEdges.size());
  for (size_t i = 0; i < graphDesc.InputEdgeCount; ++i) {
    dmlInputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &inputEdges[i]};
  }
  graphDesc.InputEdges = dmlInputEdges.data();

  // Set the output edges
  graphDesc.OutputEdgeCount = graphDesc.OutputCount;
  for (size_t i = 0; i < graphDesc.OutputEdgeCount; ++i) {
    dmlOutputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &outputEdges[i]};
  }
  graphDesc.OutputEdges = dmlOutputEdges.data();

  // Set the intermediate edges
  graphDesc.IntermediateEdgeCount =
    static_cast<uint32_t>(intermediateEdges.size());
  for (size_t i = 0; i < graphDesc.IntermediateEdgeCount; ++i) {
    dmlIntermediateEdges[i] = DML_GRAPH_EDGE_DESC{
      DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediateEdges[i]
    };
  }
  graphDesc.IntermediateEdges = dmlIntermediateEdges.data();

  CComPtr<IDMLDevice1> dmlDevice1;
  status = pContext.DmlDevice()->QueryInterface(IID_PPV_ARGS(&dmlDevice1));
  if (FAILED(status)) {
    throw std::runtime_error("Failed to acquire dml graph interface.");
  }

  DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_NONE;
  executionFlags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;

  status = dmlDevice1->CompileGraph(
    &graphDesc, executionFlags, IID_PPV_ARGS(&m_operator)
  );
  if (FAILED(status)) {
    throw std::runtime_error("Failed to compile graph to a DirectML operator.");
  }
}

// =====================================================================================================================
// Code shared between our constructors.
void RotaryEmbeddingOperator::SharedInit() {
  if (!(m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16 ||
        m_dataType == DML_TENSOR_DATA_TYPE_FLOAT32)) {
    std::wstringstream ss;
    ss << "RotaryEmbedding only supports Float16 and Float32, not "
       << m_ropeParams.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // Build the tensor descriptors. For DirectML's benefit, we must have one
  // pointer for every possible tensor and must use empty pointers as
  // placeholders for unused optional tensors.

  // const uint32_t hiddenSize = inputIs4D ? inputDataSizes[1] *
  // inputDataSizes[3] : inputDataSizes.back();

  const uint32_t hiddenSize = m_ropeParams.inputShape.back();

  // Handle independent rope implementation which might or might not
  // have the numHeads attribute. Calculate headsize, and/or rot_emb_dim
  // and / or numheads based on the attributes available.
  // numHeads must be provided if rot_emb_dim is specified
  // These attributes are used to provide support for partial
  // rotary embedding
  if (m_ropeParams.numHeads == 0) {
    m_ropeParams.headSize = m_ropeParams.cosShape.back() * 2;
    m_ropeParams.rotaryEmbeddingDim = m_ropeParams.headSize;
    m_ropeParams.numHeads = hiddenSize / m_ropeParams.headSize;
  } else {
    m_ropeParams.headSize = hiddenSize / m_ropeParams.numHeads;
    if (m_ropeParams.rotaryEmbeddingDim == 0) {
      // temporary as of now. We need to check this value from the model
      // attributes. The current fused model doesn't has this field as of now.
      m_ropeParams.rotaryEmbeddingDim = m_ropeParams.cosShape.back() * 2;
    }
  }
  //     m_ropeParams.headSize = m_ropeParams.numHeads == 0
  //         ? m_ropeParams.cosShape.back() * 2
  //         : hiddenSize / m_ropeParams.numHeads;
  //// }
  // if (m_ropeParams.rotaryEmbeddingDim == 0) {
  //     if (m_ropeParams.numHeads == 0)
  //     {
  //         m_ropeParams.rotaryEmbeddingDim = m_ropeParams.headSize;
  //     }
  //     else
  //     {
  //         m_ropeParams.rotaryEmbeddingDim = m_ropeParams.cosShape.back() * 2;
  //     }
  // }

  // if (m_ropeParams.numHeads == 0)
  // {
  //     m_ropeParams.numHeads = hiddenSize / m_ropeParams.headSize;
  // }

  const std::vector<int64> sizeInput{
    m_ropeParams.batchSize,
    m_ropeParams.sequenceLength,
    m_ropeParams.inputShape.back(),
  };
  const std::vector<int64> sizePositionIds{
    m_ropeParams.batchSize, m_ropeParams.sequenceLength
  };
  const std::vector<int64> sizeCache = m_ropeParams.cosShape;

  const std::vector<int64> sizeOutput{
    m_ropeParams.batchSize, m_ropeParams.sequenceLength,
    m_ropeParams.inputShape.back()
  };

  const std::vector<int64> autoStrides_1d{-1};
  const std::vector<int64> autoStrides_2d{-1, -1};
  const std::vector<int64> autoStrides_3d{-1, -1, -1};
  const std::vector<int64> autoStrides{-1, -1, -1, -1};
  const std::vector<int64> autoStrides_5d{-1, -1, -1, -1, -1};

  // 0
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"InputTensor", m_ropeParams.order, m_dataType, sizeInput, autoStrides_3d
  ));
  // 1
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"PositionIdsTensor", m_ropeParams.order, DML_TENSOR_DATA_TYPE_INT32,
    sizePositionIds, autoStrides_2d
  ));
  // 2
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"CosCacheTensor", m_ropeParams.order, m_dataType, sizeCache, autoStrides_2d
  ));
  // 3
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"SinCacheTensor", m_ropeParams.order, m_dataType, sizeCache, autoStrides_2d
  ));
  m_outputTensorDescVec.emplace_back(CreateTensorDesc(
    L"OutputTensor", m_ropeParams.order, m_dataType, sizeOutput, autoStrides_3d
  ));

  // Additional tensor descriptions for intermediate tensors
  const std::vector<int64> splitInputDataTensorShape =
    m_ropeParams.interleaved
      ? std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength,
           m_ropeParams.numHeads, m_ropeParams.rotaryEmbeddingDim / 2, 1}
        )
      : std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength,
           m_ropeParams.numHeads, 1, m_ropeParams.rotaryEmbeddingDim / 2}
        );

  const std::vector<int64> partialInputDataTensorShape =
    m_ropeParams.interleaved
      ? std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength,
           m_ropeParams.numHeads, m_ropeParams.rotaryEmbeddingDim / 2, 2}
        )
      : std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength,
           m_ropeParams.numHeads, 2, m_ropeParams.rotaryEmbeddingDim / 2}
        );

  const std::vector<int64> gatheredCosSinShape = {
    1, m_ropeParams.batchSize, m_ropeParams.sequenceLength,
    m_ropeParams.rotaryEmbeddingDim / 2
  };

  const std::vector<int64> reshapedCosSinShape =
    m_ropeParams.interleaved
      ? std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength, 1,
           m_ropeParams.rotaryEmbeddingDim / 2, 1}
        )
      : std::vector<int64>(
          {m_ropeParams.batchSize, m_ropeParams.sequenceLength, 1, 1,
           m_ropeParams.rotaryEmbeddingDim / 2}
        );

  const std::vector<int64> signTensorShape = {2};
  const std::vector<int64> reshapedSignShape =
    m_ropeParams.interleaved ? std::vector<int64>({1, 1, 1, 1, 2})
                             : std::vector<int64>({1, 1, 1, 2, 1});
  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"SplitInputDataTensor", m_ropeParams.order, m_dataType,
    splitInputDataTensorShape, autoStrides_5d
  ));

  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"JoinedDataTensor", m_ropeParams.order, m_dataType,
    partialInputDataTensorShape, autoStrides_5d
  ));

  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"GatheredCosSinTensor", m_ropeParams.order, m_dataType,
    gatheredCosSinShape, autoStrides
  ));

  std::vector<int64> reshapeStride = ComputeStrides(reshapedCosSinShape);
  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"BroadcastedCosSinTensor", m_ropeParams.order, m_dataType,
    partialInputDataTensorShape, reshapeStride
  ));

  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"SignTensor", L"N", m_dataType, signTensorShape, autoStrides_1d
  ));

  std::vector<int64> reshapeSignStride = ComputeStrides(reshapedSignShape);
  m_interTensorDescVec.emplace_back(CreateTensorDesc(
    L"BroadcastedSignCosSinTensor", m_ropeParams.order, m_dataType,
    partialInputDataTensorShape, reshapeSignStride
  ));
}

#ifdef CPU_IMPL
// =====================================================================================================================
void RotaryEmbeddingOperator::CpuImpl(
  const float* input, const int64_t* position_ids, const float* cos_cache,
  const float* sin_cache, float* output
) {
  const int batch_size = m_ropeParams.batchSize;
  const int sequence_length = m_ropeParams.sequenceLength;
  const int n_heads = m_ropeParams.numHeads;
  const int head_size = m_ropeParams.headSize;
  const int head_stride = head_size;
  const int seq_stride = n_heads * head_size;
  const int batch_stride = sequence_length * seq_stride;
  const int position_ids_format = 1;
  const int rotary_emb_dim = m_ropeParams.cosShape.back() * 2;
  const int half_rotary_emb_dim = rotary_emb_dim / 2;

  const bool transposed = false;

  const int loop_len = batch_size * sequence_length * n_heads;
  const double cost = static_cast<double>(rotary_emb_dim);

  for (std::ptrdiff_t ptr = 0; ptr < loop_len; ++ptr) {
    const int b = static_cast<int>((ptr / n_heads) / sequence_length);
    const int s = static_cast<int>((ptr / n_heads) % sequence_length);
    const int n = static_cast<int>(ptr % n_heads);

    const int block_offset =
      b * batch_stride + s * seq_stride + n * head_stride;

    const float* input_data = input + block_offset;
    float* output_data = output + block_offset;

    // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
    const int position_id =
      (position_ids_format == 0)
        ? static_cast<int>(position_ids[0]) + s
        : static_cast<int>(position_ids[b * sequence_length + s]);
    const int cache_offset = position_id * half_rotary_emb_dim;
    const float* cos_data = cos_cache + cache_offset;
    const float* sin_data = sin_cache + cache_offset;

    int cache_idx = 0;
    bool sign = false;
    int j = 0;
    for (int i = 0; i < rotary_emb_dim; i++) {
      if (m_ropeParams.interleaved) {
        cache_idx = (i / 2) % half_rotary_emb_dim;
        sign = i & 1;
        j = sign ? i - 1 : i + 1;  // i - sign
      } else {
        cache_idx = i % half_rotary_emb_dim;
        sign = (i >= half_rotary_emb_dim);
        j = (i + half_rotary_emb_dim) % rotary_emb_dim;
      }
      float output_data_i = static_cast<float>(input_data[i]) *
                            static_cast<float>(cos_data[cache_idx]);
      float input_data_j = static_cast<float>(input_data[j]);
      float sin_data_cache_idx = static_cast<float>(sin_data[cache_idx]);
      if (sign) {
        output_data_i += input_data_j * sin_data_cache_idx;
      } else {
        output_data_i -= input_data_j * sin_data_cache_idx;
      }
      output_data[i] = static_cast<float>(output_data_i);
    }
    for (int i = rotary_emb_dim; i < head_size; i++) {
      output_data[i] = input_data[i];
    }
  }
}
#endif

}  // namespace ryzenai::onnx_utils
