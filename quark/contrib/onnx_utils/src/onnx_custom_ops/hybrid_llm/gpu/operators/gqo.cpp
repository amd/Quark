// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/gqo.h"

#include <fstream>
#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
void GQOOperator::CreateInputEdge(
  std::string name, uint32 graphInputIndex, uint32 toNodeIndex,
  uint32 toNodeInputIndex
) {
  std::string edgeName = name;
  DML_INPUT_GRAPH_EDGE_DESC edge = {};
  edge.GraphInputIndex = graphInputIndex;
  edge.ToNodeIndex = toNodeIndex;
  edge.ToNodeInputIndex = toNodeInputIndex;
  m_inputEdges.push_back(edge);
}

// =====================================================================================================================
void GQOOperator::CreateIntermediateEdge(
  std::string name, uint32 fromNodeIndex, uint32 fromNodeOutputIndex,
  uint32 toNodeIndex, uint32 toNodeInputIndex
) {
  DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
  edge.FromNodeIndex = fromNodeIndex;
  edge.FromNodeOutputIndex = fromNodeOutputIndex;
  edge.ToNodeIndex = toNodeIndex;
  edge.ToNodeInputIndex = toNodeInputIndex;
  m_intermediateEdges.push_back(edge);
}

// =====================================================================================================================
void GQOOperator::DynamicInitialization(
  const Context& pContext, void* params, bool rebuildOp
) {
  m_firstRun = false;
  GQOParams* gqoParams = (GQOParams*)params;

  m_gqoParams.gqa.inputPastKeyShape = gqoParams->gqa.inputPastKeyShape;
  m_gqoParams.gqa.inputPastValueShape = gqoParams->gqa.inputPastValueShape;
  m_gqoParams.gqa.inputSubCastShape = gqoParams->gqa.inputSubCastShape;
  m_gqoParams.gqa.inputGatherCastShape = gqoParams->gqa.inputGatherCastShape;
  m_gqoParams.gqa.outputKeyShape = gqoParams->gqa.outputKeyShape;
  m_gqoParams.gqa.outputValueShape = gqoParams->gqa.outputValueShape;
  m_gqoParams.gqa.outputQueryShape = gqoParams->gqa.outputQueryShape;

  constexpr bool useGraph = true;
  // create Rotary Embedding GQA MatmulNBit constructors

  if (!rebuildOp || !m_isGraphConstructed) {
    m_rotEmbQueryOp = std::make_shared<RotaryEmbeddingOperator>(
      pContext, m_gqoParams.rotEmbQuery, useGraph
    );
    m_rotEmbKeyOp = std::make_shared<RotaryEmbeddingOperator>(
      pContext, m_gqoParams.rotEmbKey, useGraph
    );
    m_internalMatmul = std::make_shared<MatMulNBitsOperator>(
      pContext, m_gqoParams.matMulNBits, useGraph
    );
  }
  m_groupQueryOp =
    std::make_shared<GQAOperator>(pContext, m_gqoParams.gqa, useGraph);

  if (useGraph) {
    std::vector<CComPtr<IDMLOperator>> dmlOperators;

    uint32 graphInputCount = 0;

    // these are outputs from two Ropes which need to go as input to GQA
    std::shared_ptr<DmlTensorDesc> dmlInputToGQAFromRopeQ =
      std::make_shared<DmlTensorDesc>();
    std::shared_ptr<DmlTensorDesc> dmlInputToGQAFromRopeK =
      std::make_shared<DmlTensorDesc>();

    m_rotEmbQueryOp.get()->CreateRotaryEmbeddingOperator(
      pContext, dmlOperators, dmlInputToGQAFromRopeQ
    );
    m_rotEmbKeyOp.get()->CreateRotaryEmbeddingOperator(
      pContext, dmlOperators, dmlInputToGQAFromRopeK
    );

    bool is_partial_rope = false;
    is_partial_rope = m_rotEmbQueryOp.get()->IsPartialRotaryEmbedding();

    // these are inputs/ outputs to/from GQA. Output will go to Matmul
    std::shared_ptr<DmlTensorDesc> dmlInputToMatMulFromGQA =
      std::make_shared<DmlTensorDesc>();
    m_groupQueryOp.get()->CreateGQAOperator(
      pContext, dmlOperators, dmlInputToGQAFromRopeQ, dmlInputToGQAFromRopeK,
      dmlInputToMatMulFromGQA
    );
    m_internalMatmul.get()->CreateMatmulNBitOperator(
      pContext, dmlOperators, dmlInputToMatMulFromGQA
    );

    m_inputTensorDescVec.clear();
    m_outputTensorDescVec.clear();

    m_inputEdges.clear();
    m_intermediateEdges.clear();
    m_outputEdges.clear();

    for (const std::shared_ptr<TensorDesc>& pDesc :
         m_rotEmbQueryOp.get()->GetInputTensorDescVector()) {
      m_inputTensorDescVec.emplace_back(pDesc);
    }
    m_inputTensorDescVec.emplace_back(
      m_rotEmbKeyOp.get()->GetInputTensorDescVector()[0]
    );
    m_inputTensorDescVec.emplace_back(
      m_groupQueryOp.get()->GetInputTensorDescVector()[2]
    );
    if (m_gqoParams.gqa.sequenceLength == 1)
      m_inputTensorDescVec.emplace_back(
        m_groupQueryOp.get()->GetInputTensorDescVector()[3]
      );

    for (int i = 1;
         i < m_internalMatmul.get()->GetInputTensorDescVector().size(); i++) {
      m_inputTensorDescVec.emplace_back(
        m_internalMatmul.get()->GetInputTensorDescVector()[i]
      );
    }

    // output
    m_outputTensorDescVec.emplace_back(
      m_groupQueryOp.get()->GetOutputTensorDescVector()[1]
    );
    m_outputTensorDescVec.emplace_back(
      m_groupQueryOp.get()->GetOutputTensorDescVector()[2]
    );

    m_outputTensorDescVec.emplace_back(
      m_internalMatmul.get()->GetOutputTensorDescVector()[0]
    );

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

    uint32 nodeCount = 0;

    // Rope Query
    uint16_t qCopyInputOpIndex = nodeCount++;          // 0
    uint16_t qSplitPartialInputOpIndex = nodeCount++;  // 1
    uint16_t qGatherCosOpIndex = nodeCount++;          // 2
    uint16_t qGatherSinOpIndex = nodeCount++;          // 3
    uint16_t qSignRangeOpIndex = nodeCount++;          // 4
    uint16_t qJoinPartialInputOpIndex = nodeCount++;   // 5
    uint16_t qMulCosOpIndex = nodeCount++;             // 6
    uint16_t qMulSinOpIndex = nodeCount++;             // 7
    uint16_t qMulSignOpIndex = nodeCount++;            // 8
    uint16_t qAddOpIndex = nodeCount++;                // 9

    // conditional
    uint16_t qSplitInputOpIndex = 0xffff;
    uint16_t qJoinOutputOpIndex = 0xffff;
    if (is_partial_rope) {
      qSplitInputOpIndex = nodeCount++;  // 10
      qJoinOutputOpIndex = nodeCount++;  // 11
    }

    enum RopeQueryInputIndex : uint32_t {
      rqInputDataIndex,
      positionIdsIndex,
      cosCacheIndex,
      sinCacheIndex,
      rqInputCount,
    };

    if (is_partial_rope) {
      CreateInputEdge(
        "inputToSplitInputEdge", RopeQueryInputIndex::rqInputDataIndex,
        qSplitInputOpIndex, 0
      );

      CreateIntermediateEdge(
        "partialInputToCopyInputEdge", qSplitInputOpIndex, 0, qCopyInputOpIndex,
        0
      );
    } else {
      // Rotary Embedding Query graph construction
      CreateInputEdge(
        "inputToCopyInputEdge", RopeQueryInputIndex::rqInputDataIndex,
        qCopyInputOpIndex, 0
      );
    }
    CreateInputEdge(
      "positionIdsToGatherCosEdge", RopeQueryInputIndex::positionIdsIndex,
      qGatherCosOpIndex, 1
    );
    CreateInputEdge(
      "positionIdsToGatherSinEdge", RopeQueryInputIndex::positionIdsIndex,
      qGatherSinOpIndex, 1
    );
    CreateInputEdge(
      "cosToGatherEdge", RopeQueryInputIndex::cosCacheIndex, qGatherCosOpIndex,
      0
    );
    CreateInputEdge(
      "sinToGatherEdge", RopeQueryInputIndex::sinCacheIndex, qGatherSinOpIndex,
      0
    );

    CreateIntermediateEdge(
      "inputToSplitEdge", qCopyInputOpIndex, 0, qSplitPartialInputOpIndex, 0
    );
    CreateIntermediateEdge(
      "nonRotatedDataToMulEdge", qCopyInputOpIndex, 0, qMulCosOpIndex, 0
    );
    CreateIntermediateEdge(
      "secondHalfDataToJoinEdge", qSplitPartialInputOpIndex, 1,
      qJoinPartialInputOpIndex, 0
    );
    CreateIntermediateEdge(
      "firstHalfDataToJoinEdge", qSplitPartialInputOpIndex, 0,
      qJoinPartialInputOpIndex, 1
    );
    CreateIntermediateEdge(
      "cosToMulEdge", qGatherCosOpIndex, 0, qMulCosOpIndex, 1
    );
    CreateIntermediateEdge(
      "rotatedDataToMulEdge", qJoinPartialInputOpIndex, 0, qMulSinOpIndex, 0
    );
    CreateIntermediateEdge(
      "sinToMulEdge", qGatherSinOpIndex, 0, qMulSinOpIndex, 1
    );
    CreateIntermediateEdge(
      "rotatedSinToMulEdge", qMulSinOpIndex, 0, qMulSignOpIndex, 0
    );
    CreateIntermediateEdge(
      "signToMulEdge", qSignRangeOpIndex, 0, qMulSignOpIndex, 1
    );
    CreateIntermediateEdge(
      "nonRotatedCosToAddEdge", qMulCosOpIndex, 0, qAddOpIndex, 0
    );
    CreateIntermediateEdge(
      "rotatedSinToAddEdge", qMulSignOpIndex, 0, qAddOpIndex, 1
    );

    if (is_partial_rope) {
      CreateIntermediateEdge(
        "addToJoinOutputEdge", qAddOpIndex, 0, qJoinOutputOpIndex, 0
      );

      CreateIntermediateEdge(
        "remainingInputToJoinOutputEdge", qSplitInputOpIndex, 1,
        qJoinOutputOpIndex, 1
      );
    }

    uint16_t kCopyInputOpIndex = nodeCount++;
    uint16_t kSplitPartialInputOpIndex = nodeCount++;
    uint16_t kGatherCosOpIndex = nodeCount++;
    uint16_t kGatherSinOpIndex = nodeCount++;
    uint16_t kSignRangeOpIndex = nodeCount++;

    uint16_t kJoinPartialInputOpIndex = nodeCount++;
    uint16_t kMulCosOpIndex = nodeCount++;
    uint16_t kMulSinOpIndex = nodeCount++;
    uint16_t kMulSignOpIndex = nodeCount++;
    uint16_t kAddOpIndex = nodeCount++;

    // conditional
    uint16_t kSplitInputOpIndex = 0xffff;
    uint16_t kJoinOutputOpIndex = 0xffff;
    if (is_partial_rope) {
      kSplitInputOpIndex = nodeCount++;
      kJoinOutputOpIndex = nodeCount++;
    }

    enum RopeKeyInputIndex : uint32_t {
      rkInputDataIndex = RopeQueryInputIndex::rqInputCount,
      rkInputCount,
    };

    if (is_partial_rope) {
      CreateInputEdge(
        "inputToSplitInputEdge", RopeKeyInputIndex::rkInputDataIndex,
        /*RopeKeyNodeIndex::splitInputOpIndex*/ kSplitInputOpIndex, 0
      );

      CreateIntermediateEdge(
        "partialInputToCopyInputEdge",
        /*RopeKeyNodeIndex::splitInputOpIndex*/ kSplitInputOpIndex, 0,
        kCopyInputOpIndex, 0
      );
    } else {
      // Rotary Embedding Key graph construction
      CreateInputEdge(
        "inputToCopyInputEdge", RopeKeyInputIndex::rkInputDataIndex,
        kCopyInputOpIndex, 0
      );
    }
    CreateInputEdge(
      "positionIdsToGatherCosEdge", RopeQueryInputIndex::positionIdsIndex,
      kGatherCosOpIndex, 1
    );
    CreateInputEdge(
      "positionIdsToGatherSinEdge", RopeQueryInputIndex::positionIdsIndex,
      kGatherSinOpIndex, 1
    );
    CreateInputEdge(
      "cosToGatherEdge", RopeQueryInputIndex::cosCacheIndex, kGatherCosOpIndex,
      0
    );
    CreateInputEdge(
      "sinToGatherEdge", RopeQueryInputIndex::sinCacheIndex, kGatherSinOpIndex,
      0
    );

    CreateIntermediateEdge(
      "inputToSplitEdge", kCopyInputOpIndex, 0, kSplitPartialInputOpIndex, 0
    );
    CreateIntermediateEdge(
      "nonRotatedDataToMulEdge", kCopyInputOpIndex, 0, kMulCosOpIndex, 0
    );
    CreateIntermediateEdge(
      "secondHalfDataToJoinEdge", kSplitPartialInputOpIndex, 1,
      kJoinPartialInputOpIndex, 0
    );
    CreateIntermediateEdge(
      "firstHalfDataToJoinEdge", kSplitPartialInputOpIndex, 0,
      kJoinPartialInputOpIndex, 1
    );
    CreateIntermediateEdge(
      "cosToMulEdge", kGatherCosOpIndex, 0, kMulCosOpIndex, 1
    );
    CreateIntermediateEdge(
      "rotatedDataToMulEdge", kJoinPartialInputOpIndex, 0, kMulSinOpIndex, 0
    );
    CreateIntermediateEdge(
      "sinToMulEdge", kGatherSinOpIndex, 0, kMulSinOpIndex, 1
    );
    CreateIntermediateEdge(
      "rotatedSinToMulEdge", kMulSinOpIndex, 0, kMulSignOpIndex, 0
    );
    CreateIntermediateEdge(
      "signToMulEdge", kSignRangeOpIndex, 0, kMulSignOpIndex, 1
    );
    CreateIntermediateEdge(
      "nonRotatedCosToAddEdge", kMulCosOpIndex, 0, kAddOpIndex, 0
    );
    CreateIntermediateEdge(
      "rotatedSinToAddEdge", kMulSignOpIndex, 0, kAddOpIndex, 1
    );

    if (is_partial_rope) {
      CreateIntermediateEdge(
        "addToJoinOutputEdge", kAddOpIndex, 0, kJoinOutputOpIndex, 0
      );

      CreateIntermediateEdge(
        "remainingInputToJoinOutputEdge", kSplitInputOpIndex, 1,
        kJoinOutputOpIndex, 1
      );
    }

    const bool isFp16 = (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16);

    uint16_t gqaMhaOpIndex = nodeCount++;
    uint16_t gqaInQueryCastOpIndex = 0xffff;
    uint16_t gqaInKeyCastOpIndex = 0xffff;
    uint16_t gqaInValueCastOpIndex = 0xffff;
    uint16_t gqaOutQueryCastOpIndex = 0xffff;
    uint16_t gqaOutKeyCastOpIndex = 0xffff;
    uint16_t gqaOutValueCastOpIndex = 0xffff;

    if (isFp16) {
      gqaInQueryCastOpIndex = nodeCount++;
      gqaInKeyCastOpIndex = nodeCount++;
      gqaInValueCastOpIndex = nodeCount++;
      gqaOutQueryCastOpIndex = nodeCount++;
      gqaOutKeyCastOpIndex = nodeCount++;
      gqaOutValueCastOpIndex = nodeCount++;
    }

    enum GqaInputIndex : uint32_t {
      gqaInputFromMatmulIndex = RopeKeyInputIndex::rkInputCount,
      gqaInputCount,
    };

    if (isFp16) {
      // Link the query/key/value inputs to the cast nodes
      CreateInputEdge(
        "inputFromMatmulToGqaVCastEdge", GqaInputIndex::gqaInputFromMatmulIndex,
        gqaInValueCastOpIndex, 0
      );  // Value

      if (is_partial_rope) {
        CreateIntermediateEdge(
          "inputFromRopeQAddToGqaQCastEdge", qJoinOutputOpIndex, 0,
          gqaInQueryCastOpIndex,
          0
        );  // Query
        CreateIntermediateEdge(
          "inputFromRopeKAddToGqaKCastEdge", kJoinOutputOpIndex, 0,
          gqaInKeyCastOpIndex,
          0
        );  // Key
      } else {
        CreateIntermediateEdge(
          "inputFromRopeQAddToGqaQCastEdge", qAddOpIndex, 0,
          gqaInQueryCastOpIndex, 0
        );  // Query
        CreateIntermediateEdge(
          "inputFromRopeKAddToGqaKCastEdge", kAddOpIndex, 0,
          gqaInKeyCastOpIndex, 0
        );  // Key
      }
      // Link the input cast nodes to MHA
      CreateIntermediateEdge(
        "inputFromGqaQCastToMhaEdge", gqaInQueryCastOpIndex, 0, gqaMhaOpIndex,
        0
      );  // Query
      CreateIntermediateEdge(
        "inputFromGqaKCastToMhaEdge", gqaInKeyCastOpIndex, 0, gqaMhaOpIndex, 1
      );  // Key
      CreateIntermediateEdge(
        "inputFromGqaVCastToMhaEdge", gqaInValueCastOpIndex, 0, gqaMhaOpIndex,
        2
      );  // value
    } else {
      if (is_partial_rope) {
        CreateIntermediateEdge(
          "inputQFromRopeQAddToMhaEdge", qJoinOutputOpIndex, 0, gqaMhaOpIndex,
          0
        );  // Query
        CreateIntermediateEdge(
          "inputKFromRopeKAddToMhaEdge", kJoinOutputOpIndex, 0, gqaMhaOpIndex, 1
        );  // Key
      } else {
        // Link the query/key/value inputs to MHA
        CreateIntermediateEdge(
          "inputQFromRopeQAddToMhaEdge", qAddOpIndex, 0, gqaMhaOpIndex, 0
        );  // Query
        CreateIntermediateEdge(
          "inputKFromRopeKAddToMhaEdge", kAddOpIndex, 0, gqaMhaOpIndex, 1
        );  // Key
      }
      CreateInputEdge(
        "inputVToMhaEdge", GqaInputIndex::gqaInputFromMatmulIndex,
        gqaMhaOpIndex, 2
      );  // Value
    }

    constexpr uint32_t dmlPastSequenceLengthsIndex = 11;

    // The GQA offline fusion does this thing where it sums the number of 1's in
    // the mask to figure out the value of the past sequence. This doesn't work
    // well for the first iteration since, obviously, there are no past
    // sequences and the mask in this case represents only the elements in the
    // initial sequence. To work around this, the CUDA implementation of the
    // operator ignores the value of pastSequenceLengths for the first iteration
    // and acts as if it was 0. This feels like a pretty dirty hack and
    // something that should be polished in the future, but for compatibility
    // with the GQA fusion and the CUDA implementation we do the same thing
    // here. We DO NOT want to do this within DirectML since DirectML should be
    // agnostic w.r.t which iteration it's currently executing MHA for, and such
    // a hack that is likely to be modified in the future shouldn't be enshrined
    // within DirectML. Doing it here is OK because the nature of contrib ops is
    // that they can change at any time.

    uint32 inputCount = GqaInputIndex::gqaInputCount;
    if (m_gqoParams.gqa.sequenceLength == 1) {
      enum GQAPastSeqLenInputIndex : uint32_t {
        pastSequenceLengthOpIndex = GqaInputIndex::gqaInputCount,
        gqaNodePastSequenceLengthCount,
      };

      inputCount = gqaNodePastSequenceLengthCount;
      // Link the PastSequenceLengths input to MHA
      CreateInputEdge(
        "inputPastSeqLengthToMhaEdge",
        GQAPastSeqLenInputIndex::pastSequenceLengthOpIndex, gqaMhaOpIndex,
        dmlPastSequenceLengthsIndex
      );
    } else {
      // Create a zero scalar edge for the past sequence lengths
      uint16_t gqaZeroScalarOpIndex = nodeCount++;

      // Link the zero scalar to MHA
      CreateIntermediateEdge(
        "zeroScalarToMhaEdge", gqaZeroScalarOpIndex, 0, gqaMhaOpIndex,
        dmlPastSequenceLengthsIndex
      );
    }

    if (isFp16) {
      // Link MHA's output to the output cast nodes
      CreateIntermediateEdge(
        "mhaToCastEdge", gqaMhaOpIndex, 0, gqaOutQueryCastOpIndex, 0
      );
      CreateIntermediateEdge(
        "mhaToCastEdge", gqaMhaOpIndex, 1, gqaOutKeyCastOpIndex, 0
      );
      CreateIntermediateEdge(
        "mhaToCastEdge", gqaMhaOpIndex, 2, gqaOutValueCastOpIndex, 0
      );

      // Link the output cast nodes to the graph's outputs
      DML_OUTPUT_GRAPH_EDGE_DESC castToOutputKeyEdge = {};
      castToOutputKeyEdge.FromNodeIndex = gqaOutKeyCastOpIndex;
      castToOutputKeyEdge.FromNodeOutputIndex = 0;
      castToOutputKeyEdge.GraphOutputIndex = 0;
      m_outputEdges.push_back(castToOutputKeyEdge);

      DML_OUTPUT_GRAPH_EDGE_DESC castToOutputValueEdge = {};
      castToOutputValueEdge.FromNodeIndex = gqaOutValueCastOpIndex;
      castToOutputValueEdge.FromNodeOutputIndex = 0;
      castToOutputValueEdge.GraphOutputIndex = 1;
      m_outputEdges.push_back(castToOutputValueEdge);
    } else {
      // Link MHA's outputs to the graph's outputs
      DML_OUTPUT_GRAPH_EDGE_DESC castToOutputKeyEdge = {};
      castToOutputKeyEdge.FromNodeIndex = gqaMhaOpIndex;
      castToOutputKeyEdge.FromNodeOutputIndex = 1;
      castToOutputKeyEdge.GraphOutputIndex = 0;
      m_outputEdges.push_back(castToOutputKeyEdge);

      DML_OUTPUT_GRAPH_EDGE_DESC castToOutputValueEdge = {};
      castToOutputValueEdge.FromNodeIndex = gqaMhaOpIndex;
      castToOutputValueEdge.FromNodeOutputIndex = 2;
      castToOutputValueEdge.GraphOutputIndex = 1;
      m_outputEdges.push_back(castToOutputValueEdge);
    }

    uint32 deQuantBOpIndex = nodeCount++;
    uint32 gemmOpIndex = nodeCount;

    uint32 matmulWeightsIndex = inputCount++;
    uint32 matmulScaleIndex = inputCount++;

    CreateInputEdge(
      "inputWeightsToDequantBEdge", matmulWeightsIndex, deQuantBOpIndex, 0
    );
    CreateInputEdge(
      "inputScaleToDequantBEdge", matmulScaleIndex, deQuantBOpIndex, 1
    );

    if (m_gqoParams.matMulNBits.hasZeroPoint) {
      uint32 matmulZeroPointIndex = inputCount++;
      CreateInputEdge(
        "inputZeroPointToDequantBEdge", matmulZeroPointIndex, deQuantBOpIndex, 2
      );
    }

    if (isFp16) {
      CreateIntermediateEdge(
        "inputFromGqaToGemmEdge", gqaOutQueryCastOpIndex, 0, gemmOpIndex, 0
      );
    } else {
      CreateIntermediateEdge(
        "inputFromGqaToGemmEdge", gqaMhaOpIndex, 0, gemmOpIndex, 0
      );
    }

    if (m_gqoParams.matMulNBits.hasC) {
      uint32 matmulCIndex = inputCount;
      CreateInputEdge("inputCToGemmEdge", matmulCIndex, gemmOpIndex, 2);
    }
    CreateIntermediateEdge(
      "inputFromDequantBToGemmEdge", deQuantBOpIndex, 0, gemmOpIndex, 1
    );

    graphInputCount = inputCount;

    DML_OUTPUT_GRAPH_EDGE_DESC gemmToOutputEdge = {};
    gemmToOutputEdge.FromNodeIndex = gemmOpIndex;
    gemmToOutputEdge.FromNodeOutputIndex = 0;
    gemmToOutputEdge.GraphOutputIndex = 2;
    m_outputEdges.push_back(gemmToOutputEdge);

    DML_GRAPH_DESC graphDesc = {};

    std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(
      static_cast<uint32_t>(m_inputEdges.size())
    );
    std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(
      static_cast<uint32_t>(m_outputEdges.size())
    );
    std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(
      static_cast<uint32_t>(m_intermediateEdges.size())
    );

    // Build the graph description from the above edges
    graphDesc.InputCount = static_cast<uint32_t>(graphInputCount);
    graphDesc.OutputCount = static_cast<uint32_t>(m_outputEdges.size());
    graphDesc.NodeCount = static_cast<uint32_t>(dmlOperators.size());
    HRESULT status = S_OK;

    graphDesc.Nodes = dmlGraphNodes.data();

    // Set the input edges
    graphDesc.InputEdgeCount = static_cast<uint32_t>(m_inputEdges.size());
    for (size_t i = 0; i < graphDesc.InputEdgeCount; ++i) {
      dmlInputEdges[i] =
        DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &m_inputEdges[i]};
    }
    graphDesc.InputEdges = dmlInputEdges.data();

    // Set the output edges
    graphDesc.OutputEdgeCount = graphDesc.OutputCount;
    for (size_t i = 0; i < graphDesc.OutputEdgeCount; ++i) {
      dmlOutputEdges[i] =
        DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &m_outputEdges[i]};
    }
    graphDesc.OutputEdges = dmlOutputEdges.data();

    // Set the intermediate edges
    graphDesc.IntermediateEdgeCount =
      static_cast<uint32_t>(m_intermediateEdges.size());
    for (size_t i = 0; i < graphDesc.IntermediateEdgeCount; ++i) {
      dmlIntermediateEdges[i] = DML_GRAPH_EDGE_DESC{
        DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &m_intermediateEdges[i]
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

    CComPtr<IDMLCompiledOperator> compiledGraph;
    status = dmlDevice1->CompileGraph(
      &graphDesc, executionFlags, IID_PPV_ARGS(&compiledGraph)
    );
    if (FAILED(status)) {
      throw std::runtime_error(
        "Failed to compile graph to a DirectML operator."
      );
    }
    m_operator = std::move(compiledGraph);
    m_isGraphConstructed = true;
  } else {
    if (rebuildOp) {
      m_fusedOps[2] = m_groupQueryOp;

      // outputs
      m_fusedOps[2]->GetOutputTensorDescVector()[0]->isOutputToNextNode =
        true;  // GQA output to Matmul
      // Inputs
      m_fusedOps[2]->GetInputTensorDescVector()[0]->isInputFromPrevNode =
        true;  // GQA 1st input from previous RopeQ
      m_fusedOps[2]->GetInputTensorDescVector()[1]->isInputFromPrevNode =
        true;  // GQA 2nd input from previous RopeK
      return;  // only gqa needs to be updated. Rest everything should stay same
    }

    m_fusedOps[0] = m_rotEmbQueryOp;
    m_fusedOps[1] = m_rotEmbKeyOp;
    m_fusedOps[2] = m_groupQueryOp;
    m_fusedOps[3] = m_internalMatmul;

    // outputs
    m_fusedOps[0]->GetOutputTensorDescVector()[0]->isOutputToNextNode =
      true;  // RopeQ output to GQA
    m_fusedOps[1]->GetOutputTensorDescVector()[0]->isOutputToNextNode =
      true;  // RopeK output to GQA
    m_fusedOps[2]->GetOutputTensorDescVector()[0]->isOutputToNextNode =
      true;  // GQA output to Matmul

    // Inputs
    m_fusedOps[2]->GetInputTensorDescVector()[0]->isInputFromPrevNode =
      true;  // GQA 1st input from previous RopeQ
    m_fusedOps[2]->GetInputTensorDescVector()[1]->isInputFromPrevNode =
      true;  // GQA 2nd input from previous RopeK
    m_fusedOps[3]->GetInputTensorDescVector()[0]->isInputFromPrevNode =
      true;  // Matmul 1st input from GQA
  }
}

// =====================================================================================================================
// Constructs an DirectML-based GQOOperator which evaluates a GQO
// operation with the given parameters.
GQOOperator::GQOOperator(
  const Context& pContext, const GQOParams& params,
  bool disableCompile
)  // Parameters for the GQO operator.
  : m_gqoParams(params), m_dataType(StringToDataType(params.gqa.dataType)) {
  const uint32_t queryHiddenSize = m_gqoParams.qkvPackedInputShape[2];
  const uint32_t queryNumHeads = m_gqoParams.gqa.num_heads;
  const uint32_t kvNumHeads = m_gqoParams.gqa.kv_num_heads;
  const uint32_t batchSize = m_gqoParams.gqa.batchSize;
  const uint32_t sequenceLength = m_gqoParams.gqa.sequenceLength;

  bool isPackedQKV = true;
  const uint32_t queryHeadSize =
    (isPackedQKV) ? (queryHiddenSize / (queryNumHeads + 2 * kvNumHeads))
                  : (queryHiddenSize / queryNumHeads);

  int64 q_size = m_gqoParams.gqa.batchSize * queryNumHeads *
                 m_gqoParams.gqa.sequenceLength * queryHeadSize;
  int64 kv_size = m_gqoParams.gqa.batchSize * kvNumHeads *
                  m_gqoParams.gqa.sequenceLength * queryHeadSize;

  m_qkvOffsetsVector.push_back(q_size * Tensor::ElementSize(m_dataType));
  m_qkvOffsetsVector.push_back(
    (q_size + kv_size) * Tensor::ElementSize(m_dataType)
  );

  // Set Rope params
  // copy the params from query to key as most of them are same
  m_gqoParams.rotEmbKey = m_gqoParams.rotEmbQuery;
  m_gqoParams.rotEmbKey.numHeads = m_gqoParams.gqa.kv_num_heads;

  // set the Q input shape for the Rotary Embedding Query
  m_gqoParams.rotEmbQuery.inputShape = {
    m_gqoParams.gqa.batchSize, m_gqoParams.gqa.sequenceLength, q_size
  };
  // set the K input shape for the Rotary Embedding Key
  m_gqoParams.rotEmbKey.inputShape = {
    m_gqoParams.gqa.batchSize, m_gqoParams.gqa.sequenceLength, kv_size
  };
  // Set the V input shape for GQA
  m_gqoParams.gqa.inputValueShape = {
    m_gqoParams.gqa.batchSize, m_gqoParams.gqa.sequenceLength, kv_size
  };

  m_gqoParams.gqa.inputQueryShape = m_gqoParams.rotEmbQuery.inputShape;
  m_gqoParams.gqa.inputKeyShape = m_gqoParams.rotEmbKey.inputShape;
}

// =====================================================================================================================
// Update node mapping for each operator
void GQOOperator::UpdateNodeMappings() {
  m_fusedOps[2]->SetInputTensorVector(
    0, m_fusedOps[0]->GetOutputTensorVector()[0]
  );  // Update tensor object 0 of
      // GQA from RopeQ
  m_fusedOps[2]->SetInputTensorVector(
    1, m_fusedOps[1]->GetOutputTensorVector()[0]
  );  // Update tensor object 1 of
      // GQA from RopeK
  m_fusedOps[3]->SetInputTensorVector(
    0, m_fusedOps[2]->GetOutputTensorVector()[0]
  );  // Update tensor object 0 of
      // Matmul from GQA
}

}  // namespace ryzenai::onnx_utils
