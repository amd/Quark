// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/gqa.h"

#include <fstream>
#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
void GQAOperator::CreateDmlCastTensorDesc(
  DML_TENSOR_DATA_TYPE dataType, const DmlTensorDesc& dmlTensorDesc,
  DmlTensorDesc& castDmlTensorDesc
) {
  std::vector<int64> strides{-1, -1, -1};  // -1
  std::vector<int64> shape{
    dmlTensorDesc.sizes[1], dmlTensorDesc.sizes[2], dmlTensorDesc.sizes[3]
  };
  std::shared_ptr<TensorDesc> tensorDesc =
    CreateTensorDesc(L"CastTensorVec", L"DHW", dataType, shape, strides);

  ConvertTensorDesc(*tensorDesc, &castDmlTensorDesc);
}

// =====================================================================================================================
void GQAOperator::CreateGQAOperator(
  const Context& pContext, std::vector<CComPtr<IDMLOperator>>& dmlOperators,
  std::shared_ptr<DmlTensorDesc>& dmlDescInRopeQ,
  std::shared_ptr<DmlTensorDesc>& dmlDescInRopeK,
  std::shared_ptr<DmlTensorDesc>& dmlDescOutMatmul
) {
  DmlTensorDesc dmlInputValueFromMatmul = {};

  DmlTensorDesc dmlPastInputKey = {};    // Caching not used in GQA for DML
  DmlTensorDesc dmlPastInputValue = {};  // Caching not used in GQA for DML
  DmlTensorDesc dmlSubCastInput = {};
  DmlTensorDesc dmlGatherCastInput = {};

  DmlTensorDesc dmlPresentKeyOutput = {};
  DmlTensorDesc dmlPresentValueOutput = {};

  std::shared_ptr<DmlTensorDesc> dmlInputToGQAFromRopeQ;
  std::shared_ptr<DmlTensorDesc> dmlInputToGQAFromRopeK;
  std::shared_ptr<DmlTensorDesc> dmlTensorOutput;

  if (dmlDescInRopeQ) {
    dmlInputToGQAFromRopeQ = dmlDescInRopeQ;
  } else {
    dmlInputToGQAFromRopeQ = std::make_shared<DmlTensorDesc>();
  }

  if (dmlDescInRopeK) {
    dmlInputToGQAFromRopeK = dmlDescInRopeK;
  } else {
    dmlInputToGQAFromRopeK = std::make_shared<DmlTensorDesc>();
  }

  if (dmlDescOutMatmul) {
    dmlTensorOutput = dmlDescOutMatmul;
  } else {
    dmlTensorOutput = std::make_shared<DmlTensorDesc>();
  }

  ConvertTensorDesc(
    *m_inputTensorDescVec[ropeQuery],
    dmlInputToGQAFromRopeQ.get()
  );  // query
  ConvertTensorDesc(
    *m_inputTensorDescVec[ropeKey],
    dmlInputToGQAFromRopeK.get()
  );  // key
  ConvertTensorDesc(
    *m_inputTensorDescVec[interMatmulValue],
    &dmlInputValueFromMatmul
  );  // value

  ConvertTensorDesc(*m_outputTensorDescVec[output], dmlTensorOutput.get());
  ConvertTensorDesc(
    *m_outputTensorDescVec[outputPresentKey], &dmlPresentKeyOutput
  );
  ConvertTensorDesc(
    *m_outputTensorDescVec[outputPresentValue], &dmlPresentValueOutput
  );

  const uint32_t sequenceLength_k = m_gqaParams.inputSubCastShape[1];
  const uint32_t totalSequenceLength = m_gqaParams.inputGatherCastShape;

  // const uint32_t kvHeadSize;

  // There will be 2 final outputs from GQO
  //  0 will be keyValue
  //  1 will be output of MatmulNBits above

  // GQA is very sensitive to overflows, so we cast all inputs to fp32 and cast
  // the outputs back to fp16. At the DML level, those casts will be eliminated
  // and replaced with half precision computation instead, which mimics the CUDA
  // EP behavior of their flash attention kernel.

  DmlTensorDesc queryCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, *dmlInputToGQAFromRopeQ.get(),
    queryCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC queryCastOpDesc{};
  queryCastOpDesc.InputTensor = &dmlInputToGQAFromRopeQ.get()->desc;
  queryCastOpDesc.OutputTensor = &queryCastDmlTensorDesc.desc;
  DML_OPERATOR_DESC queryCastDmlDesc = {DML_OPERATOR_CAST, &queryCastOpDesc};

  DmlTensorDesc keyCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, *dmlInputToGQAFromRopeK.get(),
    keyCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC keyCastOpDesc{};
  keyCastOpDesc.InputTensor = &dmlInputToGQAFromRopeK.get()->desc;
  keyCastOpDesc.OutputTensor = &keyCastDmlTensorDesc.desc;
  DML_OPERATOR_DESC keyCastDmlDesc = {DML_OPERATOR_CAST, &keyCastOpDesc};

  DmlTensorDesc valueCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, dmlInputValueFromMatmul,
    valueCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC valueCastOpDesc{};
  valueCastOpDesc.InputTensor = &dmlInputValueFromMatmul.desc;
  valueCastOpDesc.OutputTensor = &valueCastDmlTensorDesc.desc;
  DML_OPERATOR_DESC valueCastDmlDesc = {DML_OPERATOR_CAST, &valueCastOpDesc};

  // output
  DmlTensorDesc outputCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, *dmlTensorOutput.get(),
    outputCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC outputCastOpDesc{};
  outputCastOpDesc.InputTensor = &outputCastDmlTensorDesc.desc;
  outputCastOpDesc.OutputTensor = &dmlTensorOutput.get()->desc;
  DML_OPERATOR_DESC outputCastDmlDesc = {DML_OPERATOR_CAST, &outputCastOpDesc};

  // output key
  DmlTensorDesc outputPresentKeyCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, dmlPresentKeyOutput,
    outputPresentKeyCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC outputPresentKeyCastOpDesc{};
  outputPresentKeyCastOpDesc.InputTensor =
    &outputPresentKeyCastDmlTensorDesc.desc;
  outputPresentKeyCastOpDesc.OutputTensor = &dmlPresentKeyOutput.desc;
  DML_OPERATOR_DESC outputPresentKeyCastDmlDesc = {
    DML_OPERATOR_CAST, &outputPresentKeyCastOpDesc
  };

  // output value
  DmlTensorDesc outputPresentValueCastDmlTensorDesc = {};
  CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE_FLOAT32, dmlPresentValueOutput,
    outputPresentValueCastDmlTensorDesc
  );
  DML_CAST_OPERATOR_DESC outputPresentValueCastOpDesc{};
  outputPresentValueCastOpDesc.InputTensor =
    &outputPresentValueCastDmlTensorDesc.desc;
  outputPresentValueCastOpDesc.OutputTensor = &dmlPresentValueOutput.desc;
  DML_OPERATOR_DESC outputPresentValueCastDmlDesc = {
    DML_OPERATOR_CAST, &outputPresentValueCastOpDesc
  };

  const bool isFp16 =
    dmlDescInRopeQ
      ? (dmlDescInRopeQ.get()->buffer.DataType == DML_TENSOR_DATA_TYPE_FLOAT16)
      : (dmlInputToGQAFromRopeQ.get()->buffer.DataType ==
         DML_TENSOR_DATA_TYPE_FLOAT16);

  DmlTensorDesc pastSequenceLengthsDmlTensorDesc = {};
  const std::vector<int64_t> pastSequenceLengthsShape = {m_gqaParams.batchSize};
  m_inputTensorDescVec[pastSequenceLength] = CreateTensorDesc(
    L"PastSequenceLengths", L"C", DML_TENSOR_DATA_TYPE_INT32,
    pastSequenceLengthsShape, {1}
  );

  // create local resource for past sequence lengths for properly tracking local
  // attention
  m_inputTensorDescVec[pastSequenceLength]->createResource = true;
  ConvertTensorDesc(
    *m_inputTensorDescVec[pastSequenceLength], &pastSequenceLengthsDmlTensorDesc
  );

  DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC mhaDesc = {};
  mhaDesc.QueryTensor =
    isFp16 ? &queryCastDmlTensorDesc.desc : &dmlInputToGQAFromRopeQ.get()->desc;
  mhaDesc.KeyTensor =
    isFp16 ? &keyCastDmlTensorDesc.desc : &dmlInputToGQAFromRopeK.get()->desc;
  mhaDesc.ValueTensor =
    isFp16 ? &valueCastDmlTensorDesc.desc : &dmlInputValueFromMatmul.desc;
  mhaDesc.PastSequenceLengthsTensor = &pastSequenceLengthsDmlTensorDesc.desc;

  mhaDesc.OutputTensor =
    isFp16 ? &outputCastDmlTensorDesc.desc : &dmlTensorOutput.get()->desc;
  mhaDesc.OutputPresentKeyTensor = isFp16
                                     ? &outputPresentKeyCastDmlTensorDesc.desc
                                     : &dmlPresentKeyOutput.desc;
  mhaDesc.OutputPresentValueTensor =
    isFp16 ? &outputPresentValueCastDmlTensorDesc.desc
           : &dmlPresentValueOutput.desc;
  mhaDesc.QueryHeadCount = m_gqaParams.num_heads;
  mhaDesc.KeyValueHeadCount = m_gqaParams.kv_num_heads;
  mhaDesc.Scale = m_gqaParams.scale;
  mhaDesc.MaskFilterValue = -10'000.0f;
  DML_OPERATOR_DESC mhaDmlDesc = {DML_OPERATOR_MULTIHEAD_ATTENTION1, &mhaDesc};

  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zeroScalarDesc = {};
  zeroScalarDesc.OutputTensor = &pastSequenceLengthsDmlTensorDesc.desc;
  zeroScalarDesc.ValueDataType =
    pastSequenceLengthsDmlTensorDesc.buffer.DataType;
  DML_OPERATOR_DESC zeroScalarDmlDesc = {
    DML_OPERATOR_FILL_VALUE_CONSTANT, &zeroScalarDesc
  };

  std::vector<const DML_OPERATOR_DESC*> opDescs = {
    &mhaDmlDesc,
  };
  if (isFp16) {
    opDescs.push_back(&queryCastDmlDesc);
    opDescs.push_back(&keyCastDmlDesc);
    opDescs.push_back(&valueCastDmlDesc);
    opDescs.push_back(&outputCastDmlDesc);
    opDescs.push_back(&outputPresentKeyCastDmlDesc);
    opDescs.push_back(&outputPresentValueCastDmlDesc);
  }
  if (m_gqaParams.sequenceLength != 1) {
    opDescs.push_back(&zeroScalarDmlDesc);
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
      throw std::runtime_error("Failed to compile a DirectML operator.");
    }
  }
}

// =====================================================================================================================
// Constructs an DirectML-based GQAOperator which evaluates a Gemm
// operation with the given parameters.
GQAOperator::GQAOperator(
  const Context& pContext, const GQAParams& params,
  bool disableCompile
)  // Parameters for the SSMLP operator.
  : m_gqaParams(params), m_dataType(StringToDataType(params.dataType)) {
  SharedInit();

  if (disableCompile) return;

  std::vector<CComPtr<IDMLOperator>> dmlOperators;
  std::shared_ptr<DmlTensorDesc> nullDmlTensor = nullptr;
  CreateGQAOperator(
    pContext, dmlOperators, nullDmlTensor, nullDmlTensor, nullDmlTensor
  );

  DML_GRAPH_DESC graphDesc = {};
  std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );

  std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );

  for (size_t i = 0; i < dmlOperators.size(); ++i) {
    dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{dmlOperators[i]};
    dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{
      DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]
    };
  }

  // Construct the graph
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

  const bool isFp16 = (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16);

  if (isFp16) {
    // Link the query/key/value inputs to the cast nodes
    for (uint32_t i = 0; i < 3; ++i) {
      DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
      inputToMhaEdge.GraphInputIndex = i;
      inputToMhaEdge.ToNodeIndex = 1 + i;
      inputToMhaEdge.ToNodeInputIndex = 0;
      inputEdges.push_back(inputToMhaEdge);
    }

    // Link the input cast nodes to MHA
    for (uint32_t i = 0; i < 3; ++i) {
      DML_INTERMEDIATE_GRAPH_EDGE_DESC castToMhaEdge = {};
      castToMhaEdge.FromNodeIndex = 1 + i;
      castToMhaEdge.FromNodeOutputIndex = 0;
      castToMhaEdge.ToNodeIndex = 0;
      castToMhaEdge.ToNodeInputIndex = i;
      intermediateEdges.push_back(castToMhaEdge);
    }
  } else {
    // Link the query/key/value inputs to MHA
    for (uint32_t i = 0; i < 3; ++i) {
      DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
      inputToMhaEdge.GraphInputIndex = i;
      inputToMhaEdge.ToNodeIndex = 0;
      inputToMhaEdge.ToNodeInputIndex = i;
      inputEdges.push_back(inputToMhaEdge);
    }
  }

  constexpr uint32_t dmlPastSequenceLengthsIndex = 11;

  // The GQA offline fusion does this thing where it sums the number of 1's in
  // the mask to figure out the value of the past sequence. This doesn't work
  // well for the first iteration since, obviously, there are no past sequences
  // and the mask in this case represents only the elements in the initial
  // sequence. To work around this, the CUDA implementation of the operator
  // ignores the value of pastSequenceLengths for the first iteration and acts
  // as if it was 0. This feels like a pretty dirty hack and something that
  // should be polished in the future, but for compatibility with the GQA fusion
  // and the CUDA implementation we do the same thing here. We DO NOT want to do
  // this within DirectML since DirectML should be agnostic w.r.t which
  // iteration it's currently executing MHA for, and such a hack that is likely
  // to be modified in the future shouldn't be enshrined within DirectML. Doing
  // it here is OK because the nature of contrib ops is that they can change at
  // any time.
  if (m_gqaParams.sequenceLength == 1) {
    // Link the PastSequenceLengths input to MHA
    DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
    inputToMhaEdge.GraphInputIndex = inputEdges.size();
    inputToMhaEdge.ToNodeIndex = 0;
    inputToMhaEdge.ToNodeInputIndex = dmlPastSequenceLengthsIndex;
    inputEdges.push_back(inputToMhaEdge);
  } else {
    // Link the zero scalar to MHA
    DML_INTERMEDIATE_GRAPH_EDGE_DESC zeroScalarToMhaEdge = {};
    zeroScalarToMhaEdge.FromNodeIndex = dmlOperators.size() - 1;
    zeroScalarToMhaEdge.FromNodeOutputIndex = 0;
    zeroScalarToMhaEdge.ToNodeIndex = 0;
    zeroScalarToMhaEdge.ToNodeInputIndex = dmlPastSequenceLengthsIndex;
    intermediateEdges.push_back(zeroScalarToMhaEdge);
  }

  if (isFp16) {
    // Output cast nodes start at the 4th index (previously we have the mha,
    // query, key and value nodes)
    const uint32_t outputCastNodeStart = 4;

    // Link MHA's output to the output cast nodes
    for (uint32_t i = 0; i < 3; ++i) {
      DML_INTERMEDIATE_GRAPH_EDGE_DESC mhaToCastEdge = {};
      mhaToCastEdge.FromNodeIndex = 0;
      mhaToCastEdge.FromNodeOutputIndex = i;
      mhaToCastEdge.ToNodeIndex = outputCastNodeStart + i;
      mhaToCastEdge.ToNodeInputIndex = 0;
      intermediateEdges.push_back(mhaToCastEdge);
    }

    // Link the output cast nodes to the graph's outputs
    for (uint32_t i = 0; i < 3; ++i) {
      DML_OUTPUT_GRAPH_EDGE_DESC castToOutputEdge = {};
      castToOutputEdge.FromNodeIndex = outputCastNodeStart + i;
      castToOutputEdge.FromNodeOutputIndex = 0;
      castToOutputEdge.GraphOutputIndex = i;
      outputEdges.push_back(castToOutputEdge);
    }
  } else {
    // Link MHA's outputs to the graph's outputs
    for (uint32_t i = 0; i < 3; ++i) {
      DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
      mhaToOutputEdge.FromNodeIndex = 0;
      mhaToOutputEdge.FromNodeOutputIndex = i;
      mhaToOutputEdge.GraphOutputIndex = i;
      outputEdges.push_back(mhaToOutputEdge);
    }
  }

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
  graphDesc.NodeCount = static_cast<uint32_t>(dmlOperators.size());
  HRESULT status = S_OK;

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
    throw std::runtime_error("Failed to compile a DirectML operator.");
  }
}

// =====================================================================================================================
// Code shared between our constructors.
void GQAOperator::SharedInit() {
#ifdef DEBUG
  if (!(m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16 ||
        m_dataType == DML_TENSOR_DATA_TYPE_FLOAT32)) {
    std::wstringstream ss;
    ss << "MLP only supports Float16, not " << m_gqaParams.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }
#endif

  const std::vector<int64> strides2d{-1, -1};          // -1
  const std::vector<int64> strides{-1, -1, -1};        // -1
  const std::vector<int64> strides4d{-1, -1, -1, -1};  // -1

  m_inputTensorDescVec.resize(inputCount);

  m_inputTensorDescVec[ropeQuery] = CreateTensorDesc(
    L"InputToGQAFromRopeQ", L"DHW", m_dataType, m_gqaParams.inputQueryShape,
    strides
  );

  m_inputTensorDescVec[ropeKey] = CreateTensorDesc(
    L"InputToGQAFromRopeK", L"DHW", m_dataType, m_gqaParams.inputKeyShape,
    strides
  );

  m_inputTensorDescVec[interMatmulValue] = CreateTensorDesc(
    L"InputToGQAFromintMatmulV", L"DHW", m_dataType,
    m_gqaParams.inputValueShape, strides
  );

  m_inputTensorDescVec[interMatmulValue] = CreateTensorDesc(
    L"InputToGQAFromintMatmulV", L"DHW", m_dataType,
    m_gqaParams.inputValueShape, strides
  );

  // Outputs
  m_outputTensorDescVec.resize(outputCount);
  m_outputTensorDescVec[output] = CreateTensorDesc(
    L"OutputTensor", L"DHW", m_dataType,
    {m_gqaParams.batchSize, m_gqaParams.sequenceLength,
     m_gqaParams.inputQueryShape[2]},
    strides
  );

  std::vector<int64_t> outputKVShape = m_gqaParams.outputKeyShape;
  bool isLocalAttention =
    (m_gqaParams.local_window_size > -1) &&
    (m_gqaParams.outputKeyShape[2] > m_gqaParams.local_window_size);
  if (isLocalAttention) {
    outputKVShape[2] =
      m_gqaParams.local_window_size;  // set seq length to local window size
  }

  // Shape for KV will match past KV shape
  m_outputTensorDescVec[outputPresentKey] = CreateTensorDesc(
    L"OutputPresentKeyTensor", L"DHW", m_dataType, outputKVShape, strides4d
  );
  m_outputTensorDescVec[outputPresentValue] = CreateTensorDesc(
    L"OutputPresentValueTensor", L"DHW", m_dataType, outputKVShape, strides4d
  );
}

}  // namespace ryzenai::onnx_utils
