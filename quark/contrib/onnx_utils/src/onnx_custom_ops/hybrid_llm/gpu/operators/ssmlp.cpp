// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/ssmlp.h"

#include <fstream>
#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

DML_INPUT_GRAPH_EDGE_DESC SSMLPOperator::CreateInputEdge(
  uint32 graphInputIndex, uint32 toNodeIndex, uint32 toNodeInputIndex
) {
  DML_INPUT_GRAPH_EDGE_DESC edge = {};
  edge.GraphInputIndex = graphInputIndex;
  edge.ToNodeIndex = toNodeIndex;
  edge.ToNodeInputIndex = toNodeInputIndex;
  return edge;
}

DML_INTERMEDIATE_GRAPH_EDGE_DESC SSMLPOperator::CreateIntermediateEdge(
  uint32 fromNodeIndex, uint32 fromNodeOutputIndex, uint32 toNodeIndex,
  uint32 toNodeInputIndex
) {
  DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
  edge.FromNodeIndex = fromNodeIndex;
  edge.FromNodeOutputIndex = fromNodeOutputIndex;
  edge.ToNodeIndex = toNodeIndex;
  edge.ToNodeInputIndex = toNodeInputIndex;
  return edge;
}

void SSMLPOperator::DequantizeBForMatmul(
  DmlTensorDesc& dmlTensorQBDequant, const TensorDesc* inputTensorDesc,
  const DML_TENSOR_DATA_TYPE& dequantDataType
) {
  TensorDesc dequantBTensor = *inputTensorDesc;
  dequantBTensor.dataType = dequantDataType;
  dequantBTensor.name = L"dequantB";
  if (dequantBTensor.dataType == DML_TENSOR_DATA_TYPE_FLOAT32) {
    dequantBTensor.totalBytes *= 2 * sizeof(float);  // 4 bits to 32 bit
  } else if (dequantBTensor.dataType == DML_TENSOR_DATA_TYPE_FLOAT16) {
    dequantBTensor.totalBytes *= 2 * sizeof(uint16_t);  // 4 bits to 16 bit
  } else {
    // bad things happen here
    assert("dequantBTensor.dataType is not Float or Float16");
  }

  ConvertTensorDesc(dequantBTensor, &dmlTensorQBDequant);
}

// =====================================================================================================================
// Constructs an DirectML-based SSMLPOperator which evaluates a Gemm
// operation with the given parameters.
SSMLPOperator::SSMLPOperator(
  const Context& pContext, bool disableMetacmds,
  const SSMLPParms& params
)  // Parameters for the SSMLP operator.
  : m_ssMlp(params),
    m_dataType(StringToDataType(params.sslrnTop.dataType)),
    m_quantDataType(StringToDataType(params.matMulNBitsGate.quantizedB)) {
  SharedInit();

  // SSLRNTop
  // ----------------------------------------------------------------------------------
  const float epsilon = m_ssMlp.sslrnTop.epsilon;
  std::array<uint32_t, 2> axes = {2, 3};

  DmlTensorDesc sslrnTop_inputDesc = {};
  DmlTensorDesc sslrnTop_skipDesc = {};
  DmlTensorDesc sslrnTop_mvnScaleDesc = {};
  DmlTensorDesc sslrnTop_outputDesc = {};

  // main input to ssmlp goes to sslrntop
  ConvertTensorDesc(*m_inputTensorDescVec[0], &sslrnTop_inputDesc);
  // skip input to sslrntop
  ConvertTensorDesc(*m_inputTensorDescVec[1], &sslrnTop_skipDesc);

  if (m_ssMlp.sslrnTop.hasScale)
    ConvertTensorDesc(*m_inputTensorDescVec[2], &sslrnTop_mvnScaleDesc);

  if (m_ssMlp.sslrnTop.hasBias) assert("m_ssMlp.sslrnTop.hasBias = true");

  if (m_ssMlp.sslrnTop.hasNonMVNBias)
    assert("m_ssMlp.sslrnTop.hasNonMVNBias = true");

  // output
  ConvertTensorDesc(*m_inputTensorDescVec[0], &sslrnTop_outputDesc);

  // MatMulNBits Gate
  // ----------------------------------------------------------------------------------
  DmlTensorDesc mmGate_dmlTensorB = {};
  DmlTensorDesc mmGate_dmlTensorQBS = {};
  DmlTensorDesc mmGate_dmlTensorQBZ = {};
  DmlTensorDesc mmGate_outputDesc = {};

  ConvertTensorDesc(*m_inputTensorDescVec[3], &mmGate_dmlTensorB);
  ConvertTensorDesc(*m_inputTensorDescVec[4], &mmGate_dmlTensorQBS);
  if (m_ssMlp.matMulNBitsGate.hasZeroPoint) {
    auto& zp = m_inputTensorDescVec[5];
    MarkRepackIfOddDim(zp, pContext.IsGpuJitEnabled());
    ConvertTensorDesc(*zp, &mmGate_dmlTensorQBZ);
  }

  const std::vector<int64> sizeY{
    m_ssMlp.matMulNBitsGate.batches, m_ssMlp.matMulNBitsGate.m,
    m_ssMlp.matMulNBitsGate.n
  };

  const std::vector<int64> stridesY{
    m_ssMlp.matMulNBitsGate.stridesY[0], m_ssMlp.matMulNBitsGate.stridesY[1],
    m_ssMlp.matMulNBitsGate.stridesY[2]
  };

  std::shared_ptr<TensorDesc> gateGemmOutput = CreateTensorDesc(
    L"gateGemmOutput", m_ssMlp.matMulNBitsGate.orderY,
    StringToDataType(m_ssMlp.matMulNBitsGate.dataType), sizeY, stridesY
  );
  ConvertTensorDesc(*gateGemmOutput, &mmGate_outputDesc);

  if (m_ssMlp.matMulNBitsGate.hasC) {
    assert("m_ssMlp.matMulNBitsGate.hasC = true");
  }

  // Sigmoid
  // ----------------------------------------------------------------------------------
  DmlTensorDesc sigmoid_outputDesc = {};
  ConvertTensorDesc(*gateGemmOutput, &sigmoid_outputDesc);

  // Matmul_Up
  // ----------------------------------------------------------------------------------
  DmlTensorDesc mmUp_dmlTensorB = {};
  DmlTensorDesc mmUp_dmlTensorQBS = {};
  DmlTensorDesc mmUp_dmlTensorQBZ = {};
  DmlTensorDesc mmUp_outputDesc = {};

  ConvertTensorDesc(*m_inputTensorDescVec[6], &mmUp_dmlTensorB);
  ConvertTensorDesc(*m_inputTensorDescVec[7], &mmUp_dmlTensorQBS);
  if (m_ssMlp.matMulNBitsUp.hasZeroPoint) {
    auto& zp = m_inputTensorDescVec[8];
    MarkRepackIfOddDim(zp, pContext.IsGpuJitEnabled());
    ConvertTensorDesc(*zp, &mmUp_dmlTensorQBZ);
  }

  if (m_ssMlp.matMulNBitsUp.hasC) {
    // ConvertTensorDesc(*m_inputTensorDescVec[4], &mmGate_dmlTensorC);
    assert("m_ssMlp.matMulNBitsUp.hasC = true");
  }

  const std::vector<int64> sizeYUp{
    m_ssMlp.matMulNBitsUp.batches, m_ssMlp.matMulNBitsUp.m,
    m_ssMlp.matMulNBitsUp.n
  };

  std::shared_ptr<TensorDesc> upGemmOutput = CreateTensorDesc(
    L"upGemmOutput", m_ssMlp.matMulNBitsUp.orderY,
    StringToDataType(m_ssMlp.matMulNBitsUp.dataType), sizeYUp, stridesY
  );
  ConvertTensorDesc(*upGemmOutput, &mmUp_outputDesc);

  // Matmul_Down
  // ----------------------------------------------------------------------------------
  DmlTensorDesc mmDown_dmlTensorB = {};
  DmlTensorDesc mmDown_dmlTensorQBS = {};
  DmlTensorDesc mmDown_dmlTensorQBZ = {};
  DmlTensorDesc mmDown_outputDesc = {};

  ConvertTensorDesc(*m_inputTensorDescVec[9], &mmDown_dmlTensorB);
  ConvertTensorDesc(*m_inputTensorDescVec[10], &mmDown_dmlTensorQBS);
  if (m_ssMlp.matMulNBitsDown.hasZeroPoint) {
    auto& zp = m_inputTensorDescVec[11];
    MarkRepackIfOddDim(zp, pContext.IsGpuJitEnabled());
    ConvertTensorDesc(*zp, &mmDown_dmlTensorQBZ);
  }

  if (m_ssMlp.matMulNBitsDown.hasC) {
    // ConvertTensorDesc(*m_inputTensorDescVec[4], &mmGate_dmlTensorC);
    assert("m_ssMlp.matMulNBitsDown.hasC = true");
  }

  const std::vector<int64> sizeYDown{
    m_ssMlp.matMulNBitsDown.batches, m_ssMlp.matMulNBitsDown.m,
    m_ssMlp.matMulNBitsDown.n
  };

  std::shared_ptr<TensorDesc> downGemmOutput = CreateTensorDesc(
    L"downGemmOutput", m_ssMlp.matMulNBitsDown.orderY,
    StringToDataType(m_ssMlp.matMulNBitsDown.dataType), sizeYDown, stridesY
  );
  ConvertTensorDesc(*downGemmOutput, &mmDown_outputDesc);

  // SSLRNBottom
  // ----------------------------------------------------------------------------------
  DmlTensorDesc sslrnBottom_mvnScaleDesc = {};
  if (m_ssMlp.sslrnTop.hasScale)
    ConvertTensorDesc(*m_inputTensorDescVec[12], &sslrnBottom_mvnScaleDesc);

  // Main output off the node
  DmlTensorDesc outputDesc = {};
  DmlTensorDesc inputSkipBiasSum = {};

  ConvertTensorDesc(*m_outputTensorDescVec[0], &outputDesc);

  if (m_ssMlp.sslrnBottom.outputCount > 1) {
    ConvertTensorDesc(*m_outputTensorDescVec[1], &inputSkipBiasSum);
  } else {
    ConvertTensorDesc(*m_tmpTensorDescSkipAdd, &inputSkipBiasSum);
  }

  // List of Ops for SSMLP
  // ===================================================================================================

  // SSLRN
  // ---------------------------------------------------------------------------------------------
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC paSkipAddDesc = {};
  paSkipAddDesc.ATensor = &sslrnTop_inputDesc.desc;
  paSkipAddDesc.BTensor = &sslrnTop_skipDesc.desc;
  paSkipAddDesc.OutputTensor = &sslrnTop_inputDesc.desc;
  DML_OPERATOR_DESC paSkipAddOpDesc = {
    DML_OPERATOR_ELEMENT_WISE_ADD, &paSkipAddDesc
  };

  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC paMvnDesc = {};
  paMvnDesc.InputTensor = &sslrnTop_inputDesc.desc;
  paMvnDesc.ScaleTensor =
    m_ssMlp.sslrnTop.hasScale ? &sslrnTop_mvnScaleDesc.desc : nullptr;
  paMvnDesc.BiasTensor = nullptr;
  paMvnDesc.OutputTensor = &sslrnTop_outputDesc.desc;
  paMvnDesc.Axes = axes.data();
  paMvnDesc.AxisCount = axes.size();
  paMvnDesc.UseMean = false;
  paMvnDesc.UseVariance = true;
  paMvnDesc.Epsilon = epsilon;
  paMvnDesc.FusedActivation = nullptr;
  DML_OPERATOR_DESC paMvnOpDesc = {
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2, &paMvnDesc
  };

  // GATE MATMULNBITS
  // ----------------------------------------------------------------------------------
  DmlTensorDesc dmlTensorQBDequant = {};
  DequantizeBForMatmul(
    dmlTensorQBDequant, m_inputTensorDescVec[3].get(), m_dataType
  );

  std::vector<DML_TENSOR_DESC> quantizationParametersTensors;
  quantizationParametersTensors.push_back(mmGate_dmlTensorQBS.desc);

  if (m_ssMlp.matMulNBitsGate.hasZeroPoint) {
    quantizationParametersTensors.push_back(mmGate_dmlTensorQBZ.desc);
  }

  DML_DEQUANTIZE_OPERATOR_DESC gateDequantizeDesc = {};
  gateDequantizeDesc.InputTensor = &mmGate_dmlTensorB.desc;
  gateDequantizeDesc.QuantizationType =
    m_ssMlp.matMulNBitsGate.hasZeroPoint
      ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
      : DML_QUANTIZATION_TYPE_SCALE;

  gateDequantizeDesc.QuantizationTensorCount =
    static_cast<uint32_t>(quantizationParametersTensors.size());
  gateDequantizeDesc.QuantizationTensors = quantizationParametersTensors.data();
  gateDequantizeDesc.OutputTensor = &dmlTensorQBDequant.desc;
  DML_OPERATOR_DESC gateDequantizeOpDesc = {
    DML_OPERATOR_DEQUANTIZE, &gateDequantizeDesc
  };

  DML_GEMM_OPERATOR_DESC gateGemmDesc = {};
  gateGemmDesc.ATensor = &sslrnTop_outputDesc.desc;
  gateGemmDesc.BTensor = &dmlTensorQBDequant.desc;
  gateGemmDesc.CTensor = nullptr;
  gateGemmDesc.OutputTensor = &mmGate_outputDesc.desc;
  gateGemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
  gateGemmDesc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
  gateGemmDesc.Alpha = 1.0f;
  gateGemmDesc.Beta = 0.0f;
  DML_OPERATOR_DESC gateGemmOpDesc = {DML_OPERATOR_GEMM, &gateGemmDesc};

  // SIGMOID
  // ----------------------------------------------------------------------------------
  DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoidDesc;
  sigmoidDesc.InputTensor = &mmGate_outputDesc.desc;
  sigmoidDesc.OutputTensor = &sigmoid_outputDesc.desc;
  DML_OPERATOR_DESC sigmoidOpDesc = {
    DML_OPERATOR_ACTIVATION_SIGMOID, &sigmoidDesc
  };

  // MUL
  // ----------------------------------------------------------------------------------
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC actMulDesc = {};
  actMulDesc.ATensor = &mmGate_outputDesc.desc;
  actMulDesc.BTensor = &sigmoid_outputDesc.desc;
  actMulDesc.OutputTensor = &sigmoid_outputDesc.desc;
  DML_OPERATOR_DESC actMulOpDesc = {
    DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &actMulDesc
  };

  std::vector<DML_TENSOR_DESC> upQuantizationParametersTensors;
  upQuantizationParametersTensors.push_back(mmUp_dmlTensorQBS.desc);

  if (m_ssMlp.matMulNBitsUp.hasZeroPoint) {
    upQuantizationParametersTensors.push_back(mmUp_dmlTensorQBZ.desc);
  }

  // UP MATMULNBITS
  // ----------------------------------------------------------------------------------

  DmlTensorDesc dmlTensorQBDequantUp = {};
  DequantizeBForMatmul(
    dmlTensorQBDequantUp, m_inputTensorDescVec[6].get(), m_dataType
  );

  DML_DEQUANTIZE_OPERATOR_DESC upDequantizeDesc = {};
  upDequantizeDesc.InputTensor = &mmUp_dmlTensorB.desc;
  upDequantizeDesc.QuantizationType = m_ssMlp.matMulNBitsUp.hasZeroPoint
                                        ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
                                        : DML_QUANTIZATION_TYPE_SCALE;

  upDequantizeDesc.QuantizationTensorCount =
    static_cast<uint32_t>(upQuantizationParametersTensors.size());
  upDequantizeDesc.QuantizationTensors = upQuantizationParametersTensors.data();
  upDequantizeDesc.OutputTensor = &dmlTensorQBDequantUp.desc;
  DML_OPERATOR_DESC upDequantizeOpDesc = {
    DML_OPERATOR_DEQUANTIZE, &upDequantizeDesc
  };

  DML_GEMM_OPERATOR_DESC upGemmDesc = {};
  upGemmDesc.ATensor = &sslrnTop_outputDesc.desc;
  upGemmDesc.BTensor = &dmlTensorQBDequantUp.desc;
  upGemmDesc.CTensor = nullptr;
  upGemmDesc.OutputTensor = &mmUp_outputDesc.desc;
  upGemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
  upGemmDesc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
  upGemmDesc.Alpha = 1.0f;
  upGemmDesc.Beta = 0.0f;
  DML_OPERATOR_DESC upGemmOpDesc = {DML_OPERATOR_GEMM, &upGemmDesc};

  // MUL
  // ----------------------------------------------------------------------------------
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulDesc = {};
  mulDesc.ATensor = &sigmoid_outputDesc.desc;
  mulDesc.BTensor = &mmUp_outputDesc.desc;
  mulDesc.OutputTensor = &sigmoid_outputDesc.desc;
  DML_OPERATOR_DESC mulOpDesc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulDesc};

  // DOWN MATMULNBITS
  // ----------------------------------------------------------------------------------

  std::vector<DML_TENSOR_DESC> downQuantizationParametersTensors;
  downQuantizationParametersTensors.push_back(mmDown_dmlTensorQBS.desc);

  if (m_ssMlp.matMulNBitsDown.hasZeroPoint) {
    downQuantizationParametersTensors.push_back(mmDown_dmlTensorQBZ.desc);
  }

  DmlTensorDesc dmlTensorQBDequantDown = {};
  DequantizeBForMatmul(
    dmlTensorQBDequantDown, m_inputTensorDescVec[9].get(), m_dataType
  );

  DML_DEQUANTIZE_OPERATOR_DESC downDequantizeDesc = {};
  downDequantizeDesc.InputTensor = &mmDown_dmlTensorB.desc;
  downDequantizeDesc.QuantizationType =
    m_ssMlp.matMulNBitsDown.hasZeroPoint
      ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
      : DML_QUANTIZATION_TYPE_SCALE;

  downDequantizeDesc.QuantizationTensorCount =
    static_cast<uint32_t>(downQuantizationParametersTensors.size());
  downDequantizeDesc.QuantizationTensors =
    downQuantizationParametersTensors.data();
  downDequantizeDesc.OutputTensor = &dmlTensorQBDequantDown.desc;
  DML_OPERATOR_DESC downDequantizeOpDesc = {
    DML_OPERATOR_DEQUANTIZE, &downDequantizeDesc
  };

  DML_GEMM_OPERATOR_DESC downGemmDesc = {};
  downGemmDesc.ATensor = &sigmoid_outputDesc.desc;
  downGemmDesc.BTensor = &dmlTensorQBDequantDown.desc;
  downGemmDesc.CTensor = nullptr;
  downGemmDesc.OutputTensor = &mmDown_outputDesc.desc;
  downGemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
  downGemmDesc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
  downGemmDesc.Alpha = 1.0f;
  downGemmDesc.Beta = 0.0f;
  DML_OPERATOR_DESC downGemmOpDesc = {DML_OPERATOR_GEMM, &downGemmDesc};

  // SSLRN
  // ---------------------------------------------------------------------------------------------
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC skipAddDesc = {};
  skipAddDesc.ATensor = &sslrnTop_inputDesc.desc;
  skipAddDesc.BTensor = &mmDown_outputDesc.desc;
  skipAddDesc.OutputTensor = &inputSkipBiasSum.desc;
  DML_OPERATOR_DESC skipAddOpDesc = {
    DML_OPERATOR_ELEMENT_WISE_ADD, &skipAddDesc
  };

  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC mvnDesc = {};
  mvnDesc.InputTensor = &inputSkipBiasSum.desc;
  mvnDesc.ScaleTensor =
    m_ssMlp.sslrnTop.hasScale ? &sslrnTop_mvnScaleDesc.desc : nullptr;
  mvnDesc.BiasTensor = nullptr;
  mvnDesc.OutputTensor = &outputDesc.desc;
  mvnDesc.Axes = axes.data();
  mvnDesc.AxisCount = axes.size();
  mvnDesc.UseMean = false;
  mvnDesc.UseVariance = true;
  mvnDesc.Epsilon = epsilon;
  mvnDesc.FusedActivation = nullptr;
  DML_OPERATOR_DESC mvnOpDesc = {
    DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2, &mvnDesc
  };

  // Construct the graph
  std::vector<const DML_OPERATOR_DESC*> opDescs;
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

  // TOP SSLRN
  // ------------------------------------------------------------------------
  opDescs.push_back(&paSkipAddOpDesc);
  DML_INPUT_GRAPH_EDGE_DESC dataInputEdge =
    CreateInputEdge(InputIndex::inputIndex, NodeIndex::paSkipAddOpIndex, 0);
  inputEdges.push_back(dataInputEdge);
  DML_INPUT_GRAPH_EDGE_DESC skipInputEdge =
    CreateInputEdge(InputIndex::skipIndex, NodeIndex::paSkipAddOpIndex, 1);
  inputEdges.push_back(skipInputEdge);

  // TO DO --------------
  // Check for SKIP BIAS

  // MVN
  opDescs.push_back(&paMvnOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC paSkipAddToMvnEdge = CreateIntermediateEdge(
    NodeIndex::paSkipAddOpIndex, 0, NodeIndex::paMvnOpIndex, 0
  );
  intermediateEdges.push_back(paSkipAddToMvnEdge);
  DML_INPUT_GRAPH_EDGE_DESC gammaInputEdge =
    CreateInputEdge(InputIndex::topGammaIndex, NodeIndex::paMvnOpIndex, 1);
  inputEdges.push_back(gammaInputEdge);

  // GATE MATMULNBITS
  // ------------------------------------------------------------------------
  // input to gemm
  opDescs.push_back(&gateDequantizeOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToGateDequantizeEdge =
    CreateIntermediateEdge(
      NodeIndex::paMvnOpIndex, 0, NodeIndex::gateGemmOpIndex, 0
    );
  intermediateEdges.push_back(mvnToGateDequantizeEdge);

  // input Weights to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC gateDequantizeInputEdge = CreateInputEdge(
    InputIndex::weightsGateQ4Index, NodeIndex::gateDequantizeOpIndex, 0
  );
  inputEdges.push_back(gateDequantizeInputEdge);

  // input Scale to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC gateDequantizeScaleEdge = CreateInputEdge(
    InputIndex::weightGateScaleIndex, NodeIndex::gateDequantizeOpIndex, 1
  );
  inputEdges.push_back(gateDequantizeScaleEdge);

  // input Zero Point to Dequantize
  if (m_ssMlp.matMulNBitsGate.hasZeroPoint) {
    DML_INPUT_GRAPH_EDGE_DESC gateDequantizeZeroPointEdge = CreateInputEdge(
      InputIndex::weightGateZeroPointIndex, NodeIndex::gateDequantizeOpIndex, 2
    );
    inputEdges.push_back(gateDequantizeZeroPointEdge);
  }

  // TO DO --------------
  // Check for C

  // input to gemm from dequantize
  opDescs.push_back(&gateGemmOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC dequantizeToGemmEdge =
    CreateIntermediateEdge(
      NodeIndex::gateDequantizeOpIndex, 0, NodeIndex::gateGemmOpIndex, 1
    );
  intermediateEdges.push_back(dequantizeToGemmEdge);

  // SIGMOID
  // ------------------------------------------------------------------------
  opDescs.push_back(&sigmoidOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC gateGemmToSigmoidEdge =
    CreateIntermediateEdge(
      NodeIndex::gateGemmOpIndex, 0, NodeIndex::sigmoidOpIndex, 0
    );
  intermediateEdges.push_back(gateGemmToSigmoidEdge);

  // MUL
  // ------------------------------------------------------------------------
  opDescs.push_back(&actMulOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC sigmoidToMulEdge = CreateIntermediateEdge(
    NodeIndex::sigmoidOpIndex, 0, NodeIndex::actMulOpIndex, 0
  );
  intermediateEdges.push_back(sigmoidToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC gateGemmToMulEdge = CreateIntermediateEdge(
    NodeIndex::gateGemmOpIndex, 0, NodeIndex::actMulOpIndex, 1
  );
  intermediateEdges.push_back(gateGemmToMulEdge);

  // UP MATMULNBITS
  // ------------------------------------------------------------------------
  // input to gemm
  opDescs.push_back(&upDequantizeOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToUpDequantizeEdge =
    CreateIntermediateEdge(
      NodeIndex::paMvnOpIndex, 0, NodeIndex::upGemmOpIndex, 0
    );
  intermediateEdges.push_back(mvnToUpDequantizeEdge);

  // input Weights to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC upDequantizeInputEdge = CreateInputEdge(
    InputIndex::weightsUpQ4Index, NodeIndex::upDequantizeOpIndex, 0
  );
  inputEdges.push_back(upDequantizeInputEdge);

  // input Scale to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC upDequantizeScaleEdge = CreateInputEdge(
    InputIndex::weightUpScaleIndex, NodeIndex::upDequantizeOpIndex, 1
  );
  inputEdges.push_back(upDequantizeScaleEdge);

  // input Zero Point to Dequantize
  if (m_ssMlp.matMulNBitsUp.hasZeroPoint) {
    DML_INPUT_GRAPH_EDGE_DESC upDequantizeZeroPointEdge = CreateInputEdge(
      InputIndex::weightUpZeroPointIndex, NodeIndex::upDequantizeOpIndex, 2
    );
    inputEdges.push_back(upDequantizeZeroPointEdge);
  }

  // TO DO --------------
  // Check for C

  // input to gemm from dequantize
  opDescs.push_back(&upGemmOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC upDequantizeToGemmEdge =
    CreateIntermediateEdge(
      NodeIndex::upDequantizeOpIndex, 0, NodeIndex::upGemmOpIndex, 1
    );
  intermediateEdges.push_back(upDequantizeToGemmEdge);

  // MUL
  // ------------------------------------------------------------------------
  opDescs.push_back(&mulOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC actMulToMulEdge = CreateIntermediateEdge(
    NodeIndex::actMulOpIndex, 0, NodeIndex::mulOpIndex, 0
  );
  intermediateEdges.push_back(actMulToMulEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC upGemmToMulEdge = CreateIntermediateEdge(
    NodeIndex::upGemmOpIndex, 0, NodeIndex::mulOpIndex, 1
  );
  intermediateEdges.push_back(upGemmToMulEdge);

  // DOWN MATMULNBITS
  // ------------------------------------------------------------------------
  // input to gemm
  opDescs.push_back(&downDequantizeOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC mulToDownDequantizeEdge =
    CreateIntermediateEdge(
      NodeIndex::mulOpIndex, 0, NodeIndex::downGemmOpIndex, 0
    );
  intermediateEdges.push_back(mulToDownDequantizeEdge);

  // input Weights to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC downDequantizeInputEdge = CreateInputEdge(
    InputIndex::weightsDownQ4Index, NodeIndex::downDequantizeOpIndex, 0
  );
  inputEdges.push_back(downDequantizeInputEdge);

  // input Scale to Dequantize
  DML_INPUT_GRAPH_EDGE_DESC downDequantizeScaleEdge = CreateInputEdge(
    InputIndex::weightDownScaleIndex, NodeIndex::downDequantizeOpIndex, 1
  );
  inputEdges.push_back(downDequantizeScaleEdge);

  // input Zero Point to Dequantize
  if (m_ssMlp.matMulNBitsDown.hasZeroPoint) {
    DML_INPUT_GRAPH_EDGE_DESC downDequantizeZeroPointEdge = CreateInputEdge(
      InputIndex::weightDownZeroPointIndex, NodeIndex::downDequantizeOpIndex, 2
    );
    inputEdges.push_back(downDequantizeZeroPointEdge);
  }

  // TO DO --------------
  // Check for C

  // input to gemm from dequantize
  opDescs.push_back(&downGemmOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC downDequantizeToGemmEdge =
    CreateIntermediateEdge(
      NodeIndex::downDequantizeOpIndex, 0, NodeIndex::downGemmOpIndex, 1
    );
  intermediateEdges.push_back(downDequantizeToGemmEdge);

  // BOTTOM SSLRN
  // ------------------------------------------------------------------------
  opDescs.push_back(&skipAddOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC downGemmToSkipAddEdge =
    CreateIntermediateEdge(
      NodeIndex::downGemmOpIndex, 0, NodeIndex::skipAddOpIndex, 0
    );
  intermediateEdges.push_back(downGemmToSkipAddEdge);

  DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToSkipAddEdge = CreateIntermediateEdge(
    NodeIndex::paSkipAddOpIndex, 0, NodeIndex::skipAddOpIndex, 1
  );
  intermediateEdges.push_back(mvnToSkipAddEdge);

  opDescs.push_back(&mvnOpDesc);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC skipAddToMvnEdge = CreateIntermediateEdge(
    NodeIndex::skipAddOpIndex, 0, NodeIndex::mvnOpIndex, 0
  );
  intermediateEdges.push_back(skipAddToMvnEdge);

  DML_INPUT_GRAPH_EDGE_DESC skipAddInputEdge =
    CreateInputEdge(InputIndex::bottomGammaIndex, NodeIndex::mvnOpIndex, 1);
  inputEdges.push_back(skipAddInputEdge);

  // OUTPUT
  DML_OUTPUT_GRAPH_EDGE_DESC sslrnToOutputEdge = {};
  sslrnToOutputEdge.FromNodeIndex = NodeIndex::mvnOpIndex;
  sslrnToOutputEdge.FromNodeOutputIndex = 0;
  sslrnToOutputEdge.GraphOutputIndex = 0;
  outputEdges.push_back(sslrnToOutputEdge);

  if (m_ssMlp.sslrnBottom.outputCount > 1) {
    DML_OUTPUT_GRAPH_EDGE_DESC inputSkipBiasSumEdge = {};
    inputSkipBiasSumEdge.FromNodeIndex = NodeIndex::skipAddOpIndex;
    inputSkipBiasSumEdge.FromNodeOutputIndex = 0;
    inputSkipBiasSumEdge.GraphOutputIndex = 1;
    outputEdges.push_back(inputSkipBiasSumEdge);
  }

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
    status = pContext.DmlDevice()->CreateOperator(
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
  status = pContext.DmlDevice()->QueryInterface(IID_PPV_ARGS(&dmlDevice1));
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

// =====================================================================================================================
// Code shared between our constructors.
void SSMLPOperator::SharedInit() {
#ifdef DEBUG
  if (!(m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16 ||
        m_dataType == DML_TENSOR_DATA_TYPE_FLOAT32)) {
    std::wstringstream ss;
    ss << "MLP only supports Float16, not "
       << m_ssMlp.matMulNBitsGate.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  if ((m_ssMlp.matMulNBitsGate.quantizedB != L"Uint4") &&
      (m_ssMlp.matMulNBitsGate.quantizedB != L"Int4")) {
    std::wstringstream ss;
    ss << "MLP must have int4/uint4 quantized b matrix";
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  if (m_ssMlp.matMulNBitsGate.transB == false) {
    std::wstringstream ss;
    ss << "MLP only supports transposed B matrix";
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  if (m_ssMlp.matMulNBitsGate.quantizedA != L"None") {
    std::wstringstream ss;
    ss << "MLP does not support quantized A matrix";
    throw std::runtime_error(winrt::to_string(ss.str()));
  }
#endif

  CreateTensorDescSSLRN(m_ssMlp.sslrnTop);

  // For MatMulNBitsm Gate
  CreateTensorDescMatMul(m_ssMlp.matMulNBitsGate);
  CreateTensorDescMatMul(m_ssMlp.matMulNBitsUp);
  CreateTensorDescMatMul(m_ssMlp.matMulNBitsDown);
  // bottom sslrn only has weight input
  CreateTensorDescSSLRN(m_ssMlp.sslrnBottom, true, true);
}

// =====================================================================================================================
void SSMLPOperator::CreateTensorDescSSLRN(
  const SkipSimplifiedLayerNormParams& sslrnParam, bool onlyScaleInput,
  bool createOutput
) {
  assert(
    sslrnParam.inputShape.size() == 2 || sslrnParam.inputShape.size() == 3
  );

  const uint32_t batchSize =
    sslrnParam.inputShape[0] == -1 ? 1 : sslrnParam.inputShape[0];
  const uint32_t sequenceLength =
    sslrnParam.inputShape.size() == 3
      ? (sslrnParam.inputShape[1] == -1 ? 1 : sslrnParam.inputShape[1])
      : 1;
  const uint32_t hiddenSize = sslrnParam.inputShape.back();

  const std::vector<int64> inputTensorShape = {
    batchSize, sequenceLength, hiddenSize
  };  // 1
  const std::vector<int64> vectorShape = {1, 1, hiddenSize};  // 1
  const std::vector<int64> scalarShape = {1, 1, 1};           // 1
  const std::vector<int64> strides{-1, -1, -1};               // -1
  const std::vector<int64> biasStrides = {0, 0, 1};           // 0

  if (false == onlyScaleInput) {
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"SSLRNTopInput", L"DHW", m_dataType, inputTensorShape, strides
    ));
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"SSLRNTopSkip", L"DHW", m_dataType, inputTensorShape, strides
    ));

    if (sslrnParam.hasScale)
      m_inputTensorDescVec.emplace_back(CreateTensorDesc(
        L"SSLRNTopMvnScale", L"DHW", m_dataType, vectorShape, strides
      ));

    if (sslrnParam.hasBias)
      assert(
        "SSMLPOperator::CreateTensorDescSSLRN: sslrnParam.hasBias is true"
      );

    if (sslrnParam.hasNonMVNBias)
      assert(
        "SSMLPOperator::CreateTensorDescSSLRN: sslrnParam.hasNonMVNBias is "
        "true"
      );
  } else {
    if (sslrnParam.hasScale)
      m_inputTensorDescVec.emplace_back(CreateTensorDesc(
        L"SSLRNTopMvnScale", L"DHW", m_dataType, vectorShape, strides
      ));
  }

  if (createOutput) {
    // Final output of SSMLP is from SSLRNBottom node
    m_outputTensorDescVec.emplace_back(CreateTensorDesc(
      L"SSLRNBottomOutput", L"DHW", m_dataType, inputTensorShape, strides
    ));

    m_tmpTensorDescSkipAdd = CreateTensorDesc(
      L"SSLRNBottomSkipBiasSumOutput", L"DHW", m_dataType, inputTensorShape,
      strides
    );
    if (sslrnParam.outputCount > 1) {
      m_outputTensorDescVec.emplace_back(m_tmpTensorDescSkipAdd);
    }
  }
}

void SSMLPOperator::CreateTensorDescMatMul(const MatMulNBitsParams& matParam) {
  const int64 heightA = matParam.transA ? matParam.k : matParam.m;
  const int64 widthA = matParam.transA ? matParam.m : matParam.k;
  const int64 heightB = matParam.transB ? matParam.n : matParam.k;
  const int64 widthB = matParam.transB ? matParam.k : matParam.n;

  const std::vector<int64> sizeA{matParam.batches, heightA, widthA};
  const std::vector<int64> sizeB{matParam.batches, heightB, widthB};
  const std::vector<int64> sizeY{matParam.batches, matParam.m, matParam.n};

  const std::vector<int64> sizeQBS{
    matParam.batches, heightB, widthB / matParam.quantizedBlockB
  };
  const std::vector<int64> sizeQBZ{
    matParam.batches, heightB, widthB / matParam.quantizedBlockB
  };

  const std::vector<int64> stridesQBS{-1, -1, -1};
  const std::vector<int64> stridesQBZ{-1, -1, -1};

  const std::vector<int64> stridesA{
    matParam.stridesA[0], matParam.stridesA[1], matParam.stridesA[2]
  };
  const std::vector<int64> stridesB{
    matParam.stridesB[0], matParam.stridesB[1], matParam.stridesB[2]
  };
  const std::vector<int64> stridesY{
    matParam.stridesY[0], matParam.stridesY[1], matParam.stridesY[2]
  };

  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"BTensorData", matParam.orderB, m_quantDataType, sizeB, stridesB
  ));
  m_inputTensorDescVec.emplace_back(CreateTensorDesc(
    L"BTensorScale", matParam.orderB, m_dataType, sizeQBS, stridesQBS
  ));
  if (matParam.hasZeroPoint) {
    // if zero point is used then its type is same as BTensorData, i.e
    // uint4/int4
    m_inputTensorDescVec.emplace_back(CreateTensorDesc(
      L"BTensorZeroPoint", matParam.orderB, m_quantDataType, sizeQBZ, stridesQBZ
    ));
  }

  if (matParam.hasC) {
    // Note that C always has the same size as Y.
    assert("SSMLPOperator::CreateTensorDescMatMul: matParam.hasC = true");
  }
}

}  // namespace ryzenai::onnx_utils
