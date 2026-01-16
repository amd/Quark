// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"
#include "operators/SkipSimplifiedLayerNorm.h"
#include "operators/matMulNBits.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
enum InputIndex : uint32_t {
  inputIndex,
  skipIndex,
  topGammaIndex,
  weightsGateQ4Index,
  weightGateScaleIndex,
  weightGateZeroPointIndex,
  weightsUpQ4Index,
  weightUpScaleIndex,
  weightUpZeroPointIndex,
  weightsDownQ4Index,
  weightDownScaleIndex,
  weightDownZeroPointIndex,
  bottomGammaIndex,
};

enum NodeIndex : uint32_t {
  paSkipAddOpIndex,
  paMvnOpIndex,
  gateDequantizeOpIndex,
  gateGemmOpIndex,
  sigmoidOpIndex,
  actMulOpIndex,
  upDequantizeOpIndex,
  upGemmOpIndex,
  mulOpIndex,
  downDequantizeOpIndex,
  downGemmOpIndex,
  skipAddOpIndex,
  mvnOpIndex,
};

// =====================================================================================================================
enum OutputIndex : uint32_t {
  outputIndex,
  skipBiasSumIndex,
};

// =====================================================================================================================
struct SSMLPParms {
  SkipSimplifiedLayerNormParams sslrnTop;
  MatMulNBitsParams matMulNBitsGate;
  MatMulNBitsParams matMulNBitsUp;
  MatMulNBitsParams matMulNBitsDown;
  SkipSimplifiedLayerNormParams sslrnBottom;
};

// =====================================================================================================================
// This class implements SSMLP operator.
class SSMLPOperator : public DmlOperator {
 public:
  explicit SSMLPOperator(
    const Context& context, bool disableMetacmds, const SSMLPParms& params
  );
  virtual ~SSMLPOperator() {}

 private:
  void SharedInit();
  void CreateTensorDescMatMul(const MatMulNBitsParams& matParam);
  void CreateTensorDescSSLRN(
    const SkipSimplifiedLayerNormParams& sslrnParam,
    bool onlyScaleInput = false, bool createOutput = false
  );
  void DequantizeBForMatmul(
    DmlTensorDesc& dmlTensorQBDequant, const TensorDesc* inputTensorDesc,
    const DML_TENSOR_DATA_TYPE& dequantDataType
  );

  DML_INPUT_GRAPH_EDGE_DESC CreateInputEdge(uint32, uint32, uint32);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC CreateIntermediateEdge(
    uint32, uint32, uint32, uint32
  );

  const SSMLPParms m_ssMlp;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  const DML_TENSOR_DATA_TYPE m_quantDataType;  // Quantized datatype

  // Workaround for last ssmlp node
  std::shared_ptr<TensorDesc> m_tmpTensorDescSkipAdd;
};

}  // namespace ryzenai::onnx_utils
