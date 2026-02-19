// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"
#include "operators/SkipSimplifiedLayerNorm.h"
#include "operators/matMulNBits.h"
#include "operators/norm.h"
#include "ssmlp.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
enum SSMLPGeluInputIndex : uint32_t {
  sg_skipIndex,
  sg_xInputIndex,
  sg_topScaleIndex,
  sg_topGammaIndex,
  sg_weightsGateQ4Index,
  sg_weightGateScaleIndex,
  sg_weightGateZeroPointIndex,
  sg_weightsUpQ4Index,
  sg_weightUpScaleIndex,
  sg_weightUpZeroPointIndex,
  sg_weightsDownQ4Index,
  sg_weightDownScaleIndex,
  sg_weightDownZeroPointIndex,
  sg_bottomScaleIndex,
  sg_bottomGammaIndex,
  sg_count,
};

enum SSMLPGeluNodeIndex : uint32_t {
  sg_paNormTopIndex,
  sg_paSkipAddOpIndex,
  sg_paMvnOpIndex,
  sg_gateDequantizeOpIndex,
  sg_gateGemmOpIndex,
  sg_geluOpIndex,
  sg_upDequantizeOpIndex,
  sg_upGemmOpIndex,
  sg_mulOpIndex,
  sg_downDequantizeOpIndex,
  sg_downGemmOpIndex,
  sg_normBottomIndex,
  sg_skipAddOpIndex,
  sg_mvnOpIndex,
  sg_NodeCount
};

// =====================================================================================================================
struct SSMLPGeluParams : public SSMLPParms {
  NormParams normTop;
  NormParams normBottom;
};

// =====================================================================================================================
// This class implements SSMLP operator.
class SSMLPGeluOperator : public DmlOperator {
 public:
  explicit SSMLPGeluOperator(
    const Context& context, bool disableMetacmds, const SSMLPGeluParams& params
  );
  virtual ~SSMLPGeluOperator() {}

 private:
  void SharedInit();
  void CreateTensorDescMatMul(const MatMulNBitsParams& matParam);
  void CreateTensorDescNorm(const NormParams& sslrnParam, bool xInput = false);
  void CreateTensorDescSSLRN(
    const SkipSimplifiedLayerNormParams& sslrnParam, bool skipInput = false,
    bool scaleInput = false, bool createOutput = false
  );

  void CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE dataType, const DmlTensorDesc& dmlTensorDesc,
    DmlTensorDesc* castDmlTensorDesc
  );
  std::shared_ptr<TensorDesc> m_layerNormIOTensorDesc = nullptr;

  void DequantizeBForMatmul(
    DmlTensorDesc& dmlTensorQBDequant, const TensorDesc* inputTensorDesc,
    const DML_TENSOR_DATA_TYPE& dequantDataType
  );

  DML_INPUT_GRAPH_EDGE_DESC CreateInputEdge(uint32, uint32, uint32);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC CreateIntermediateEdge(
    uint32, uint32, uint32, uint32
  );

  const SSMLPGeluParams m_ssMlp;
  DML_TENSOR_DATA_TYPE
  m_internalDataType;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  const DML_TENSOR_DATA_TYPE m_quantDataType;  // Quantized datatype

  // Workaround for last ssmlp node
  std::shared_ptr<TensorDesc> m_tmpTensorDescSkipAdd;
};

}  // namespace ryzenai::onnx_utils
