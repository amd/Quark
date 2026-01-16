// Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"
#include "operators/matMulNBits.h"
#include "operators/rotaryEmbedding.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
enum GQAInputs : uint32_t {
  // qkvInput,
  ropeQuery,
  ropeKey,
  interMatmulValue,
  pastSequenceLength,
  inputCount,
};

enum GQAOutputs : uint32_t {
  output,
  outputPresentKey,
  outputPresentValue,
  outputCount,
};

// =====================================================================================================================
enum GQAOutputIndex : uint32_t {
  gqaoutputIndex,
  gqaskipBiasSumIndex,
};

struct GQAParams {
  int32 batchSize = 1;       // Batch size.
  int32 sequenceLength = 1;  // Sequence length.
  std::vector<int64_t> inputValueShape;
  std::vector<int64_t> inputKeyShape;
  std::vector<int64_t> inputQueryShape;

  std::vector<int64_t> inputPastKeyShape;
  std::vector<int64_t> inputPastValueShape;
  std::vector<int64_t> inputSubCastShape;  // sequence_k

  std::vector<int64_t> outputKeyShape;
  std::vector<int64_t> outputValueShape;
  std::vector<int64_t> outputQueryShape;
  int32 inputGatherCastShape;  // total sequence length
  int64 do_rotary = 1;
  int64 kv_num_heads = 1;
  int64 num_heads = 1;
  int64 local_window_size = -1;
  float scale = 1.0f;
  float softcap = 0.0f;
  hstring dataType = L"Float16";
};

// =====================================================================================================================
// This class implements GQO operator.
class GQAOperator : public DmlOperator {
 public:
  explicit GQAOperator(
    const Context& context, const GQAParams& params, bool disableCompile = false
  );
  virtual ~GQAOperator() {}

  void CreateGQAOperator(
    const Context& pContext, std::vector<CComPtr<IDMLOperator>>& dmlOperators,
    std::shared_ptr<DmlTensorDesc>& dmlDescIn1,
    std::shared_ptr<DmlTensorDesc>& dmlDescIn2,
    std::shared_ptr<DmlTensorDesc>& dmlDescOut
  );

 private:
  void SharedInit();

  DML_INPUT_GRAPH_EDGE_DESC CreateInputEdge(uint32, uint32, uint32);
  DML_INTERMEDIATE_GRAPH_EDGE_DESC CreateIntermediateEdge(
    uint32, uint32, uint32, uint32
  );
  void CreateDmlCastTensorDesc(
    DML_TENSOR_DATA_TYPE dataType, const DmlTensorDesc& dmlTensorDesc,
    DmlTensorDesc& castDmlTensorDesc
  );
  GQAParams m_gqaParams;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.

  // inputs from rotary embedding
  DmlTensorDesc m_RotEmbQueryInputDesc;
  DmlTensorDesc m_RotEmbKeyInputDesc;
};

}  // namespace ryzenai::onnx_utils
