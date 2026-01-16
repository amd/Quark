// Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"
#include "operators/gqa.h"
#include "operators/matMulNBits.h"
#include "operators/rotaryEmbedding.h"

// DEBUGGQO
#include "opInterface.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
struct GQOParams {
  RotaryEmbeddingParams rotEmbQuery;
  RotaryEmbeddingParams rotEmbKey;
  MatMulNBitsParams matMulNBits;
  GQAParams gqa;
  std::vector<int64_t> qkvPackedInputShape;
};

// =====================================================================================================================
// This class implements GQO operator.
class GQOOperator : public DmlOperator {
 public:
  explicit GQOOperator(
    const Context& context, const GQOParams& params, bool disableCompile = false
  );
  virtual ~GQOOperator() {}

  std::vector<int64_t> GetQKVOffsetsVector() { return m_qkvOffsetsVector; }
  void DynamicInitialization(
    const Context& pContext, void* params, bool rebuildOp = false
  );
  bool IsFirstRun() { return m_firstRun; }

  const int32 GQOOperator::GetLocalWindowSize() const {
    return (int32)m_gqoParams.gqa.local_window_size;
  }

 private:
  bool m_firstRun = true;
  GQOParams m_gqoParams;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.

  // Operators
  std::shared_ptr<RotaryEmbeddingOperator> m_rotEmbQueryOp = nullptr;
  std::shared_ptr<RotaryEmbeddingOperator> m_rotEmbKeyOp = nullptr;
  std::shared_ptr<GQAOperator> m_groupQueryOp = nullptr;
  std::shared_ptr<MatMulNBitsOperator> m_internalMatmul = nullptr;

  // Construct the graph
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> m_inputEdges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> m_intermediateEdges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> m_outputEdges;

  std::vector<int64_t> m_qkvOffsetsVector;

  bool m_isGraphConstructed = false;

  void UpdateNodeMappings();
  void CreateInputEdge(std::string name, uint32, uint32, uint32);
  void CreateIntermediateEdge(std::string name, uint32, uint32, uint32, uint32);
};

}  // namespace ryzenai::onnx_utils
