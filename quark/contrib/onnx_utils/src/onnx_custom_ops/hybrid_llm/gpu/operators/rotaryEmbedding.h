// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes the parameters for the RotaryEmbedding operation.
struct RotaryEmbeddingParams {
  int32 batchSize = 1;              // Batch size.
  int32 sequenceLength = 1;         // Sequence length.
  int32 numHeads = 0;               // Number of heads.
  int32 headSize = 0;               // Size of each head.
  std::vector<int64_t> inputShape;  // Input to Rotary
  std::vector<int64_t> cosShape;    // The shape of the cos table.
  std::vector<int64_t> sinShape;    // The shape of the sin table.
  int32 rotaryEmbeddingDim = 0;     // Rotary dimension.
  bool interleaved = false;         // Whether the data is interleaved.

  // Tensor properties.
  hstring order =
    L"DHW";  // The dimension packing order. For example, "HW" or "DWH".
  hstring dataType = L"Float16";  // Which TensorProto DataType to use for our
                                  // tensors (e.g., Float).
};

// =====================================================================================================================
// This class abstracts a DirectML RotaryEmbedding operator.
class RotaryEmbeddingOperator : public DmlOperator {
 public:
  explicit RotaryEmbeddingOperator(
    const Context& context, const RotaryEmbeddingParams& params,
    bool disableCompile = false
  );
  virtual ~RotaryEmbeddingOperator() {}

  void CreateRotaryEmbeddingOperator(
    const Context& pContext, std::vector<CComPtr<IDMLOperator>>& dmlOperators,
    std::shared_ptr<DmlTensorDesc>& dmlDesc
  );

  bool IsPartialRotaryEmbedding() const { return m_isPartialRotaryEmbedding; }

 private:
  void SharedInit();

  RotaryEmbeddingParams m_ropeParams;
  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.

  TensorDescVector m_interTensorDescVec;
  bool m_isPartialRotaryEmbedding =
    false;  // Check for partial rotary embedding.
};

}  // namespace ryzenai::onnx_utils
