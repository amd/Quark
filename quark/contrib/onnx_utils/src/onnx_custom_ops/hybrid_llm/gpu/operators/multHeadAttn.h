// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract MHA operation.
struct MhaParams {
  uint32 batch;      // Number of batches.
  uint32 qSeq;       // Size of sequence for query.
  uint32 kvSeq;      // Size of sequence for key and value.
  uint32 sizeHeads;  // size of each multi-attention head
  uint32 headCount;  // Size of multi-attention heads
  float scale;  // Scale to multiply the data after the QxK GEMM operation but
                // before Softmax.
  // This value is usually 1/sqrt(headSize).
  hstring packing;   // "Cross" or "Self"
  hstring dataType;  // Which TensorProto DataType to use for our tensors (e.g.,
                     // Float).
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class MhaOperator : public DmlOperator {
 public:
  explicit MhaOperator(
    const Context& context, bool disableMetacmds, const MhaParams& params
  );
  virtual ~MhaOperator() {}

 private:
  void ValidateMhaParams(const MhaParams& params) const;

  const DML_TENSOR_DATA_TYPE
    m_dataType;  // All tensors must use this data type.
  MhaParams m_mha;
  ActivationFuncInfo m_activation;
};

}  // namespace ryzenai::onnx_utils
