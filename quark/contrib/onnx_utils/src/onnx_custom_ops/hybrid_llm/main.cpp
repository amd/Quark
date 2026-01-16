// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "main.hpp"

#include "operators/cast.hpp"
#include "operators/conv.hpp"
#include "operators/conv_split_mul.hpp"
#include "operators/gather.hpp"
#include "operators/gqo.hpp"
#include "operators/lut.hpp"
#include "operators/matmul.hpp"
#include "operators/matmulnbits.hpp"
#include "operators/qmoe.hpp"
#include "operators/reducesum.hpp"
#include "operators/rope.hpp"
#include "operators/slrn.hpp"
#include "operators/ssgmlp.hpp"
#include "operators/sslrn.hpp"
#include "operators/ssmlp.hpp"
#include "operators/sub.hpp"

namespace ryzenai::onnx_utils {

std::vector<const OrtCustomOp*> create_hybrid_llm_ops(
  const Ort::ConstSessionOptions& options
) {
  std::vector<const OrtCustomOp*> hybrid_llm_ops;
  hybrid_llm_ops.emplace_back(GetCustomOp<Cast>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<Conv>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<ConvSplitMul>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<Gather>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<GQO>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<Lut>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<MatMul>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<MatMulNBits>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<MatMulNBitsBf>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<QmoeBf>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<RotaryEmbedding>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<SimplifiedLayerNorm>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<SkipSimplifiedLayerNorm>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<SSGMlp>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<SSMlp>(options));
  hybrid_llm_ops.emplace_back(GetCustomOp<Sub>(options));

  return hybrid_llm_ops;
}

}  // namespace ryzenai::onnx_utils
