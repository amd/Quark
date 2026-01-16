// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "main.hpp"

#include "cast_avx.hpp"
#include "cast_mul_avx.hpp"
#include "dynamic_dispatch.hpp"
#include "dynamic_pad.hpp"
#include "pad_input_ids.hpp"

namespace ryzenai::onnx_utils {

std::vector<const OrtCustomOp*> create_dynamic_dispatch_ops(
  const Ort::ConstSessionOptions& options
) {
  // Custom operators are static to ensure they remain valid until the library
  // is unloaded.

  const auto* cast_avx = GetCustomOp<CastAvx>(options);

  // Fused Cast+Mul AVX operation
  const auto* cast_mul_avx = GetCustomOp<CastMulAvx>(options);

  // this is only for compatibility with legacy LLM prefill models. If we drop
  // support for those, we should remove this
  static const char pad_name[] = "Pad";
  const auto* pad =
    GetCustomOp<PadInputIds<pad_name, PadInputIdsKernel>>(options);

  // custom pad op for SD
  static const char dynamic_pad_name[] = "DynamicPad";

  const auto* dynamic_pad =
    GetCustomOp<DynamicPadOp<dynamic_pad_name, DynamicPadKernel>>(options);

  static const char dd_name[] = "DynamicDispatch";
  const auto* dd =
    GetCustomOp<DynamicDispatch<dd_name, DynamicDispatchKernel>>(options);

  return {cast_avx, cast_mul_avx, dd, pad, dynamic_pad};
}

}  // namespace ryzenai::onnx_utils
