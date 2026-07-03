//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable ABI version for pre-compiled distribution.
//

#include "mx/funcs.cuh"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include "device_guard.h"

namespace quark {
namespace hw_emulation {

void fake_quantize_to_low_precision_fp_cpu(
  float *input, float *output, uint32_t num_elements, int ebits, int mbits,
  float max_norm, RoundMode round_mode
) {
  for (uint32_t i = 0; i < num_elements; i++) {
    output[i] =
      fake_quantize_element(input[i], max_norm, ebits, mbits, round_mode);
  }
}

torch::stable::Tensor fake_quantize_to_low_precision_fp_impl(
  const torch::stable::Tensor &input, int64_t ebits, int64_t mbits,
  double max_norm, int64_t round_mode
) {
  STD_TORCH_CHECK(
    round_mode >= 0,
    "fake_quantize_to_low_precision_fp: round_mode must be non-negative"
  );

  float *input_data = static_cast<float *>(input.data_ptr());
  torch::stable::Tensor output = torch::stable::empty_like(input);
  float *output_data = static_cast<float *>(output.data_ptr());

#ifdef USE_CUDA
  quark::OptionalDeviceGuard guard(input);

  if (input.is_cpu()) {
    fake_quantize_to_low_precision_fp_cpu(
      input_data, output_data, static_cast<uint32_t>(input.numel()),
      static_cast<int>(ebits), static_cast<int>(mbits),
      static_cast<float>(max_norm), static_cast<RoundMode>(round_mode)
    );
  } else {
    fake_quantize_to_low_precision_fp_cuda(
      input_data, output_data, static_cast<uint32_t>(input.numel()),
      static_cast<int>(ebits), static_cast<int>(mbits),
      static_cast<float>(max_norm), static_cast<RoundMode>(round_mode)
    );
  }
#else
  fake_quantize_to_low_precision_fp_cpu(
    input_data, output_data, static_cast<uint32_t>(input.numel()),
    static_cast<int>(ebits), static_cast<int>(mbits),
    static_cast<float>(max_norm), static_cast<RoundMode>(round_mode)
  );
#endif
  return output;
}

}  // namespace hw_emulation
}  // namespace quark
