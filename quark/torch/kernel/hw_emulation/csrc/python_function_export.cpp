//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable-ABI op registrations (PyTorch >= 2.10), allowing a single pre-built
// binary to load across multiple PyTorch minor versions.
//

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace quark {
namespace hw_emulation {

#ifdef USE_CUDA
// Defined in fake_tensor_cuda_hip.cu
torch::stable::Tensor fake_quantize_per_tensor_affine_impl(
  const torch::stable::Tensor& input, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& zero_point, int64_t quant_min, int64_t quant_max,
  int64_t round_mode
);
#else
torch::stable::Tensor fake_quantize_per_tensor_affine_impl(
  const torch::stable::Tensor& input, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& zero_point, int64_t quant_min, int64_t quant_max,
  int64_t round_mode
) {
  throw std::runtime_error(
    "fake_quantize_per_tensor_affine is not implemented on non CUDA-devices"
  );
}
#endif

// Defined in mx/cpu/funcs.cpp
torch::stable::Tensor fake_quantize_to_low_precision_fp_impl(
  const torch::stable::Tensor& input, int64_t ebits, int64_t mbits,
  double max_norm, int64_t round_mode
);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> tqt_backward_impl(
  const torch::stable::Tensor& x, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& quant_max,
  const torch::stable::Tensor& quant_min, const torch::stable::Tensor& logt,
  torch::stable::Tensor& grad_output
);

#ifdef USE_CUDA
// Defined in mxfp4/dequantize.cu
void dq_uint8_mxfp4_to_half_impl(
  const torch::stable::Tensor& inp, const torch::stable::Tensor& scales,
  torch::stable::Tensor& out, int64_t group_size
);

// Defined in mxfp4/fake.cu
torch::stable::Tensor qdq_mxfp4_impl(
  const torch::stable::Tensor& a, int64_t group_size
);

void qdq_mxfp4_inplace_impl(torch::stable::Tensor& a, int64_t group_size);
#else
void dq_uint8_mxfp4_to_half_impl(
  const torch::stable::Tensor& inp, const torch::stable::Tensor& scales,
  torch::stable::Tensor& out, int64_t group_size
) {
  throw std::runtime_error(
    "dq_uint8_mxfp4_to_half is only implemented in CUDA devices!"
  );
}

torch::stable::Tensor qdq_mxfp4_impl(
  const torch::stable::Tensor& a, int64_t group_size
) {
  throw std::runtime_error("qdq_mxfp4 is only implemented in CUDA devices!");
}

void qdq_mxfp4_inplace_impl(torch::stable::Tensor& a, int64_t group_size) {
  throw std::runtime_error("qdq_mxfp4_ is only implemented in CUDA devices!");
}
#endif

torch::stable::Tensor fake_quantize_per_tensor_affine(
  const torch::stable::Tensor& input, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& zero_point, int64_t quant_min, int64_t quant_max,
  int64_t round_mode
) {
  return fake_quantize_per_tensor_affine_impl(
    input, scale, zero_point, quant_min, quant_max, round_mode
  );
}

torch::stable::Tensor fake_quantize_to_low_precision_fp(
  const torch::stable::Tensor& input, int64_t ebits, int64_t mbits,
  double max_norm, int64_t round_mode
) {
  return fake_quantize_to_low_precision_fp_impl(
    input, ebits, mbits, max_norm, round_mode
  );
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> tqt_backward(
  const torch::stable::Tensor& x, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& quant_max,
  const torch::stable::Tensor& quant_min, const torch::stable::Tensor& logt,
  torch::stable::Tensor& grad_output
) {
  return tqt_backward_impl(x, scale, quant_max, quant_min, logt, grad_output);
}

void dq_uint8_mxfp4_to_half(
  const torch::stable::Tensor& inp, const torch::stable::Tensor& scales,
  torch::stable::Tensor& out, int64_t group_size
) {
  dq_uint8_mxfp4_to_half_impl(inp, scales, out, group_size);
}

torch::stable::Tensor qdq_mxfp4(
  const torch::stable::Tensor& a, int64_t group_size
) {
  return qdq_mxfp4_impl(a, group_size);
}

void qdq_mxfp4_(torch::stable::Tensor& a, int64_t group_size) {
  qdq_mxfp4_inplace_impl(a, group_size);
}

}  // namespace hw_emulation
}  // namespace quark

// Wrap m.def / m.impl so a single bad schema/op surfaces in CI output rather
// than aborting the whole library registration silently.
#define SAFE_DEF(m, schema)                                                 \
  try {                                                                     \
    m.def(schema);                                                          \
  } catch (const std::exception& e) {                                       \
    std::cerr << "[quark_hw_emulation] m.def FAILED for schema: " << schema \
              << "\n  exception: " << e.what() << std::endl;                \
  } catch (...) {                                                           \
    std::cerr << "[quark_hw_emulation] m.def FAILED for schema: " << schema \
              << "\n  exception: <unknown>" << std::endl;                   \
  }

#define SAFE_IMPL(m, name, fn)                                         \
  try {                                                                \
    m.impl(name, fn);                                                  \
  } catch (const std::exception& e) {                                  \
    std::cerr << "[quark_hw_emulation] m.impl FAILED for op: " << name \
              << "\n  exception: " << e.what() << std::endl;           \
  } catch (...) {                                                      \
    std::cerr << "[quark_hw_emulation] m.impl FAILED for op: " << name \
              << "\n  exception: <unknown>" << std::endl;              \
  }

STABLE_TORCH_LIBRARY(quark_hw_emulation, m) {
  SAFE_DEF(
    m,
    "fake_quantize_per_tensor_affine(Tensor inputs, Tensor scale, Tensor "
    "zero_point, int quant_min, int quant_max, int round_mode) -> Tensor"
  );
  SAFE_DEF(
    m,
    "fake_quantize_to_low_precision_fp(Tensor input, int ebits, int mbits, "
    "float max_norm, int round_mode) -> Tensor"
  );
  SAFE_DEF(
    m,
    "tqt_backward(Tensor x, Tensor scale, Tensor quant_max, Tensor quant_min, "
    "Tensor logt, Tensor(a!) grad_output) -> (Tensor(a!), Tensor)"
  );
  SAFE_DEF(
    m,
    "dq_uint8_mxfp4_to_half(Tensor inp, Tensor scales, Tensor(a!) out, int "
    "group_size) -> ()"
  );
  SAFE_DEF(m, "qdq_mxfp4(Tensor a, int group_size) -> Tensor");
  SAFE_DEF(m, "qdq_mxfp4_(Tensor(a!) a, int group_size) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(quark_hw_emulation, CompositeExplicitAutograd, m) {
  SAFE_IMPL(
    m, "fake_quantize_per_tensor_affine",
    TORCH_BOX(&quark::hw_emulation::fake_quantize_per_tensor_affine)
  );
  SAFE_IMPL(
    m, "fake_quantize_to_low_precision_fp",
    TORCH_BOX(&quark::hw_emulation::fake_quantize_to_low_precision_fp)
  );
  SAFE_IMPL(m, "tqt_backward", TORCH_BOX(&quark::hw_emulation::tqt_backward));
  SAFE_IMPL(
    m, "dq_uint8_mxfp4_to_half",
    TORCH_BOX(&quark::hw_emulation::dq_uint8_mxfp4_to_half)
  );
  SAFE_IMPL(m, "qdq_mxfp4", TORCH_BOX(&quark::hw_emulation::qdq_mxfp4));
  SAFE_IMPL(m, "qdq_mxfp4_", TORCH_BOX(&quark::hw_emulation::qdq_mxfp4_));
}
