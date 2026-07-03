//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable ABI version for pre-compiled distribution.
//

#include <math.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cmath>
#include <stdexcept>
#include <tuple>

#include "device_guard.h"
#include "stable_ops.h"

#ifdef USE_CUDA
void tqt_backward_kernel(
  const int N, float* x, float* scale, float* quant_min, float* quant_max,
  float* grad_logt, float* grad_output
);
#endif

namespace quark {
namespace hw_emulation {

namespace ops = quark::torch_stable;

namespace {

// CPU path expressed in stable-ABI ops; used for CPU inputs under any build.
std::tuple<torch::stable::Tensor, torch::stable::Tensor> tqt_backward_cpu(
  const torch::stable::Tensor& x, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& quant_max,
  const torch::stable::Tensor& quant_min, const torch::stable::Tensor& logt,
  const torch::stable::Tensor& grad_output
) {
  auto scaled_x = ops::div(x, scale);
  auto floor_scaled = ops::floor(scaled_x);
  auto diff = ops::sub(scaled_x, floor_scaled);

  auto half = ops::full_like(diff, 0.5);
  auto zero = ops::zeros_like(scaled_x);

  auto neg_mask = ops::lt(scaled_x, zero);
  auto eq_half = ops::eq(diff, half);
  auto use_ceil = ops::logical_and(neg_mask, eq_half);

  auto ceil_scaled = ops::ceil(scaled_x);
  auto round_scaled = ops::round(scaled_x);
  auto rounded_scaled_x = ops::where(use_ceil, ceil_scaled, round_scaled);

  auto is_lt_min = ops::lt(rounded_scaled_x, quant_min);
  auto is_gt_max = ops::gt(rounded_scaled_x, quant_max);
  auto is_ge_min_and_le_max =
    ops::logical_and(ops::logical_not(is_lt_min), ops::logical_not(is_gt_max));

  auto log2_val = ops::full_like(scale, log(2.0));
  auto grad_logt_result = ops::mul(ops::mul(grad_output, scale), log2_val);

  auto diff_rounded = ops::sub(rounded_scaled_x, scaled_x);
  auto grad_logt_in_range = ops::mul(grad_logt_result, diff_rounded);
  grad_logt_result =
    ops::where(is_ge_min_and_le_max, grad_logt_in_range, grad_logt_result);

  auto grad_logt_lt_min = ops::mul(grad_logt_result, quant_min);
  grad_logt_result = ops::where(is_lt_min, grad_logt_lt_min, grad_logt_result);

  auto grad_logt_gt_max = ops::mul(grad_logt_result, quant_max);
  grad_logt_result = ops::where(is_gt_max, grad_logt_gt_max, grad_logt_result);

  auto sum_grad_logt = ::torch::stable::sum(grad_logt_result);
  grad_logt_result = ops::expand_as(sum_grad_logt, logt);

  auto grad_x = ::torch::stable::clone(grad_output);
  auto zero_grad = ops::zeros_like(grad_x);
  grad_x = ops::where(is_ge_min_and_le_max, grad_x, zero_grad);

  return std::make_tuple(grad_x, grad_logt_result);
}

}  // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor> tqt_backward_impl(
  const torch::stable::Tensor& x, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& quant_max,
  const torch::stable::Tensor& quant_min, const torch::stable::Tensor& logt,
  torch::stable::Tensor& grad_output
) {
#ifdef USE_CUDA
  if (x.is_cpu()) {
    return tqt_backward_cpu(x, scale, quant_max, quant_min, logt, grad_output);
  }
  quark::OptionalDeviceGuard device_guard(x);

  auto log2_val = ops::full_like(scale, log(2.0));
  auto grad_logt_result = ops::mul(ops::mul(grad_output, scale), log2_val);

  auto quant_max_float =
    ::torch::stable::to(quant_max, torch::headeronly::ScalarType::Float);
  auto quant_min_float =
    ::torch::stable::to(quant_min, torch::headeronly::ScalarType::Float);

  tqt_backward_kernel(
    static_cast<int>(x.numel()),
    static_cast<float*>(const_cast<void*>(x.data_ptr())),
    static_cast<float*>(const_cast<void*>(scale.data_ptr())),
    static_cast<float*>(quant_min_float.data_ptr()),
    static_cast<float*>(quant_max_float.data_ptr()),
    static_cast<float*>(grad_logt_result.data_ptr()),
    static_cast<float*>(grad_output.data_ptr())
  );

  auto sum_grad_logt = ::torch::stable::sum(grad_logt_result);
  grad_logt_result = ops::expand_as(sum_grad_logt, logt);

  return std::make_tuple(grad_output, grad_logt_result);
#else
  return tqt_backward_cpu(x, scale, quant_max, quant_min, logt, grad_output);
#endif
}

}  // namespace hw_emulation
}  // namespace quark
