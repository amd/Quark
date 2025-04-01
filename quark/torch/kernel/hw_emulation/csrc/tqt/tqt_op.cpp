//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <math.h>
#include <cmath>
#include <torch/extension.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

void tqt_backward_kernel(const int N, float *x, float *scale, float *quant_min, float *quant_max, float *grad_logt, float *grad_output);

std::vector<at::Tensor> tqt_backward(
    at::Tensor& x, at::Tensor& scale, at::Tensor& quant_max,
    at::Tensor& quant_min, at::Tensor& logt, at::Tensor& grad_output) {
#ifdef USE_CUDA
  if (x.device().is_cpu()) {
    auto scaled_x = x / scale;
    auto rounded_scaled_x = torch::where(
          (scaled_x < 0) & (scaled_x - torch::floor(scaled_x) == 0.5),
          torch::ceil(scaled_x), torch::round(scaled_x));
    auto is_lt_min = rounded_scaled_x < quant_min;
    auto is_gt_max = rounded_scaled_x > quant_max;
    auto is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max;
    auto grad_logt = grad_output * scale * log(2);
    grad_logt = torch::where(is_ge_min_and_le_max,
                              grad_logt * (rounded_scaled_x - scaled_x),
                              grad_logt);
    grad_logt = torch::where(is_lt_min, grad_logt * quant_min, grad_logt);
    grad_logt = torch::where(is_gt_max, grad_logt * quant_max, grad_logt);
    grad_logt = grad_logt.sum().expand_as(logt);
    auto grad_x = grad_output.clone();
    grad_x = torch::where(is_ge_min_and_le_max, grad_x, 0 * grad_x);
    return {grad_x, grad_logt};
  } else {
    auto grad_logt = grad_output * scale * log(2);
    quant_max = quant_max.toType(at::kFloat);
    quant_min = quant_min.toType(at::kFloat);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    tqt_backward_kernel(
      x.numel(), x.data_ptr<float>(), scale.data_ptr<float>(),
      quant_min.data_ptr<float>(), quant_max.data_ptr<float>(),
      grad_logt.data_ptr<float>(), grad_output.data_ptr<float>());

    grad_logt = grad_logt.sum().expand_as(logt);
    return {grad_output, grad_logt};
  }
#else
  auto scaled_x = x / scale;
  auto rounded_scaled_x = torch::where(
        (scaled_x < 0) & (scaled_x - torch::floor(scaled_x) == 0.5),
        torch::ceil(scaled_x), torch::round(scaled_x));
  auto is_lt_min = rounded_scaled_x < quant_min;
  auto is_gt_max = rounded_scaled_x > quant_max;
  auto is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max;
  auto grad_logt = grad_output * scale * log(2);
  grad_logt = torch::where(is_ge_min_and_le_max,
                            grad_logt * (rounded_scaled_x - scaled_x),
                            grad_logt);
  grad_logt = torch::where(is_lt_min, grad_logt * quant_min, grad_logt);
  grad_logt = torch::where(is_gt_max, grad_logt * quant_max, grad_logt);
  grad_logt = grad_logt.sum().expand_as(logt);
  auto grad_x = grad_output.clone();
  grad_x = torch::where(is_ge_min_and_le_max, grad_x, 0 * grad_x);
  return {grad_x, grad_logt};
#endif
}
