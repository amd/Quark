//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "cu_utils.h"

__global__ static void _tqt_backward_kernel(
    const int N, float *x, float *scale,
    float *quant_min, float *quant_max,
    float *grad_logt, float *grad_output){

  QUARK_KERNEL_LOOP(idx, N){
    float scaled_x = x[idx] / *scale;
    float rounded_scaled_x = 0.0;
    if (scaled_x - floorf(scaled_x) == 0.5) {
      rounded_scaled_x = ceilf(scaled_x);
    } else {
      rounded_scaled_x = roundf(scaled_x);
    }

    if (grad_logt[idx] < *quant_min) {
      grad_logt[idx] *= *quant_min;
      grad_output[idx] = 0;
    } else if (grad_logt[idx] > *quant_max) {
      grad_logt[idx] *= *quant_min;
      grad_output[idx] = 0;
    } else {
      grad_logt[idx] *= (rounded_scaled_x - scaled_x);
    }
  }
}

void tqt_backward_kernel(
    const int N, float *x, float *scale,
    float *quant_min, float *quant_max,
    float *grad_logt, float *grad_output){
  _tqt_backward_kernel<<<QUARK_GET_BLOCKS(N),QUARK_CUDA_NUM_THREADS>>>(
    N, x, scale, quant_min, quant_max, grad_logt, grad_output
  );
}
