//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "bfp/cuda/bfp_kernel.h"
#include "mx/cuda/mx_kernel.h"
#include <stdio.h>
#include <string>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "mx/funcs.cuh"  // Re-use the MX kernel of Quark-Torch


// Get a unsinged value of the biased exponent.
__device__ uint32_t GetExponent_(float v) {
  uint32_t uint_v = __float_as_uint(v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

__device__ float dpu_round_(float x) {
  return ((x < 0) && (x - floor(x) == 0.5))
              ? std::ceil(x)
              : std::round(x);
}

__device__ float py3_round_(float x) {
  float x_floor = std::floor(x);
  float diff = x - x_floor;
  if (diff > 0.5) return x_floor + 1;
  else if (diff == 0.5) return (int)x_floor % 2 == 1 || (int)x_floor % 2 == -1 ? x_floor + 1 : x_floor;
  else return x_floor;
}

__global__ void MXCUDAKernel(const float* input,
                   float* output,
                   const int num_threads,
                   const int axis_size,
                   const int block_size,
                   const int ebits,
                   const int mbits,
                   const int emax,
                   const float max_norm,
                   const float min_norm,
                   const rounding_mode_enum rounding_mode) { 
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {
    return;
  }

  const int axis_blocks = axis_size / block_size;
  const int block_index = index % axis_blocks;
  const int axis_index = index / axis_blocks;
  
  int offset = axis_index * axis_size + block_index * block_size;
  // Loop over bounding box to find shared exponent
  uint32_t shared_exp = 0;
  for (int i = 0; i < block_size; i++) {
    uint32_t exp = GetExponent_(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  shared_exp_value = shared_exp_value - emax;
  // The format of share scale should be 'E8M0' at this moment
  float shared_scale = std::pow(2.0, shared_exp_value);
  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    auto index = offset + i;
    uint32_t exp = GetExponent_(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } else {
      float element = input[index] / shared_scale;
      if (ebits > 0) {
        // For MXFP8, MXFP6, MXFP4
        // TODO: The kernel should support more rounding modes, currently it
        // supports rounding half to even (corresponding to PY3_ROUND) only
        element = fake_quantize_element(element, max_norm, ebits, mbits, ROUND_HALF_TO_EVEN);
      } else {
        // For MXINT8
        int implicit_scale_bits = 6;
        float scale = std::pow(2.0, -implicit_scale_bits);
        switch (rounding_mode)
        {
            case STD_ROUND:
                element = std::round(element / scale);
                break;
            case DPU_ROUND:
                element = dpu_round_(element / scale);
                break;
            case PY3_ROUND:
                element = py3_round_(element / scale);
                break;
            default:
                element = std::round(element / scale);
                break;
        }
        // Clamping (saturating) the value to the maximum INT8 magnitude
        element = std::max(min_norm, std::min(element, max_norm));
        // Restore true value of the elements (similar to dequantizing)
        element *= scale;
      }

      output[index] = shared_scale * element;
    }
  }
}

void LaunchMXCUDAKernel(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int block_size,
    const int ebits,
    const int mbits,
    const int emax,
    const float max_norm,
    const float min_norm,
    const int rounding_mode) {

  const int threads = n / block_size;

  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));
  cudaDeviceSynchronize();
  MXCUDAKernel<<<blocks, threads_per_block>>>(
      input, output, threads,
      axis_size, block_size,
      ebits, mbits, emax, max_norm, min_norm,
      (rounding_mode_enum)rounding_mode
    );
  cudaDeviceSynchronize();
}
