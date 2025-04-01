//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#ifndef _INCLUDE_CUDA_BFP_KERNEL_H_
#define _INCLUDE_CUDA_BFP_KERNEL_H_

#include <stdint.h>

enum rounding_mode_enum {
  STD_ROUND = 0, // round half away from zero
  DPU_ROUND = 1, // round half upward
  PY3_ROUND = 2  // round half to even
};

#ifdef __CUDACC__
extern "C" {
// TODO: set --relocatable-device-code=true for NVCC
__device__ uint32_t GetExponent(float v);

__device__ uint32_t GetMaxExponent(const float* input, int n);

__device__ float dpu_round(float x);

__device__ float py3_round(float x);
}
#endif

void LaunchBFPCUDAKernel(const float* input,
                           float* output,
                           const int n,
                           const int axis_size,
                           const int bit_width,
                           const int block_size,
                           const int rounding_mode,
                           int use_compiler_version_cpu_kernel);
                           
void LaunchBFloatCUDAKernel(float* input, int n);

void LaunchBFPPrimeCUDAKernel(const float* input,
                              float* output,
                              const int n,
                              const int axis_size,
                              const int bit_width,
                              const int block_size,
                              const int sub_block_size,
                              const int sub_block_shift_bits,
                              const int rounding_mode);

#endif // _INCLUDE_CUDA_BFP_KERNEL_H_
