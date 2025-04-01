//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#ifndef _INCLUDE_CPU_BFP_KERNEL_H_
#define _INCLUDE_CPU_BFP_KERNEL_H_

#include <stdint.h>

enum rounding_mode_enum {
  STD_ROUND = 0, // round half away from zero
  DPU_ROUND = 1, // round half upward
  PY3_ROUND = 2  // round half to even
};

uint32_t GetExponentCPU(float v);

uint32_t GetMaxExponentCPU(const float* input, int n);

float dpu_round(float x);

float py3_round(float x);

void LaunchBFPCPUKernel(const float* input,
                          float* output,
                          int n,
                          int bit_width,
                          int block_size,
                          int rounding_mode,
                          int use_compiler_version_cpu_kernel);

void Float2BFloat(float* input, int n);

void LaunchBFPPrimeCPUKernel(const float* input,
                             float* output,
                             int n,
                             const int bit_width,
                             const int block_size,
                             const int sub_block_size,
                             const int sub_block_shift_bits,
                             const int rounding_mode);

#endif // _INCLUDE_CPU_BFP_KERNEL_H_
