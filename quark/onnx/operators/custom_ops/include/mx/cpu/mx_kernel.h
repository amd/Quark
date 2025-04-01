//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#ifndef _INCLUDE_CPU_MX_KERNEL_H_
#define _INCLUDE_CPU_MX_KERNEL_H_

void LaunchMXCPUKernel(const float* input,
                       float* output,
                       const int n,
                       const int block_size,
                       const int ebits,
                       const int mbits,
                       const int emax,
                       const float max_norm, // max normal value
                       const float min_norm, // min normal value
                       const int rounding_mode);

#endif // _INCLUDE_CPU_BFP_KERNEL_H_
