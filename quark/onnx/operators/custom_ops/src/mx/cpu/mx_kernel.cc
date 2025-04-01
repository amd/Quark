//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include "bfp/cpu/bfp_kernel.h"
#include "mx/cpu/mx_kernel.h"

#include "mx/funcs.cuh"  // Re-use the MX kernel of Quark-Torch

#ifdef USE_ORT_FP8_KERNELS
#include "core/framework/float8.h"  // Use the fp8 data type of ORT
#endif


void MXCPUKernel(const float* input,
                 float* output,
                 const int n,
                 const int index,
                 const int stride,
                 const int ebits,
	         const int mbits,
		 const int emax,
		 const float max_norm,
		 const float min_norm,
                 const rounding_mode_enum rounding_mode) {
  uint32_t shared_exp = 0;
  // Loop over block to find shared exponent.
  for (int i = index; i < n; i += stride) {
    uint32_t exp = GetExponentCPU(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents.
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  shared_exp_value = shared_exp_value - emax;
  // The format of share scale should be 'E8M0' at this moment
  float shared_scale = std::pow(2.0, shared_exp_value);
  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponentCPU(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float element = input[i] / shared_scale;

      if (ebits > 0) {
	// For MXFP8, MXFP6, MXFP4
#ifdef USE_ORT_FP8_KERNELS
	if (ebits == 5 && mbits == 2) {
	    onnxruntime::Float8E5M2 blf = onnxruntime::Float8E5M2(element);
	    element = blf.ToFloat();
	} else if (ebits == 4 && mbits == 3) {
	    onnxruntime::Float8E4M3FN blf = onnxruntime::Float8E4M3FN(element);
	    element = blf.ToFloat();
	} else {
            element = fake_quantize_element(element, max_norm, ebits, mbits, ROUND_HALF_TO_EVEN);
	}
#else
        // TODO: The kernel should support more rounding modes, currently it
        // supports rounding half to even (corresponding to PY3_ROUND) only
        element = fake_quantize_element(element, max_norm, ebits, mbits, ROUND_HALF_TO_EVEN);
#endif
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
                element = dpu_round(element / scale);
                break;
            case PY3_ROUND:
                element = py3_round(element / scale);
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

      output[i] = shared_scale * element;
    }
  }
}

void LaunchMXCPUKernel(const float* input,
                       float* output,
                       const int n,
                       const int block_size,
                       const int ebits,
                       const int mbits,
		       const int emax,
                       const float max_norm,
                       const float min_norm,
                       const int rounding_mode) {
  int num_blocks = n / block_size;
  for (int index = 0; index < num_blocks; index++) {
    MXCPUKernel(input, output, index * block_size + block_size,
                index * block_size, 1, ebits, mbits, emax, max_norm, min_norm,
		(rounding_mode_enum)rounding_mode);
  }
}
