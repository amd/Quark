//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "bfp/cuda/bfp_kernel.h"
#include <stdio.h>
#include <cstdint>
#include <string>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>



// Get a unsinged value of the biased exponent.
__device__ uint32_t GetExponent(float v) {
  uint32_t uint_v = __float_as_uint(v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

// Get a unsinged value of the max biased exponent.
__device__ uint32_t GetMaxExponent(const float* input, int n) {
  uint32_t max_exp = 0;
  for (int i = 0; i < n; i++) {
    max_exp = max(max_exp, GetExponent(input[i]));
  }
  return max_exp;
}

__device__ uint32_t GetMantissa(float v) {
  uint32_t uint_v = *reinterpret_cast<uint32_t*>(&v);
  return uint_v & 0x7fffff;
}

__device__ uint32_t Rand(int seed) {
  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + seed, 0, 0, &state);
  return curand(&state);
}

__device__ float RandUniform(int seed) {
  curandState state;
  curand_init(static_cast<uint64_t>(clock()) + seed, 0, 0, &state);
  return curand_uniform(&state);
}

__device__ float StochsticRound(float v, int seed) {
  return floorf(v + RandUniform(seed));
}

__device__ float dpu_round(float x) {
  return ((x < 0) && (x - floor(x) == 0.5))
              ? std::ceil(x)
              : std::round(x);
}

__device__ float py3_round(float x) {
  float x_floor = std::floor(x);
  float diff = x - x_floor;
  if (diff > 0.5) return x_floor + 1;
  else if (diff == 0.5) return (int)x_floor % 2 == 1 || (int)x_floor % 2 == -1 ? x_floor + 1 : x_floor;
  else return x_floor;
}

namespace vai_q_cuda {

  typedef union value_convert {
    std::uint32_t u;
    std::int32_t i;
    float f;
  } value_convert_t;

   inline __device__ std::uint32_t f_to_u(float data) {
    value_convert_t vc{};
    vc.f = data;
    return vc.u;
  }

   inline __device__ float u_to_f(std::uint32_t data) {
    value_convert_t vc{};
    vc.u = data;
    return vc.f;
  }
  
  float __device__ float2bfloat_cuda(const float x) {
    std::uint32_t itmp = f_to_u(x);           // float32 bitwise to int32
    if ((itmp & 0x00008000) == 0x00008000) {  // half even
      if ((itmp & 0xFFFF) > 0x00008000 ||
          (((itmp & 0xFFFF) == 0x00008000) && (itmp & 0x10000) == 0x10000)) {
        itmp += 0x10000;
      }
    }
    itmp &= 0xFFFF0000;
    return u_to_f(itmp);  // int32 bitwise to float32
  }
}

__global__ void BFloatCUDAKernel(float* input, int num_threads)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {
    return;
  }
  input[index] = vai_q_cuda::float2bfloat_cuda(input[index]);

}

__global__ void BFPCUDAKernel(const float* input,
                                float* output,
                                const int num_threads,
                                const int axis_size,
                                const int bit_width,
                                const int block_size,
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
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = pow(2.0, shared_exp_value - (m_bits - 1));
  auto max_v = pow(2.0, shared_exp_value + 1) - scale;
  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    auto index = offset + i;
    auto exp = GetExponent(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } else {
      auto scaled_x = input[index] / scale;
      float rounded_scaled_x;
      switch (rounding_mode)
      {
      case STD_ROUND:
        rounded_scaled_x = std::round(scaled_x);
        break;
      case DPU_ROUND:
        rounded_scaled_x = dpu_round(scaled_x);
        break;
      case PY3_ROUND:
        rounded_scaled_x = py3_round(scaled_x);
        break;
      default:
        break;
      }
      output[index] = max(-max_v, min(rounded_scaled_x * scale, max_v));
    }
  }
}

__global__ void BFPCUDAKernelCompiler(const float* input,
                                float* output,
                                const int num_threads,
                                const int axis_size,
                                const int bit_width,
                                const int block_size,
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
    uint32_t exp = GetExponent(input[offset + i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }
  
  // Minus 127 to get unbiased value.
  int shared_exp_value = static_cast<int>(shared_exp) - 127;
  // 1 sign bit, 8 exp bits.
  int m_bits = bit_width - 9;
  auto scale = pow(2.0, shared_exp_value - (m_bits - 1));
  
  for (int i = 0; i < block_size; i++) {
    // Output +-0/NaN/Inf as is.
    auto index = offset + i;
    auto exp = GetExponent(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } else {
      auto scaled_x = input[index] / scale;
      float x;
      switch (rounding_mode)
      {
      case STD_ROUND:
        x = std::round(scaled_x);
        break;
      case DPU_ROUND:
        x = dpu_round(scaled_x);
        break;
      case PY3_ROUND:
        x = py3_round(scaled_x);
        break;
      default:
        break;
      }
      if (x >= 128 || x < -128) {
        shared_exp += 1;
        shared_exp_value += 1;
        scale *= 2.0;
        break;
      }
    }
  }
  float max_v = std::pow(2.0, shared_exp_value) * (std::pow(2.0, m_bits) - 1);
  float min_v = -std::pow(2.0, shared_exp_value) * (std::pow(2.0, m_bits));
  for (int i = 0; i < block_size; i++) {
    auto index = offset + i;
    // Output +-0/NaN/Inf as is.
    uint32_t exp = GetExponent(input[index]);
    if (exp == 0xff) {
      output[index] = input[index];
    } 
    else {
      float x;
      switch (rounding_mode)
      {
      case STD_ROUND:
        x = std::round(input[index] / scale) * scale;
        break;
      case DPU_ROUND:
        x = dpu_round(input[index] / scale) * scale;
        break;
      case PY3_ROUND:
        x = py3_round(input[index] / scale) * scale;
        break;
      default:
        break;
      }
      // Clamp(x, min_v, max_v)
      output[index] = max(min_v, min(x, max_v));
    }
  }
}
void LaunchBFloatCUDAKernel(float* input, int n){
  int threads_per_block = 256;
  int blocks = static_cast<int>(ceil(static_cast<float>(n) / threads_per_block));
  cudaDeviceSynchronize();
  BFloatCUDAKernel<<<blocks, threads_per_block>>>(input, n);
  cudaDeviceSynchronize();
}
void LaunchBFPCUDAKernel(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int bit_width,
    const int block_size,
    const int rounding_mode,
    int use_compiler_version_cpu_kernel) {
  
  const int threads = n / block_size;
  
  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));
  cudaDeviceSynchronize();
  if (use_compiler_version_cpu_kernel == 0) {
    BFPCUDAKernel<<<blocks, threads_per_block>>>(
      input, output, threads,
      axis_size, bit_width, block_size,
      (rounding_mode_enum)rounding_mode
    );
  } else {
    BFPCUDAKernelCompiler<<<blocks, threads_per_block>>>(
      input, output, threads,
      axis_size, bit_width, block_size,
      (rounding_mode_enum)rounding_mode
    );
  }
  cudaDeviceSynchronize();
}

// Notable things:
// 1. +-INF are converted to NaNs
// 2. All subnormal numbers are flushed to zeros.
// 3. When the shared exponent is 2^w - 1, all k values in a block are NaNs
__global__ void BFPPrimeCUDAKernel(const float* input,
                                   float* output,
                                   const int num_threads,
                                   const int axis_size,
                                   const int bit_width,
                                   const int block_size,
                                   const int sub_block_size,
                                   const int sub_block_shift_bits,
                                   const int rounding_mode) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= num_threads) {
    return;
  }

  const int axis_blocks = axis_size / block_size;
  const int block_index = index % axis_blocks;
  const int axis_index = index / axis_blocks;


  // Mantissa bits of float32.
  const uint32_t m_float = 23;
  // Mantissa bits of bfp, sign: 1 bit, exponent: 8 bits.
  const uint32_t m_bfp = bit_width - 9;
  const uint32_t exp_bias = 127;

  int offset = axis_index * axis_size + block_index * block_size;
  uint32_t shared_exp = GetMaxExponent(input + offset, block_size);

  for (int i = 0; i < block_size / sub_block_size; i++) {
    uint32_t max_sub_exp = GetMaxExponent(
        input + offset + i * sub_block_size, sub_block_size);

    // Sub-block shift is the difference between the shared exponent and
    // the maximum exponent in the sub-block, upper bounded by 2^d - 1.
    uint32_t shift;
    uint32_t shift_upper_bound = (1 << sub_block_shift_bits) - 1;
    if (shared_exp - max_sub_exp > shift_upper_bound) {
      shift = shift_upper_bound;
    } else {
      shift = shared_exp - max_sub_exp;
    }

    for (int j = 0; j < sub_block_size; j++) {
      auto idx = offset + i * sub_block_size + j;
      uint32_t input_x = __float_as_uint(input[idx]);
      uint32_t exp = (input_x & 0x7f800000) >> m_float;
      uint32_t mantissa;
      if (exp == 0) {
        // Subnormals are flushed to zero.
        mantissa = 0;
      } else {
        // Add leading 1.
        mantissa = (input_x & 0x7fffff) | (1 << m_float);
      }
      // Right shift mantissa by the exponent difference + the mantissa bitwidth difference
      uint32_t num_bits_shifting = shared_exp - shift - exp + m_float - m_bfp;
      if (num_bits_shifting >= 32) {
        // Shift a number of bits equal or greater than the bit width is undefined behavior.
        num_bits_shifting = 31;
      }
      mantissa >>= num_bits_shifting;
      // Do not round up if mantissa is all 1s.
      if (rounding_mode != 0 && mantissa != ((1 << (m_bfp + 1)) - 1)) {
        // 0: Round toward zero
        // 1: Round half away from zero
        // 2: Stochstic rounding
        if (rounding_mode == 1) {
          mantissa += 1;
        } else if (rounding_mode == 2) {
          if (num_bits_shifting < 1 + m_float) {
            uint32_t mask = (1 << (num_bits_shifting + 1)) - 1;
            uint32_t p = Rand(idx);
            if ((p & mask) < (input_x & mask)) {
              mantissa += 2;
            }
          }
        }
      }

      // Truncate the last bit
      mantissa >>= 1;
      int sign = input_x & 0x80000000 ? -1 : 1;
      // If any number in a block is +-Inf/NaN,
      // then every number in the output is a NaN.
      if (shared_exp == 0xff) {
        output[idx] = __uint_as_float(0x7fffffff);
      } else {
        // v = (−1)^s * 2^(E - bias) * 2^(-D) * 2^(1-m) * M
        output[idx] = sign * std::pow(2.0,
            static_cast<int>(shared_exp - exp_bias - shift + 1 - m_bfp)) * mantissa;
      }
    }
  }
}

void LaunchBFPPrimeCUDAKernel(
    const float* input,
    float* output,
    const int n,
    const int axis_size,
    const int bit_width,
    const int block_size,
    const int sub_block_size,
    const int sub_block_shift_bits,
    const int rounding_mode) {

  const int threads = n / block_size;

  int threads_per_block = 256;
  int blocks = static_cast<int>(
      ceil(static_cast<float>(threads) / threads_per_block));

  cudaDeviceSynchronize();
  BFPPrimeCUDAKernel<<<blocks, threads_per_block>>>(
    input,
    output,
    threads,
    axis_size,
    bit_width,
    block_size,
    sub_block_size,
    sub_block_shift_bits,
    rounding_mode
  );
  cudaDeviceSynchronize();
}
