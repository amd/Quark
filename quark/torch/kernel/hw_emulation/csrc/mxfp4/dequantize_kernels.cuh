// Shared device-side templates for the MXFP4 dequantization kernel.
//
// Both the stable-ABI build (`csrc/mxfp4/dequantize.cu`) and the legacy
// pybind11 build (`csrc/legacy/mxfp4/dequantize.cu`) include this header so
// the kernel implementation lives in exactly one place. The `.cu` files only
// contribute the host-side wrapper that bridges to their respective PyTorch
// API surface.

#pragma once

#include <cstdint>

#include "mxfp4/low_precision_intrinsics.cuh"
#include "mxfp4/mxfp4_format.cuh"

#ifdef USE_CUDA

// ---------------------------------------------------------------------------
// Dequantization kernel work distribution.
//
// Each thread loads one uint32_t (4 packed bytes = 8 fp4 values) and writes
// out 8 dequantized half/bfloat16 values via a single 16-byte vectorized
// store. Since one MX scale covers a group of 32 fp4 values, exactly 4
// threads cooperate on a single group.
// ---------------------------------------------------------------------------
constexpr uint32_t BYTES_PER_THREAD = sizeof(uint32_t);                   // 4
constexpr uint32_t OUTPUTS_PER_THREAD = BYTES_PER_THREAD * FP4_PER_BYTE;  // 8
constexpr uint32_t THREADS_PER_GROUP =
  MXFP4_GROUP_SIZE / OUTPUTS_PER_THREAD;  // 4

template <
  typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits,
  uint32_t half_exp_bias>
__device__ float_type upcast_fp4_to_fp16_or_bf16(uint8_t val) {
  // Takes one fp4_e2m1 value packed in the low nibble (b0000xxxx) and
  // converts it to the corresponding fp16 / bf16 value.

  bool sign = val >> FP4_SIGN_BIT_POS;

  uint8_t exp = (val >> FLOAT4_MANTISSA_BITS) & FP4_EXP_MASK;
  uint8_t new_mantissa = val & FP4_MANTISSA_MASK;

  // if exp == 0 and new_mantissa == 0:
  //     new_exp = 0
  // else:
  //     new_exp = exp - FLOAT4_EXP_BIAS + half_exp_bias

  // int8_t works with float16, but may overflow with bfloat16.
  int16_t new_exp = exp - FLOAT4_EXP_BIAS + half_exp_bias;

  // Cast b0000 to 0. in fp16/bf16.
  new_exp = new_exp * (exp > 0 || new_mantissa > 0);

  // Cast b0001 to 0.5 in fp16/bf16.
  new_mantissa = new_mantissa && (exp > 0);

  uint16_t qdq_val = (sign << HALF_SIGN_BIT_POS) +
                     (new_exp << half_mantissa_bits) +
                     (new_mantissa << (half_mantissa_bits - 1));
  float_type result = *(float_type*)(&qdq_val);
  return result;
}

template <
  typename float_type, typename scale_type, uint32_t half_exp_bits,
  uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__global__ void dq_uint8_mxfp4_to_half_kernel(
  uint8_t* inp, scale_type* scales, float_type* out
) {
  // One thread handles OUTPUTS_PER_THREAD output values.
  // Thus, THREADS_PER_GROUP threads handle one group.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float_type out_thread[OUTPUTS_PER_THREAD];
  uint8_t elems[BYTES_PER_THREAD];

  reinterpret_cast<float*>(elems)[0] = reinterpret_cast<float*>(inp)[idx];

  float_type scale_half =
    e8m0_to_half<float_type, scale_type>(scales[idx / THREADS_PER_GROUP]);

  for (uint32_t i = 0; i < BYTES_PER_THREAD; i++) {
    uint8_t elem = elems[i];

    // Tensor packed as [elem0, elem1], but the logical order is [elem1, elem0].
    uint8_t elem0 = elem >> FP4_NUM_BITS;
    uint8_t elem1 = elem & FP4_VAL_MASK;

    float_type elem0_half = upcast_fp4_to_fp16_or_bf16<
      float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(elem0);
    float_type elem1_half = upcast_fp4_to_fp16_or_bf16<
      float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(elem1);

    // Tensor packed as [elem0, elem1], but the logical order is [elem1, elem0].
    // TODO: We could probably use half2 dtype here.
    out_thread[FP4_PER_BYTE * i + 1] = hmul_impl(elem0_half, scale_half);
    out_thread[FP4_PER_BYTE * i] = hmul_impl(elem1_half, scale_half);
  }

  // Maps to a global_store_dwordx4
  // (4 * sizeof(float) bytes = 16 bytes = OUTPUTS_PER_THREAD halfs).
  reinterpret_cast<double2*>(out)[idx] =
    reinterpret_cast<double2*>(out_thread)[0];
}

#endif  // USE_CUDA
