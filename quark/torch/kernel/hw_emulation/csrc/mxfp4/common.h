#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)

#define FLOAT16_MANTISSA_BITS 10
#define FLOAT16_EXP_BITS 5
#define FLOAT16_EXP_BIAS 15

#define FLOAT4_MANTISSA_BITS 1
#define FLOAT4_EXP_BITS 2
#define FLOAT4_EXP_BIAS 1

#define FLOAT8_E8M0_MAX_EXP 127

#define BFLOAT16_MANTISSA_BITS 7
#define BFLOAT16_EXP_BITS 8
#define BFLOAT16_EXP_BIAS 127

#define FLOAT16_VAL_TO_ADD (1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define FLOAT16_SIGN_EXPONENT_MASK (((1 << (FLOAT16_EXP_BITS + 1)) - 1) << FLOAT16_MANTISSA_BITS)

#define BFLOAT16_VAL_TO_ADD (1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define BFLOAT16_SIGN_EXPONENT_MASK (((1 << (BFLOAT16_EXP_BITS + 1)) - 1) << BFLOAT16_MANTISSA_BITS)

#ifndef func_defined_common
#define func_defined_common

template <typename T>
__device__ int bf16_or_half2int_rn(const T h);

template <typename T>
__device__ T float_to_bf16_or_half(const float x);

template <typename T>
__device__ float bf16_or_half_to_float(const T x);

template <typename T>
__device__ T shfl_xor_bf16_or_half(T x, int laneMask);

template<typename float_type, typename scale_type>
__device__ float_type e8m0_to_half(scale_type scale);

// Definitions

template <>
inline __device__ int bf16_or_half2int_rn(const __half h) {
    return __half2int_rn(h);
}

template <>
inline __device__ int bf16_or_half2int_rn(const __nv_bfloat16 h) {
    // __bfloat162int_rn is not implemented in ROCm hip/amd_detail/amd_hip_bf16.h.
    return __float2int_rn(__bfloat162float(h));
}

template <>
inline __device__ __half float_to_bf16_or_half(const float x) {
    return __float2half(x);
}

template <>
inline __device__ __nv_bfloat16 float_to_bf16_or_half(const float x) {
    return __float2bfloat16(x);
}

template <>
inline __device__ float bf16_or_half_to_float(const __half x) {
    return __half2float(x);
}

template <>
inline  __device__ float bf16_or_half_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
inline __device__ __half shfl_xor_bf16_or_half(__half x, int laneMask) {
#if defined USE_ROCM
    // `__shfl_xor_sync` does not exist for float16 in rocm 6.3.
    return __shfl_xor(x, laneMask);
#else
    return __shfl_xor_sync(0xffffffff, x, laneMask);
#endif
}

template <>
inline __device__ __nv_bfloat16 shfl_xor_bf16_or_half(__nv_bfloat16 x, int laneMask) {
#if defined USE_ROCM
    // `__shfl_xor_sync` does not exist for float16 in rocm 6.3.
    return __ushort_as_bfloat16(__shfl_xor(__bfloat16_as_ushort(x), laneMask));
#else
    return __ushort_as_bfloat16(__shfl_xor_sync(0xffffffff, __bfloat16_as_ushort(x), laneMask));
#endif
}

template <>
inline __device__ __half e8m0_to_half(__half scale) {
    return scale;
}

template <>
inline __device__ __nv_bfloat16 e8m0_to_half(__nv_bfloat16 scale) {
    return scale;
}

template <>
inline __device__ __half e8m0_to_half(uint8_t scale) {
    int16_t scale_exp = (int16_t)scale - 127;

    int16_t scale_biased = scale_exp + FLOAT16_EXP_BIAS;

    // Exactly representable scales in fp16: 2**(-15 - 10 + 1), ..., 2**15.
    // Round to 0 and 2**15.

    // Handle scales larger than 2**15.
    uint16_t scale_bits = (scale_exp > FLOAT16_EXP_BIAS) * 0x7800;

    // Scales within [2**0, ..., 2**15].
    scale_bits = scale_bits + (scale_biased << FLOAT16_MANTISSA_BITS) * (scale_biased > 0 && scale_exp <= FLOAT16_EXP_BIAS);

    // Scales within [2**(-127), ..., 2**(-1)], with rounding to 0.
    scale_bits = scale_bits + (scale_biased <= 0 && scale_biased >= 1 - FLOAT16_MANTISSA_BITS) * (1 << (FLOAT16_MANTISSA_BITS + scale_biased - 1));

    __half scale_half = *(__half*)(&scale_bits);

    return scale_half;
}

template <>
inline __device__ __nv_bfloat16 e8m0_to_half(uint8_t scale) {
    int16_t scale_exp = (int16_t)scale - 127;

    __nv_bfloat16 scale_half = __float2bfloat16(powf(2.0, (float)scale_exp));

    scale_exp = scale_exp + BFLOAT16_EXP_BIAS;

    // Exactly representable scales in bf16: 2**(-127 - 7 + 1), ..., 2**127. All good!
    uint16_t scale_bits = (scale_exp << BFLOAT16_MANTISSA_BITS) * (scale_exp > 0);
    scale_bits = scale_bits + (scale_exp <= 0) * (1 << (BFLOAT16_MANTISSA_BITS + scale_exp - 1));

    return scale_half;
}

#endif
#endif
