// Cross-platform `__device__` math shims for the 16-bit float types used by
// the MXFP4 kernels (`__half`, `__nv_bfloat16`) plus the E8M0 scale decoder.
//
// The wrappers paper over two portability gaps:
//   * CUDA compute capabilities <sm_80 (e.g. V100) lack native bfloat16
//     math, so the templates fall back to fp16-only paths under a
//     `__CUDA_ARCH__ >= 800` guard.
//   * ROCm/HIP names some intrinsics differently (`__shfl_xor` vs.
//     `__shfl_xor_sync`, missing `__bfloat162int_rn`, etc.).
//
// This header has *no* libtorch dependency; it lets the kernel `.cuh` files
// compile against any libtorch version (or none) and prepares the ground for
// moving the host wrappers to alternate libtorch surfaces (e.g. the stable
// ABI) without retouching the kernel sources. Format constants used below
// (`HALF_SIGN_BIT_POS`, `FLOAT16_*`, `BFLOAT16_*`) live in
// `mxfp4/mxfp4_format.cuh`.

#pragma once

#ifdef USE_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

// Required for ::cuda::std::is_same_v, used only for Nvidia GPUs.
// `cuda/std/type_traits` is not properly transpiled on AMD platforms (fatal
// error: 'cuda/std/type_traits' file not found), and likely requires
// https://github.com/ROCm/libhipcxx that is not shipped as part of ROCm.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_HCC__)
#include <cuda/std/type_traits>
#endif

#include "mxfp4/mxfp4_format.cuh"

template <typename T>
__device__ int bf16_or_half2int_rn(const T h);

template <typename T>
__device__ T float_to_bf16_or_half(const float x);

template <typename T>
__device__ float bf16_or_half_to_float(const T x);

template <typename T>
__device__ T shfl_xor_bf16_or_half(T x, int laneMask);

template <typename float_type, typename scale_type>
__device__ float_type e8m0_to_half(scale_type scale);

// Definitions

template <>
inline __device__ int bf16_or_half2int_rn(const __half h) {
  return __half2int_rn(h);
}

template <>
inline __device__ int bf16_or_half2int_rn(const __nv_bfloat16 h) {
  // __bfloat162int_rn is not implemented in ROCm hip/amd_detail/amd_hip_bf16.h.
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __float2int_rn(__bfloat162float(h));
#else
  printf(
    "ERROR in bf16_or_half2int_rn: compute capability does not support "
    "bfloat16.\n"
  );
  assert(1);
#endif
}

template <>
inline __device__ __half float_to_bf16_or_half(const float x) {
  return __float2half(x);
}

template <>
inline __device__ __nv_bfloat16 float_to_bf16_or_half(const float x) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __float2bfloat16(x);
#else
  printf(
    "ERROR in float_to_bf16_or_half: compute capability does not support "
    "bfloat16.\n"
  );
  assert(1);
#endif
}

template <>
inline __device__ float bf16_or_half_to_float(const __half x) {
  return __half2float(x);
}

template <>
inline __device__ float bf16_or_half_to_float(const __nv_bfloat16 x) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __bfloat162float(x);
#else
  printf(
    "ERROR in bf16_or_half_to_float: compute capability does not support "
    "bfloat16.\n"
  );
  assert(1);
#endif
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
inline __device__ __nv_bfloat16
shfl_xor_bf16_or_half(__nv_bfloat16 x, int laneMask) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)

#if defined USE_ROCM
  // `__shfl_xor_sync` does not exist for float16 in rocm 6.3.
  return __ushort_as_bfloat16(__shfl_xor(__bfloat16_as_ushort(x), laneMask));
#else
  return __ushort_as_bfloat16(
    __shfl_xor_sync(0xffffffff, __bfloat16_as_ushort(x), laneMask)
  );
#endif

#else
  printf(
    "ERROR in shfl_xor_bf16_or_half: compute capability does not support "
    "bfloat16.\n"
  );
  assert(1);
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
  scale_bits =
    scale_bits + (scale_biased << FLOAT16_MANTISSA_BITS) *
                   (scale_biased > 0 && scale_exp <= FLOAT16_EXP_BIAS);

  // Scales within [2**(-127), ..., 2**(-1)], with rounding to 0.
  scale_bits = scale_bits + (scale_biased <= 0 &&
                             scale_biased >= 1 - FLOAT16_MANTISSA_BITS) *
                              (1 << (FLOAT16_MANTISSA_BITS + scale_biased - 1));

  __half scale_half = *(__half*)(&scale_bits);

  return scale_half;
}

template <>
inline __device__ __nv_bfloat16 e8m0_to_half(uint8_t scale) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  int16_t scale_exp = (int16_t)scale - 127;

  __nv_bfloat16 scale_half = __float2bfloat16(powf(2.0, (float)scale_exp));

  scale_exp = scale_exp + BFLOAT16_EXP_BIAS;

  // Exactly representable scales in bf16: 2**(-127 - 7 + 1), ..., 2**127. All
  // good!
  uint16_t scale_bits = (scale_exp << BFLOAT16_MANTISSA_BITS) * (scale_exp > 0);
  scale_bits = scale_bits + (scale_exp <= 0) *
                              (1 << (BFLOAT16_MANTISSA_BITS + scale_exp - 1));

  return scale_half;
#else
  printf(
    "ERROR in e8m0_to_half: compute capability does not support bfloat16.\n"
  );
  assert(1);
#endif
}

// CUDA compute capabilities <sm_80 (e.g. V100, etc.) do not support bfloat16
// math. As `__CUDA_ARCH__` is only defined in device code, we handle the checks
// here.
template <typename float_type>
__device__ float_type hmul_impl(float_type a, float_type b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __hmul(a, b);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return __hmul(a, b);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

template <typename float_type>
__device__ float_type hdiv_impl(float_type a, float_type b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __hdiv(a, b);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return __hdiv(a, b);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

template <typename float_type>
__device__ float_type hmax_impl(float_type a, float_type b) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __hmax(a, b);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return __hmax(a, b);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

template <typename float_type>
__device__ float_type habs_impl(float_type a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return __habs(a);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return __habs(a);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

template <typename float_type>
__device__ float_type hlog2_impl(float_type a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return hlog2(a);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return hlog2(a);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

template <typename float_type>
__device__ float_type hfloor_impl(float_type a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || \
  defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  return hfloor(a);
#else
  if constexpr (::cuda::std::is_same_v<float_type, __half>) {
    return hfloor(a);
  } else {
    printf("ERROR: compute capability does not support bfloat16.\n");
    assert(1);
  }
#endif
}

#endif  // USE_CUDA
