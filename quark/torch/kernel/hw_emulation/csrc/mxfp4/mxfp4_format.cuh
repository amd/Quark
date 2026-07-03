// Bit-layout constants for the MXFP4 (FP4 e2m1 + E8M0 scale) format and the
// FP16 / BF16 types it round-trips through.
//
// This header is pure C++: it has no CUDA, HIP, or libtorch dependency, so it
// can be included from any translation unit (host code, device code, kernel
// `.cuh` files, or alternate libtorch surfaces) without dragging in toolchain-
// specific headers. Device-side math helpers that consume these constants live
// in `mxfp4/low_precision_intrinsics.cuh`.

#pragma once

#include <climits>
#include <cstdint>

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

#define FLOAT16_VAL_TO_ADD \
  (1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define FLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (FLOAT16_EXP_BITS + 1)) - 1) << FLOAT16_MANTISSA_BITS)

#define BFLOAT16_VAL_TO_ADD \
  (1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define BFLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (BFLOAT16_EXP_BITS + 1)) - 1) << BFLOAT16_MANTISSA_BITS)

// ---------------------------------------------------------------------------
// FP4 (e2m1) bit layout: [sign : 1 | exp : 2 | mantissa : 1]. See the
// `FLOAT4_*` macros above for the per-field widths/biases.
// ---------------------------------------------------------------------------
constexpr uint32_t FP4_NUM_BITS =
  1 + FLOAT4_EXP_BITS + FLOAT4_MANTISSA_BITS;  // 4
constexpr uint32_t FP4_SIGN_BIT_POS =
  FLOAT4_EXP_BITS + FLOAT4_MANTISSA_BITS;  // 3
constexpr uint8_t FP4_EXP_MASK =
  static_cast<uint8_t>((1u << FLOAT4_EXP_BITS) - 1u);  // 0b11
constexpr uint8_t FP4_MANTISSA_MASK =
  static_cast<uint8_t>((1u << FLOAT4_MANTISSA_BITS) - 1u);  // 0b1
constexpr uint8_t FP4_VAL_MASK =
  static_cast<uint8_t>((1u << FP4_NUM_BITS) - 1u);          // 0xF
constexpr uint32_t FP4_PER_BYTE = CHAR_BIT / FP4_NUM_BITS;  // 2

// FP4 e2m1 unbiased exponent range: smallest non-zero subnormal magnitude is
// 0.5 = 2**-1, largest finite magnitude is 6.0 (mantissa=1, exp=3) which has
// unbiased exponent 2.
constexpr int32_t FP4_MAX_NORMAL_EXP_UNBIASED =
  static_cast<int32_t>((1u << FLOAT4_EXP_BITS) - 1u) -
  static_cast<int32_t>(FLOAT4_EXP_BIAS);  // 2
constexpr int32_t FP4_MIN_NONZERO_EXP_UNBIASED =
  1 - static_cast<int32_t>(FLOAT4_EXP_BIAS) -
  static_cast<int32_t>(FLOAT4_MANTISSA_BITS);  // -1

// fp16 and bf16 are 16-bit types whose sign bit lives in the MSB. Both
// `FLOAT16_EXP_BITS + FLOAT16_MANTISSA_BITS` and the bf16 equivalent are 15.
constexpr uint32_t HALF_SIGN_BIT_POS = 15;

// MX block / scale group layout. See OCP MX spec.
constexpr uint32_t MXFP4_GROUP_SIZE = 32;

// CUDA / HIP warp width. All currently supported architectures use 32-lane
// warps for the warp-level shuffles used by the device intrinsics in
// `mxfp4/low_precision_intrinsics.cuh`.
constexpr uint32_t WARP_SIZE = 32;
