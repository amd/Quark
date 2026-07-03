// Shared device-side templates for the MXFP4 fake-quantization kernel.
//
// Both the stable-ABI build (`csrc/mxfp4/fake.cu`) and the legacy pybind11
// build (`csrc/legacy/mxfp4/fake.cu`) include this header so the kernel
// implementation lives in exactly one place. The `.cu` files only contribute
// the host-side wrapper that bridges to their respective PyTorch API
// surface.

#pragma once

#include <climits>
#include <cstdint>

#include "mxfp4/low_precision_intrinsics.cuh"
#include "mxfp4/mxfp4_format.cuh"

#ifdef USE_CUDA

// `(1 << tail_bits) - 1` is computed below in `uint16_t` arithmetic, so
// `tail_bits` must stay strictly below 16 to avoid undefined-behavior shifts.
constexpr int kUint16Bits = static_cast<int>(sizeof(uint16_t) * CHAR_BIT);

template <
  typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits,
  uint32_t half_exp_bias>
__device__ float_type fp16_to_fp4_simulate(float_type* val) {
  // Casts an fp16 input to the restricted values of float4_e2m1,
  // that is to say [0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0,
  // -1.5, -2.0, -3.0, -4.0, -6.0].

  uint16_t val_view = *(uint16_t*)val;

  uint16_t exp = val_view >> half_mantissa_bits;
  exp = exp & ((1u << half_exp_bits) - 1u);

  bool sign = (val_view >> HALF_SIGN_BIT_POS) & 1u;

  // Last bit of the half mantissa that survives rounding to fp4 mantissa.
  bool mantissa_last =
    (val_view >> (half_mantissa_bits - FLOAT4_MANTISSA_BITS)) & 1u;

  int16_t exp_unbias = exp - half_exp_bias;
  int16_t new_exp = exp_unbias + FLOAT4_EXP_BIAS;

  int16_t exp_shift = (new_exp <= 0) * (1 - new_exp);

  // Typically `half_mantissa_bits - FLOAT4_MANTISSA_BITS` (= 9 for fp16,
  // 6 for bf16). Cap at the uint16 bit width to prevent overflow on
  // `uint16_t half` for very small values, which are correctly mapped to
  // `round_close` regardless.
  uint16_t tail_bits = min(
    kUint16Bits, static_cast<int>(half_mantissa_bits) -
                   static_cast<int>(FLOAT4_MANTISSA_BITS) + exp_shift
  );

  // Half mantissa plus one extra low exponent bit (acts as the implicit
  // leading 1 for the rounding arithmetic below).
  uint16_t mantissa_plus_one =
    val_view & ((1u << (half_mantissa_bits + FLOAT4_MANTISSA_BITS)) - 1u);

  uint16_t half = 1u << (tail_bits - 1);

  uint16_t tail = mantissa_plus_one & ((1u << tail_bits) - 1u);

  bool round_close = (tail < half);  // round towards 0
  bool round_away = (tail > half);   // round away from 0
  bool tie = tail == half;

  uint16_t new_mantissa;

  bool new_mantissa_close = 0;
  uint16_t new_exp_close = 0;

  bool new_mantissa_away = 0;
  uint16_t new_exp_away = 0;

  uint16_t new_exp_tie = 0;

  // # 1. round down
  // if new_exp == 0: # case [0.5, 0.749999]
  //     new_mantissa = 0
  // elif new_exp < 0:  # case [0, 0.24999]
  //     new_mantissa = 0
  // else:
  //     new_mantissa = mantissa_last

  new_mantissa_close = (new_exp > 0) * mantissa_last;
  new_exp_close = exp;

  // # 2. round up
  // if new_exp <= 0:  # case [0.250001, 0.499999] and [0.75001, 0.99999]
  //     new_mantissa = 0
  //     new_exp += 1
  // elif mantissa_last == 0:
  //     new_mantissa = 1
  // else:
  //     new_mantissa = 0
  //     new_exp += 1

  new_mantissa_away = (new_exp > 0) && (mantissa_last == 0);
  new_exp_away = exp + ((new_exp <= 0) || (mantissa_last == 1));

  // # 3. tie
  // Smallest non-zero fp4 magnitude is 0.5 (fp16-biased exp =
  // half_exp_bias + FP4_MIN_NONZERO_EXP_UNBIASED = half_exp_bias - 1); the
  // tie midpoint to zero sits one binade below at 0.25. Values with biased
  // exp at-or-below that midpoint are mapped to 0.
  // 0.25 -> 0.
  // 0.75 -> 1.
  // 1.25 -> 1.
  // 1.75 -> 2.
  // 2.5 -> 2.
  // 3.5 -> 4.
  // 5. -> 4.
  constexpr int half_exp_bias_signed = static_cast<int>(half_exp_bias);
  constexpr int fp4_tie_to_zero_biased_exp =
    half_exp_bias_signed + FP4_MIN_NONZERO_EXP_UNBIASED - 1;
  constexpr int fp4_max_biased_exp =
    half_exp_bias_signed + FP4_MAX_NORMAL_EXP_UNBIASED;

  new_exp_tie =
    (exp > fp4_tie_to_zero_biased_exp) * (exp + (mantissa_last == 1));

  // # Gather round up, round down and tie.
  new_exp =
    round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie;
  new_mantissa =
    round_away * new_mantissa_away + round_close * new_mantissa_close;

  // if new_exp > 3:
  //     new_mantissa = 1
  new_mantissa =
    new_mantissa + (new_exp > fp4_max_biased_exp) * (new_mantissa == 0);

  // Clamp the exponent to the fp4-representable range.
  new_exp = (new_exp >= fp4_tie_to_zero_biased_exp) *
            max(
              fp4_tie_to_zero_biased_exp,
              min(static_cast<int>(new_exp), fp4_max_biased_exp)
            );

  uint16_t qdq_val =
    (sign << HALF_SIGN_BIT_POS) + (new_exp << half_mantissa_bits) +
    (new_mantissa << (half_mantissa_bits - FLOAT4_MANTISSA_BITS));
  float_type result = *(float_type*)(&qdq_val);
  return result;
}

template <
  typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits,
  uint32_t half_exp_bias, uint16_t val_to_add, uint16_t sign_exponent_mask>
__global__ void qdq_mxfp4_kernel(float_type* inp, float_type* out) {
  // Each thread handles one element.

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float_type elem = inp[idx];
  float_type block_max = habs_impl(elem);

  // Compute the max across one warp via butterfly shuffles. Each thread
  // handles a single value, so we need `log2(WARP_SIZE)` rounds of shuffle
  // (5 rounds for a 32-lane warp).
  for (int i = 1; i < static_cast<int>(WARP_SIZE); i *= 2) {
    block_max =
      hmax_impl(block_max, habs_impl(shfl_xor_bf16_or_half(block_max, i)));
  }

  // TODO: fix as well in quantize kernel.
  // Apply rounding strategy to block_max.
  // cannot take the address of an rvalue so need this intermediate
  // `block_max_uint` variable?
  uint16_t block_max_uint =
    (*(uint16_t*)(&block_max) + val_to_add) & sign_exponent_mask;

  block_max = *(float_type*)(&block_max_uint);

  // Pick the largest power-of-two scale s.t. block_max / scale fits in fp4.
  // Max fp4 magnitude is 6.0 (unbiased exp = FP4_MAX_NORMAL_EXP_UNBIASED), so
  // we want the scale's unbiased exp to be `floor(log2(block_max)) -
  // FP4_MAX_NORMAL_EXP_UNBIASED`.
  uint8_t scale_exp = max(
    0, FLOAT8_E8M0_MAX_EXP +
         min(
           bf16_or_half2int_rn<float_type>(hfloor_impl(hlog2_impl(block_max))) -
             FP4_MAX_NORMAL_EXP_UNBIASED,
           FLOAT8_E8M0_MAX_EXP
         )
  );
  float_type scale = float_to_bf16_or_half<float_type>(
    powf(2.0, scale_exp - FLOAT8_E8M0_MAX_EXP)
  );

  elem = hdiv_impl(elem, scale);

  float_type elem_fp4 = fp16_to_fp4_simulate<
    float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(&elem);

  out[idx] = hmul_impl(elem_fp4, scale);
}

#endif  // USE_CUDA
