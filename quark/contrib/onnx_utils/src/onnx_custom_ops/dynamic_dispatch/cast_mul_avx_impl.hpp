// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX_IMPL
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX_IMPL

#include <immintrin.h>  // AVX intrinsics

#include <cstddef>
#include <cstdint>
#include <cstring>

// MSVC compatibility: MSVC doesn't define __F16C__ even when F16C is available
// F16C is implied by AVX2 and above
#if defined(_MSC_VER) && !defined(__F16C__)
#define __F16C__ 1
#endif

namespace ryzenai::onnx_utils {

// Helper: store __m256bh to uint16_t* (MSVC compatibility)
// MSVC doesn't allow reinterpret_cast between __m256bh and __m256i
inline void store_bf16_vec(__m256bh bf16_vec, uint16_t* ptr) {
  _mm256_storeu_si256(
    reinterpret_cast<__m256i*>(ptr), *reinterpret_cast<__m256i*>(&bf16_vec)
  );
}

// AVX optimized Cast+Mul fusion implementations
namespace cast_mul_avx_impl {

// ============================================================================
// Float32 ↔ BFloat16 conversions
// ============================================================================

// Float32 -> BFloat16 with vector multiplication (AVX-512 BF16)
// in_a: FP32 input, in_b: BF16 multiplier, output: BF16
// Operation: in_a(FP32) * in_b(BF16) -> BF16
// Note: uses AVX-512 BF16 instructions
inline void float_to_bfloat16_mul_vec_avx(
  const float* in_a, const uint16_t* in_b, size_t num_elements, uint16_t* out
) {
  static_assert(sizeof(float) == 4, "Float must be 4 bytes");
  static_assert(sizeof(uint16_t) == 2, "BFloat16 must be 2 bytes");

  constexpr size_t FLOATS_PER_VECTOR = 16;  // AVX-512: 16 floats

  const size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * FLOATS_PER_VECTOR;

  // Process full vectors (FP32 * BF16 -> BF16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP32 values from in_a
    __m512 vec_a = _mm512_loadu_ps(in_a);

    // Load 16 BF16 values from in_b and convert to FP32
    __m256i bf16_vec_b =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_b));
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec_b);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 vec_b = _mm512_castsi512_ps(shifted);

    // Multiply FP32 vectors
    __m512 result = _mm512_mul_ps(vec_a, vec_b);

    // Convert result to BF16 using AVX-512 BF16 instruction
    __m256bh bf16_out = _mm512_cvtneps_pbh(result);
    store_bf16_vec(bf16_out, out);

    in_a += FLOATS_PER_VECTOR;
    in_b += FLOATS_PER_VECTOR;
    out += FLOATS_PER_VECTOR;
  }

  // Process remainder
  if (remainder > 0) {
    for (size_t i = 0; i < remainder; ++i) {
      // in_a is FP32, use directly
      float val_a = *in_a++;

      // Convert in_b from BF16 to FP32
      uint32_t bf16_bits = static_cast<uint32_t>(*in_b++) << 16;
      float val_b = *reinterpret_cast<float*>(&bf16_bits);

      // Multiply and convert result to BF16
      float result = val_a * val_b;
      uint32_t result_bits = *reinterpret_cast<const uint32_t*>(&result);
      *out++ = static_cast<uint16_t>(result_bits >> 16);
    }
  }
}

// BFloat16 -> Float32 with FP32 vector multiplication (AVX-512)
// in_a: BF16 input, in_b: FP32 multiplier, output: FP32
// Operation: in_a(BF16) * in_b(FP32) -> FP32
inline void bfloat16_to_float_mul_vec_fp32_avx(
  const uint16_t* in_a, const float* in_b, size_t num_elements, float* out
) {
  static_assert(sizeof(uint16_t) == 2, "BFloat16 must be 2 bytes");
  static_assert(sizeof(float) == 4, "Float must be 4 bytes");

  constexpr size_t ELEMENTS_PER_VECTOR = 16;  // AVX-512: 16 floats

  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Process full vectors (16 BF16 * 16 FP32 -> 16 FP32)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 BF16 values and convert to FP32
    __m256i bf16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_a));
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 vec_a = _mm512_castsi512_ps(shifted);

    // Load 16 FP32 values directly
    __m512 vec_b = _mm512_loadu_ps(in_b);

    // Multiply FP32 vectors
    __m512 result = _mm512_mul_ps(vec_a, vec_b);
    _mm512_storeu_ps(out, result);

    in_a += ELEMENTS_PER_VECTOR;
    in_b += ELEMENTS_PER_VECTOR;
    out += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    uint32_t bf16_bits = static_cast<uint32_t>(in_a[i]) << 16;
    float val_a = *reinterpret_cast<float*>(&bf16_bits);
    out[i] = val_a * in_b[i];
  }
}

// ============================================================================
// Float32 ↔ Float16 conversions (requires F16C)
// ============================================================================

#ifdef __F16C__

// Float32 -> Float16 with FP16 vector multiplication (AVX-512)
// in_a: FP32 input, in_b: FP16 multiplier, output: FP16
// Operation: in_a(FP32) * in_b(FP16) -> FP16
inline void float_to_float16_mul_vec_fp16_avx(
  const float* in_a, const uint16_t* in_b, size_t num_elements, uint16_t* out
) {
  static_assert(sizeof(float) == 4, "Float must be 4 bytes");
  static_assert(sizeof(uint16_t) == 2, "Float16 must be 2 bytes");

  constexpr size_t FLOATS_PER_VECTOR = 16;  // AVX-512: 16 floats

  const size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * FLOATS_PER_VECTOR;

  // Process full vectors (FP32 * FP16 -> FP16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP32 values from in_a
    __m512 vec_a = _mm512_loadu_ps(in_a);

    // Load 16 FP16 values from in_b and convert to FP32
    // _mm512_cvtph_ps: 16 x FP16 -> 16 x FP32
    __m256i fp16_vec_b =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_b));
    __m512 vec_b = _mm512_cvtph_ps(fp16_vec_b);

    // Multiply in FP32 domain
    __m512 result = _mm512_mul_ps(vec_a, vec_b);

    // Convert result to FP16
    // _mm512_cvtps_ph: 16 x FP32 -> 16 x FP16
    __m256i fp16_out = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), fp16_out);

    in_a += FLOATS_PER_VECTOR;
    in_b += FLOATS_PER_VECTOR;
    out += FLOATS_PER_VECTOR;
  }

  // Process remainder with scalar operations (safe, no out-of-bounds access)
  for (size_t i = 0; i < remainder; ++i) {
    // Load FP32 input
    float val_a = in_a[i];

    // Load FP16 multiplier and convert to FP32
    __m128i fp16_scalar = _mm_set1_epi16(static_cast<short>(in_b[i]));
    __m128 fp32_scalar = _mm_cvtph_ps(fp16_scalar);
    float val_b = _mm_cvtss_f32(fp32_scalar);

    // Multiply in FP32
    float result = val_a * val_b;

    // Convert result to FP16 and store
    __m128 result_vec = _mm_set_ss(result);
    __m128i fp16_result = _mm_cvtps_ph(result_vec, _MM_FROUND_TO_NEAREST_INT);
    out[i] = static_cast<uint16_t>(_mm_extract_epi16(fp16_result, 0));
  }
}

// Float16 -> Float32 with FP32 vector multiplication (AVX-512)
// in_a: FP16 input, in_b: FP32 multiplier, output: FP32
// Operation: in_a(FP16) * in_b(FP32) -> FP32
inline void float16_to_float_mul_vec_avx(
  const uint16_t* in_a, const float* in_b, size_t num_elements, float* out
) {
  static_assert(sizeof(uint16_t) == 2, "Float16 must be 2 bytes");
  static_assert(sizeof(float) == 4, "Float must be 4 bytes");

  constexpr size_t ELEMENTS_PER_VECTOR = 16;  // AVX-512: 16 floats

  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Process full vectors (FP16 * FP32 -> FP32)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP16 values and convert to FP32
    __m256i fp16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_a));
    __m512 vec_a = _mm512_cvtph_ps(fp16_vec);

    // Load 16 FP32 values directly
    __m512 vec_b = _mm512_loadu_ps(in_b);

    // Multiply FP32 vectors
    __m512 result = _mm512_mul_ps(vec_a, vec_b);
    _mm512_storeu_ps(out, result);

    in_a += ELEMENTS_PER_VECTOR;
    in_b += ELEMENTS_PER_VECTOR;
    out += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // FP16 -> FP32
    __m128i fp16_scalar = _mm_set1_epi16(static_cast<short>(in_a[i]));
    __m128 fp32_scalar = _mm_cvtph_ps(fp16_scalar);
    float val_a = _mm_cvtss_f32(fp32_scalar);

    out[i] = val_a * in_b[i];
  }
}

// ============================================================================
// Float16 ↔ BFloat16 conversions (via Float32 intermediate)
// ============================================================================

// Float16 -> BFloat16 with BF16 vector multiplication (AVX-512)
// in_a: FP16 input, in_b: BF16 multiplier, output: BF16
// Operation: in_a(FP16) * in_b(BF16) -> BF16
inline void float16_to_bfloat16_mul_vec_fp16_avx(
  const uint16_t* in_a, const uint16_t* in_b, size_t num_elements, uint16_t* out
) {
  static_assert(sizeof(uint16_t) == 2, "Float16/BFloat16 must be 2 bytes");

  constexpr size_t ELEMENTS_PER_VECTOR = 16;  // AVX-512: 16 elements

  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Process full vectors (FP16 * BF16 -> BF16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP16 values from input A and convert to FP32
    __m256i fp16_vec_a =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_a));
    __m512 vec_a = _mm512_cvtph_ps(fp16_vec_a);

    // Load 16 BF16 values from input B and convert to FP32
    __m256i bf16_vec_b =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_b));
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec_b);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 vec_b = _mm512_castsi512_ps(shifted);

    // Multiply in FP32 domain
    __m512 result = _mm512_mul_ps(vec_a, vec_b);

    // Convert result to BF16
    __m256bh bf16_out = _mm512_cvtneps_pbh(result);
    store_bf16_vec(bf16_out, out);

    in_a += ELEMENTS_PER_VECTOR;
    in_b += ELEMENTS_PER_VECTOR;
    out += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // FP16 -> FP32 for input A
    __m128i fp16_a = _mm_set1_epi16(static_cast<short>(in_a[i]));
    __m128 fp32_a = _mm_cvtph_ps(fp16_a);
    float val_a = _mm_cvtss_f32(fp32_a);

    // BF16 -> FP32 for input B
    uint32_t bf16_bits = static_cast<uint32_t>(in_b[i]) << 16;
    float val_b = *reinterpret_cast<float*>(&bf16_bits);

    // Multiply in FP32
    float result = val_a * val_b;

    // FP32 -> BF16
    uint32_t bits = *reinterpret_cast<uint32_t*>(&result);
    out[i] = static_cast<uint16_t>(bits >> 16);
  }
}

// BFloat16 -> Float16 with FP16 vector multiplication (AVX-512)
// in_a: BF16 input, in_b: FP16 multiplier, output: FP16
// Operation: in_a(BF16) * in_b(FP16) -> FP16
inline void bfloat16_to_float16_mul_vec_fp16_avx(
  const uint16_t* in_a, const uint16_t* in_b, size_t num_elements, uint16_t* out
) {
  static_assert(sizeof(uint16_t) == 2, "BFloat16/Float16 must be 2 bytes");

  constexpr size_t ELEMENTS_PER_VECTOR = 16;  // AVX-512: 16 elements

  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Process full vectors (BF16 * FP16 -> FP16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 BF16 values and convert to FP32
    __m256i bf16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_a));
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 vec_a = _mm512_castsi512_ps(shifted);

    // Load 16 FP16 values and convert to FP32
    __m256i fp16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_b));
    __m512 vec_b = _mm512_cvtph_ps(fp16_vec);

    // Multiply in FP32 domain
    __m512 result = _mm512_mul_ps(vec_a, vec_b);

    // Convert result to FP16
    __m256i fp16_out = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), fp16_out);

    in_a += ELEMENTS_PER_VECTOR;
    in_b += ELEMENTS_PER_VECTOR;
    out += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // BF16 -> FP32
    uint32_t bf16_bits = static_cast<uint32_t>(in_a[i]) << 16;
    float val_a = *reinterpret_cast<float*>(&bf16_bits);

    // FP16 -> FP32
    __m128i fp16_scalar = _mm_set1_epi16(static_cast<short>(in_b[i]));
    __m128 fp32_scalar = _mm_cvtph_ps(fp16_scalar);
    float val_b = _mm_cvtss_f32(fp32_scalar);

    // Multiply in FP32
    float result = val_a * val_b;

    // FP32 -> FP16
    __m128 result_vec = _mm_set_ss(result);
    __m128i fp16_result = _mm_cvtps_ph(result_vec, _MM_FROUND_TO_NEAREST_INT);
    out[i] = static_cast<uint16_t>(_mm_extract_epi16(fp16_result, 0));
  }
}

#endif  // __F16C__

}  // namespace cast_mul_avx_impl

// Optimized broadcast multiplication functions
namespace cast_mul_avx_broadcast {

// ============================================================================
// Optimized FP32 broadcast multiplication with casting
// ============================================================================

// Scalar broadcast: single value broadcast to all elements (most efficient)
// Operation: input(FP32) * scalar(FP32) -> BF16
inline void float_to_bfloat16_mul_scalar_broadcast_avx(
  const float* input, size_t num_elements, float scalar_value, uint16_t* output
) {
  constexpr size_t FLOATS_PER_VECTOR = 16;  // AVX-512
  const size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * FLOATS_PER_VECTOR;

  const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

  // Process full vectors
  for (size_t i = 0; i < num_iter; ++i) {
    __m512 input_vec = _mm512_loadu_ps(input);
    __m512 result = _mm512_mul_ps(input_vec, scalar_vec);

    // Convert to BF16 using AVX-512 BF16 instruction
    __m256bh bf16_vec = _mm512_cvtneps_pbh(result);
    store_bf16_vec(bf16_vec, output);

    input += FLOATS_PER_VECTOR;
    output += FLOATS_PER_VECTOR;
  }

  // Process remainder with scalar operations (safe, no out-of-bounds access)
  for (size_t i = 0; i < remainder; ++i) {
    float val = input[i] * scalar_value;
    // Convert FP32 to BF16: truncate lower 16 bits
    uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

// Scalar broadcast: FP32 -> BF16, multiplier is BF16 (target type)
inline void float_to_bfloat16_mul_scalar_broadcast_bf16_avx(
  const float* input, size_t num_elements, uint16_t scalar_bf16,
  uint16_t* output
) {
  // Convert BF16 scalar to FP32
  uint32_t bf16_bits = static_cast<uint32_t>(scalar_bf16) << 16;
  float scalar_value = *reinterpret_cast<float*>(&bf16_bits);

  // Use FP32 scalar broadcast
  float_to_bfloat16_mul_scalar_broadcast_avx(
    input, num_elements, scalar_value, output
  );
}

// Innermost dimension broadcast: [N, 1] broadcast to [N, M]
// Each outer dimension element is broadcast across the inner dimension
// input_a is float, input_b is bfloat16 (broadcast multiplier)
// Operation: input_a(FP32) * broadcast(input_b(BF16)) -> BF16
inline void float_to_bfloat16_mul_innermost_broadcast_avx(
  const float* input_a, const uint16_t* input_b, size_t outer_size,
  size_t inner_size, uint16_t* output
) {
  constexpr size_t FLOATS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Convert BF16 to FP32 for the broadcast value
    uint32_t bf16_bits = static_cast<uint32_t>(input_b[outer]) << 16;
    const float scalar_value = *reinterpret_cast<float*>(&bf16_bits);
    const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

    const float* input_ptr = input_a + outer * inner_size;
    uint16_t* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / FLOATS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * FLOATS_PER_VECTOR;

    // Vectorized inner loop
    for (size_t i = 0; i < num_iter; ++i) {
      __m512 input_vec = _mm512_loadu_ps(input_ptr);
      __m512 result = _mm512_mul_ps(input_vec, scalar_vec);

      // Convert to BF16 using AVX-512 BF16 instruction
      __m256bh bf16_vec = _mm512_cvtneps_pbh(result);
      store_bf16_vec(bf16_vec, output_ptr);

      input_ptr += FLOATS_PER_VECTOR;
      output_ptr += FLOATS_PER_VECTOR;
    }

    // Remainder with scalar operations (safe, no out-of-bounds access)
    for (size_t i = 0; i < remainder; ++i) {
      float val = input_ptr[i] * scalar_value;
      // Convert FP32 to BF16: truncate lower 16 bits
      uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
      output_ptr[i] = static_cast<uint16_t>(bits >> 16);
    }
  }
}

// ============================================================================
// BFloat16 -> Float32 broadcast multiplication
// ============================================================================

// Scalar broadcast: BF16 * scalar(FP32) -> FP32
// Operation: input(BF16) * scalar(FP32) -> FP32
inline void bfloat16_to_float_mul_scalar_broadcast_avx(
  const uint16_t* input, size_t num_elements, float scalar_value, float* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;  // AVX-512: 16 floats
  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

  // Process full vectors (16 BF16 -> 16 FP32, then multiply)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 BF16 values
    __m256i bf16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));

    // Convert BF16 to FP32: zero-extend to 32-bit, then shift left by 16
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 float_vec = _mm512_castsi512_ps(shifted);

    // Multiply with scalar
    __m512 result = _mm512_mul_ps(float_vec, scalar_vec);
    _mm512_storeu_ps(output, result);

    input += ELEMENTS_PER_VECTOR;
    output += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    uint32_t bf16_bits = static_cast<uint32_t>(input[i]) << 16;
    float val = *reinterpret_cast<float*>(&bf16_bits);
    output[i] = val * scalar_value;
  }
}

// Innermost dimension broadcast: BF16 input * FP32 broadcast multiplier -> FP32
// Shape: input[N, M](BF16) * multiplier[N, 1](FP32) -> output[N, M](FP32)
// Each multiplier element is broadcast across the inner dimension
inline void bfloat16_to_float_mul_innermost_broadcast_avx(
  const uint16_t* input, const float* multiplier, size_t outer_size,
  size_t inner_size, float* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Get the broadcast multiplier value for this outer index
    const float scalar_value = multiplier[outer];
    const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

    const uint16_t* input_ptr = input + outer * inner_size;
    float* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / ELEMENTS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * ELEMENTS_PER_VECTOR;

    // Vectorized inner loop
    for (size_t i = 0; i < num_iter; ++i) {
      // Load 16 BF16 values
      __m256i bf16_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr));

      // Convert BF16 to FP32
      __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
      __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
      __m512 float_vec = _mm512_castsi512_ps(shifted);

      // Multiply with broadcast scalar
      __m512 result = _mm512_mul_ps(float_vec, scalar_vec);
      _mm512_storeu_ps(output_ptr, result);

      input_ptr += ELEMENTS_PER_VECTOR;
      output_ptr += ELEMENTS_PER_VECTOR;
    }

    // Remainder with scalar operations (safe, no out-of-bounds access)
    for (size_t i = 0; i < remainder; ++i) {
      uint32_t bf16_bits = static_cast<uint32_t>(input_ptr[i]) << 16;
      float val = *reinterpret_cast<float*>(&bf16_bits);
      output_ptr[i] = val * scalar_value;
    }
  }
}

#ifdef __F16C__

// ============================================================================
// BFloat16 -> Float16 broadcast multiplication
// ============================================================================

// Scalar broadcast: BF16 * scalar(FP16) -> FP16
// Operation: input(BF16) * scalar(FP16) -> FP16
inline void bfloat16_to_float16_mul_scalar_broadcast_fp16_avx(
  const uint16_t* input, size_t num_elements, uint16_t scalar_fp16,
  uint16_t* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;
  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Convert FP16 scalar to FP32
  __m128i fp16_scalar_vec = _mm_set1_epi16(static_cast<short>(scalar_fp16));
  __m128 fp32_scalar_128 = _mm_cvtph_ps(fp16_scalar_vec);
  float scalar_value = _mm_cvtss_f32(fp32_scalar_128);
  const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

  // Process full vectors (BF16 -> FP32 -> multiply -> FP16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 BF16 values and convert to FP32
    __m256i bf16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
    __m512 float_vec = _mm512_castsi512_ps(shifted);

    // Multiply with scalar
    __m512 result = _mm512_mul_ps(float_vec, scalar_vec);

    // Convert to FP16
    __m256i fp16_out = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), fp16_out);

    input += ELEMENTS_PER_VECTOR;
    output += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // BF16 -> FP32
    uint32_t bf16_bits = static_cast<uint32_t>(input[i]) << 16;
    float val = *reinterpret_cast<float*>(&bf16_bits);

    // Multiply
    float result = val * scalar_value;

    // FP32 -> FP16
    __m128 result_vec = _mm_set_ss(result);
    __m128i fp16_result = _mm_cvtps_ph(result_vec, _MM_FROUND_TO_NEAREST_INT);
    output[i] = static_cast<uint16_t>(_mm_extract_epi16(fp16_result, 0));
  }
}

// Innermost dimension broadcast: BF16 input * FP16 broadcast multiplier -> FP16
// Shape: input[N, M](BF16) * multiplier[N, 1](FP16) -> output[N, M](FP16)
inline void bfloat16_to_float16_mul_innermost_broadcast_fp16_avx(
  const uint16_t* input, const uint16_t* multiplier, size_t outer_size,
  size_t inner_size, uint16_t* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Convert FP16 multiplier to FP32
    __m128i fp16_scalar_vec =
      _mm_set1_epi16(static_cast<short>(multiplier[outer]));
    __m128 fp32_scalar_128 = _mm_cvtph_ps(fp16_scalar_vec);
    float scalar_value = _mm_cvtss_f32(fp32_scalar_128);
    const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

    const uint16_t* input_ptr = input + outer * inner_size;
    uint16_t* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / ELEMENTS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * ELEMENTS_PER_VECTOR;

    // Vectorized inner loop
    for (size_t i = 0; i < num_iter; ++i) {
      // Load 16 BF16 values and convert to FP32
      __m256i bf16_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr));
      __m512i bf16_32bit = _mm512_cvtepu16_epi32(bf16_vec);
      __m512i shifted = _mm512_slli_epi32(bf16_32bit, 16);
      __m512 float_vec = _mm512_castsi512_ps(shifted);

      // Multiply with broadcast scalar
      __m512 result = _mm512_mul_ps(float_vec, scalar_vec);

      // Convert to FP16
      __m256i fp16_out = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_ptr), fp16_out);

      input_ptr += ELEMENTS_PER_VECTOR;
      output_ptr += ELEMENTS_PER_VECTOR;
    }

    // Remainder with scalar operations
    for (size_t i = 0; i < remainder; ++i) {
      // BF16 -> FP32
      uint32_t bf16_bits = static_cast<uint32_t>(input_ptr[i]) << 16;
      float val = *reinterpret_cast<float*>(&bf16_bits);

      // Multiply
      float result = val * scalar_value;

      // FP32 -> FP16
      __m128 result_vec = _mm_set_ss(result);
      __m128i fp16_result = _mm_cvtps_ph(result_vec, _MM_FROUND_TO_NEAREST_INT);
      output_ptr[i] = static_cast<uint16_t>(_mm_extract_epi16(fp16_result, 0));
    }
  }
}

// ============================================================================
// Float16 -> Float32 broadcast multiplication
// ============================================================================

// Scalar broadcast: FP16 * scalar(FP16) -> FP32
// Operation: input(FP16) * scalar(FP16) -> FP32
inline void float16_to_float_mul_scalar_broadcast_avx(
  const uint16_t* input, size_t num_elements, float scalar_fp32, float* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;
  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Convert FP16 scalar (stored as uint16) to FP32
  const __m512 scalar_vec = _mm512_set1_ps(scalar_fp32);

  // Process full vectors (FP16 -> FP32 -> multiply)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP16 values and convert to FP32
    __m256i fp16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    __m512 float_vec = _mm512_cvtph_ps(fp16_vec);

    // Multiply with scalar
    __m512 result = _mm512_mul_ps(float_vec, scalar_vec);
    _mm512_storeu_ps(output, result);

    input += ELEMENTS_PER_VECTOR;
    output += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // FP16 -> FP32
    __m128i fp16_val = _mm_set1_epi16(static_cast<short>(input[i]));
    __m128 fp32_val = _mm_cvtph_ps(fp16_val);
    float val = _mm_cvtss_f32(fp32_val);

    output[i] = val * scalar_fp32;
  }
}

// Innermost dimension broadcast: FP16 input * FP16 broadcast multiplier -> FP32
// Shape: input[N, M](FP16) * multiplier[N, 1](FP16) -> output[N, M](FP32)
inline void float16_to_float_mul_innermost_broadcast_avx(
  const uint16_t* input, const float* multiplier, size_t outer_size,
  size_t inner_size, float* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Convert FP16 multiplier to FP32
    const __m512 scalar_vec = _mm512_set1_ps(multiplier[outer]);

    const uint16_t* input_ptr = input + outer * inner_size;
    float* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / ELEMENTS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * ELEMENTS_PER_VECTOR;

    // Vectorized inner loop
    for (size_t i = 0; i < num_iter; ++i) {
      // Load 16 FP16 values and convert to FP32
      __m256i fp16_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr));
      __m512 float_vec = _mm512_cvtph_ps(fp16_vec);

      // Multiply with broadcast scalar
      __m512 result = _mm512_mul_ps(float_vec, scalar_vec);
      _mm512_storeu_ps(output_ptr, result);

      input_ptr += ELEMENTS_PER_VECTOR;
      output_ptr += ELEMENTS_PER_VECTOR;
    }

    // Remainder with scalar operations
    for (size_t i = 0; i < remainder; ++i) {
      // FP16 -> FP32
      __m128i fp16_val = _mm_set1_epi16(static_cast<short>(input_ptr[i]));
      __m128 fp32_val = _mm_cvtph_ps(fp16_val);
      float val = _mm_cvtss_f32(fp32_val);

      output_ptr[i] = val * multiplier[outer];
    }
  }
}

// ============================================================================
// Float16 -> BFloat16 broadcast multiplication
// ============================================================================

// Scalar broadcast: FP16 * scalar(BF16) -> BF16
// Operation: input(FP16) * scalar(BF16) -> BF16
inline void float16_to_bfloat16_mul_scalar_broadcast_avx(
  const uint16_t* input, size_t num_elements, uint16_t scalar_bf16,
  uint16_t* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;
  const size_t num_iter = num_elements / ELEMENTS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * ELEMENTS_PER_VECTOR;

  // Convert BF16 scalar to FP32
  uint32_t bf16_bits = static_cast<uint32_t>(scalar_bf16) << 16;
  float scalar_value = *reinterpret_cast<float*>(&bf16_bits);
  const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

  // Process full vectors (FP16 -> FP32 -> multiply -> BF16)
  for (size_t i = 0; i < num_iter; ++i) {
    // Load 16 FP16 values and convert to FP32
    __m256i fp16_vec =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    __m512 float_vec = _mm512_cvtph_ps(fp16_vec);

    // Multiply with scalar
    __m512 result = _mm512_mul_ps(float_vec, scalar_vec);

    // Convert to BF16
    __m256bh bf16_out = _mm512_cvtneps_pbh(result);
    store_bf16_vec(bf16_out, output);

    input += ELEMENTS_PER_VECTOR;
    output += ELEMENTS_PER_VECTOR;
  }

  // Process remainder with scalar operations
  for (size_t i = 0; i < remainder; ++i) {
    // FP16 -> FP32
    __m128i fp16_val = _mm_set1_epi16(static_cast<short>(input[i]));
    __m128 fp32_val = _mm_cvtph_ps(fp16_val);
    float val = _mm_cvtss_f32(fp32_val);

    // Multiply
    float result = val * scalar_value;

    // FP32 -> BF16
    uint32_t bits = *reinterpret_cast<uint32_t*>(&result);
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

// Innermost dimension broadcast: FP16 input * BF16 broadcast multiplier -> BF16
// Shape: input[N, M](FP16) * multiplier[N, 1](BF16) -> output[N, M](BF16)
inline void float16_to_bfloat16_mul_innermost_broadcast_avx(
  const uint16_t* input, const uint16_t* multiplier, size_t outer_size,
  size_t inner_size, uint16_t* output
) {
  constexpr size_t ELEMENTS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Convert BF16 multiplier to FP32
    uint32_t bf16_bits = static_cast<uint32_t>(multiplier[outer]) << 16;
    float scalar_value = *reinterpret_cast<float*>(&bf16_bits);
    const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

    const uint16_t* input_ptr = input + outer * inner_size;
    uint16_t* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / ELEMENTS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * ELEMENTS_PER_VECTOR;

    // Vectorized inner loop
    for (size_t i = 0; i < num_iter; ++i) {
      // Load 16 FP16 values and convert to FP32
      __m256i fp16_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_ptr));
      __m512 float_vec = _mm512_cvtph_ps(fp16_vec);

      // Multiply with broadcast scalar
      __m512 result = _mm512_mul_ps(float_vec, scalar_vec);

      // Convert to BF16
      __m256bh bf16_out = _mm512_cvtneps_pbh(result);
      store_bf16_vec(bf16_out, output_ptr);

      input_ptr += ELEMENTS_PER_VECTOR;
      output_ptr += ELEMENTS_PER_VECTOR;
    }

    // Remainder with scalar operations
    for (size_t i = 0; i < remainder; ++i) {
      // FP16 -> FP32
      __m128i fp16_val = _mm_set1_epi16(static_cast<short>(input_ptr[i]));
      __m128 fp32_val = _mm_cvtph_ps(fp16_val);
      float val = _mm_cvtss_f32(fp32_val);

      // Multiply
      float result = val * scalar_value;

      // FP32 -> BF16
      uint32_t bits = *reinterpret_cast<uint32_t*>(&result);
      output_ptr[i] = static_cast<uint16_t>(bits >> 16);
    }
  }
}

// ============================================================================
// Float32 -> Float16 broadcast multiplication
// ============================================================================

// Scalar broadcast: FP32 -> FP16 with scalar broadcast
inline void float_to_float16_mul_scalar_broadcast_avx(
  const float* input, size_t num_elements, float scalar_value, uint16_t* output
) {
  constexpr size_t FLOATS_PER_VECTOR = 16;
  const size_t num_iter = num_elements / FLOATS_PER_VECTOR;
  const size_t remainder = num_elements - num_iter * FLOATS_PER_VECTOR;

  const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

  for (size_t i = 0; i < num_iter; ++i) {
    __m512 input_vec = _mm512_loadu_ps(input);
    __m512 result = _mm512_mul_ps(input_vec, scalar_vec);
    __m256i fp16_vec = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), fp16_vec);

    input += FLOATS_PER_VECTOR;
    output += FLOATS_PER_VECTOR;
  }

  for (size_t i = 0; i < remainder; ++i) {
    float val = (*input++) * scalar_value;
    __m128 temp = _mm_set_ss(val);
    __m128i fp16 = _mm_cvtps_ph(temp, _MM_FROUND_TO_NEAREST_INT);
    *output++ = static_cast<uint16_t>(_mm_extract_epi16(fp16, 0));
  }
}

// Scalar broadcast: FP32 -> FP16, multiplier is FP16 (target type)
inline void float_to_float16_mul_scalar_broadcast_fp16_avx(
  const float* input, size_t num_elements, uint16_t scalar_fp16,
  uint16_t* output
) {
  // Convert FP16 scalar to FP32
  __m128i fp16_scalar = _mm_set1_epi16(scalar_fp16);
  __m128 fp32_scalar_128 = _mm_cvtph_ps(fp16_scalar);
  float scalar_value = _mm_cvtss_f32(fp32_scalar_128);

  // Use FP32 scalar broadcast
  float_to_float16_mul_scalar_broadcast_avx(
    input, num_elements, scalar_value, output
  );
}

// Innermost dimension broadcast: FP32 -> FP16, multiplier is FP16
inline void float_to_float16_mul_innermost_broadcast_fp16_avx(
  const float* input_a, const uint16_t* input_b, size_t outer_size,
  size_t inner_size, uint16_t* output
) {
  constexpr size_t FLOATS_PER_VECTOR = 16;

  for (size_t outer = 0; outer < outer_size; ++outer) {
    // Convert FP16 to FP32 for the broadcast value
    __m128i fp16_val = _mm_set1_epi16(input_b[outer]);
    __m128 fp32_val = _mm_cvtph_ps(fp16_val);
    const float scalar_value = _mm_cvtss_f32(fp32_val);
    const __m512 scalar_vec = _mm512_set1_ps(scalar_value);

    const float* input_ptr = input_a + outer * inner_size;
    uint16_t* output_ptr = output + outer * inner_size;

    const size_t num_iter = inner_size / FLOATS_PER_VECTOR;
    const size_t remainder = inner_size - num_iter * FLOATS_PER_VECTOR;

    for (size_t i = 0; i < num_iter; ++i) {
      __m512 input_vec = _mm512_loadu_ps(input_ptr);
      __m512 result = _mm512_mul_ps(input_vec, scalar_vec);
      __m256i fp16_vec = _mm512_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_ptr), fp16_vec);

      input_ptr += FLOATS_PER_VECTOR;
      output_ptr += FLOATS_PER_VECTOR;
    }

    for (size_t i = 0; i < remainder; ++i) {
      float val = (*input_ptr++) * scalar_value;
      __m128 temp = _mm_set_ss(val);
      __m128i fp16 = _mm_cvtps_ph(temp, _MM_FROUND_TO_NEAREST_INT);
      *output_ptr++ = static_cast<uint16_t>(_mm_extract_epi16(fp16, 0));
    }
  }
}

#endif  // __F16C__

}  // namespace cast_mul_avx_broadcast

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX_IMPL
