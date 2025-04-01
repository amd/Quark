//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "qdq/cuda/quantize_linear.cuh"


namespace vai_q {

// InT : [float]; T : [uint8/int8, uint16/int16, uint32/int32]
// formula is Y = X / Scale + ZeroPoint
#define QUANTIZE_LINEAR_APPLY(InT)                                                                                           \
  template <typename OutT>                                                                                                   \
  struct QuantizeLinearApply {                                                                                               \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                       \
     		    const InT* input, const InT* scale, OutT* output, const OutT* zero_point) {                              \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                  \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                 \
	  auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                    \
	  auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                   \
          auto scale_ptr      = scale + bd;                                                                                  \
	  auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                      \
          CudaQuantizeLinearStdInt<OutT, InT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
        }                                                                                                                    \
      }                                                                                                                      \
    }                                                                                                                        \
  };                                                                                                                         \

// InT : [float]; T : [float16]
// formula is Y = X / Scale + ZeroPoint
#define QUANTIZE_LINEAR_APPLY_FP16(InT)                                                                                     \
  template <typename OutT>                                                                                                  \
  struct QuantizeLinearApplyFp16 {                                                                                          \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                      \
                    const InT* input, const InT* scale, OutT* output, const OutT* zero_point) {                             \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                 \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                \
	  auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                   \
          auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                  \
          auto scale_ptr      = scale + bd;                                                                                 \
          auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                     \
          CudaQuantizeLinearStdFp<OutT, InT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
        }                                                                                                                   \
      }                                                                                                                     \
    }                                                                                                                       \
  };                                                                                                                        \

// InT : [float]; T : [bfloat16]
// formula is Y = X / Scale + ZeroPoint
#define QUANTIZE_LINEAR_APPLY_BF16(InT)                                                                                     \
  template <typename OutT>                                                                                                  \
  struct QuantizeLinearApplyBf16 {                                                                                          \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                      \
                    const InT* input, const InT* scale, OutT* output, const OutT* zero_point) {                             \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                 \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                \
          auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                   \
          auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                  \
          auto scale_ptr      = scale + bd;                                                                                 \
          auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                     \
          CudaQuantizeLinearStdBf<OutT, InT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
        }                                                                                                                   \
      }                                                                                                                     \
    }                                                                                                                       \
  };                                                                                                                        \


// T : [uint8/int8, uint16/int16, uint32/int32]; OutT : [float]
// formula is Y = (X - ZeroPoint) * Scale
#define DEQUANTIZE_LINEAR_APPLY(OutT)                                                                                          \
  template <typename InT>                                                                                                      \
  struct DequantizeLinearApply {                                                                                               \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                         \
  		    const InT* input, const OutT* scale, OutT* output, const InT* zero_point) {                                \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                    \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                   \
	  auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                      \
          auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                     \
          auto scale_ptr      = scale + bd;                                                                                    \
          auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                        \
          CudaDequantizeLinearStdInt<InT, OutT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
	}                                                                                                                      \
      }                                                                                                                        \
    }                                                                                                                          \
  };                                                                                                                           \

// T : [float16]; OutT : [float]
// formula is Y = (X - ZeroPoint) * Scale
#define DEQUANTIZE_LINEAR_APPLY_FP16(OutT)                                                                                    \
  template <typename InT>                                                                                                     \
  struct DequantizeLinearApplyFp16 {                                                                                          \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                        \
                    const InT* input, const OutT* scale, OutT* output, const InT* zero_point) {                               \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                   \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                  \
	  auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                     \
          auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                    \
          auto scale_ptr      = scale + bd;                                                                                   \
          auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                       \
          CudaDequantizeLinearStdFp<InT, OutT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
	}                                                                                                                     \
      }                                                                                                                       \
    }                                                                                                                         \
  };                                                                                                                          \

// T : [bfloat16]; OutT : [float]
// formula is Y = (X - ZeroPoint) * Scale
#define DEQUANTIZE_LINEAR_APPLY_BF16(OutT)                                                                                    \
  template <typename InT>                                                                                                     \
  struct DequantizeLinearApplyBf16 {                                                                                          \
    void op(cudaStream_t stream, int64_t N, int64_t broadcast_dim, int64_t block_size,                                        \
                    const InT* input, const OutT* scale, OutT* output, const InT* zero_point) {                               \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                   \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                  \
          auto input_ptr      = input + n * broadcast_dim * block_size + bd * block_size;                                     \
          auto output_ptr     = output + n * broadcast_dim * block_size + bd * block_size;                                    \
          auto scale_ptr      = scale + bd;                                                                                   \
          auto zero_point_ptr = zero_point ? zero_point + bd : nullptr;                                                       \
          CudaDequantizeLinearStdBf<InT, OutT>(stream, input_ptr, output_ptr, scale_ptr, zero_point_ptr, (size_t)block_size); \
        }                                                                                                                     \
      }                                                                                                                       \
    }                                                                                                                         \
  };                                                                                                                          \

}  // namespace vai_q
