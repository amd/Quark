// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX

#include <immintrin.h>  // AVX intrinsics

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "dynamic_dispatch.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "operator.hpp"

#if __has_include("ops/matmul_noqdq/bfp.hpp")
#include "ops/matmul_noqdq/bfp.hpp"
#ifndef DD_COMPUTEX
#define DD_COMPUTEX
#endif  // DD_COMPUTEX
#else
#include "ops/ops_common/dtype_utils.h"
#endif  // __has_include("ops/matmul_noqdq/bfp.hpp")

// Include AVX implementation (must be before namespace to avoid nesting)
#include "cast_mul_avx_impl.hpp"

namespace ryzenai::onnx_utils {

// Broadcasting and shape utilities
namespace broadcast_utils {

// Broadcast pattern types for optimization
enum class BroadcastPattern {
  NONE,           // No broadcasting needed (shapes match)
  SCALAR,         // Broadcast from scalar (all dims are 1)
  INNERMOST_DIM,  // Last dimension is broadcast (e.g., [N,1] -> [N,M])
  GENERAL         // General broadcasting pattern
};

// Compute output shape and strides for broadcasting
struct BroadcastInfo {
  std::vector<int64_t> output_shape;
  std::vector<int64_t> a_strides;
  std::vector<int64_t> b_strides;
  size_t output_size;
  BroadcastPattern a_pattern;
  BroadcastPattern b_pattern;
  size_t
    innermost_dim_size;  // Size of the innermost dimension (for vectorization)
};

// Detect broadcast pattern for optimization
inline BroadcastPattern get_broadcast_pattern(
  const std::vector<int64_t>& shape, const std::vector<int64_t>& output_shape
) {
  if (shape == output_shape) {
    return BroadcastPattern::NONE;
  }

  // Check if it's a scalar broadcast (all dimensions are 1)
  bool is_scalar = true;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1) {
      is_scalar = false;
      break;
    }
  }
  if (is_scalar) {
    return BroadcastPattern::SCALAR;
  }

  // Check if only the innermost (last) dimension is broadcast
  if (shape.size() > 0 && shape.back() == 1 && output_shape.back() > 1) {
    bool other_dims_match = true;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      if (shape[i] != output_shape[i]) {
        other_dims_match = false;
        break;
      }
    }
    if (other_dims_match) {
      return BroadcastPattern::INNERMOST_DIM;
    }
  }

  return BroadcastPattern::GENERAL;
}

inline BroadcastInfo compute_broadcast_info(
  const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b
) {
  BroadcastInfo info;

  size_t rank = shape_a.size();
  info.output_shape.resize(rank);
  info.a_strides.resize(rank);
  info.b_strides.resize(rank);

  // First, compute output shape and check compatibility
  if (shape_a.size() == 1) {
    info.output_shape = shape_b;
  } else if (shape_b.size() == 1) {
    info.output_shape = shape_a;
  } else {
    for (size_t i = 0; i < rank; ++i) {
      if (shape_a[i] == shape_b[i]) {
        info.output_shape[i] = shape_a[i];
      } else if (shape_a[i] == 1) {
        info.output_shape[i] = shape_b[i];
      } else if (shape_b[i] == 1) {
        info.output_shape[i] = shape_a[i];
      } else {
        throw std::runtime_error(
          "Broadcasting: incompatible shapes at dimension " +
          std::to_string(i) + ". Got " + std::to_string(shape_a[i]) + " and " +
          std::to_string(shape_b[i])
        );
      }
    }
  }

  // Now detect broadcast patterns (after output_shape is computed)
  info.a_pattern = get_broadcast_pattern(shape_a, info.output_shape);
  info.b_pattern = get_broadcast_pattern(shape_b, info.output_shape);

  // Compute strides for broadcasting (from right to left)
  int64_t stride_a = 1, stride_b = 1;
  for (int i = rank - 1; i >= 0; --i) {
    info.a_strides[i] = (shape_a[i] == 1) ? 0 : stride_a;
    info.b_strides[i] = (shape_b[i] == 1) ? 0 : stride_b;

    stride_a *= shape_a[i];
    stride_b *= shape_b[i];
  }

  // Compute output size
  info.output_size = std::accumulate(
    info.output_shape.begin(), info.output_shape.end(), 1LL,
    std::multiplies<int64_t>()
  );

  // Get innermost dimension size for vectorization
  info.innermost_dim_size = rank > 0 ? info.output_shape.back() : 1;

  return info;
}

// Compute linear index from multi-dimensional indices
inline size_t compute_offset(
  const std::vector<int64_t>& indices, const std::vector<int64_t>& strides
) {
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

// Convert flat index to multi-dimensional indices
inline std::vector<int64_t> index_to_coords(
  size_t index, const std::vector<int64_t>& shape
) {
  std::vector<int64_t> coords(shape.size());
  for (int i = shape.size() - 1; i >= 0; --i) {
    coords[i] = index % shape[i];
    index /= shape[i];
  }
  return coords;
}

}  // namespace broadcast_utils

/**
 * CastMulAvx Kernel: Fused Cast + Multiply operation with AVX optimization
 *
 * Supported conversions (all with multiplication and broadcasting):
 *   - Float32 ↔ BFloat16
 *   - Float32 ↔ Float16
 *   - Float16 ↔ BFloat16
 *
 * Attributes:
 *   - to: Target data type (int64)
 *
 * Inputs:
 *   - input[0]: Tensor to be cast (does not support broadcasting)
 *   - input[1]: Multiplier tensor (supports broadcasting, can be a scalar)
 */
struct CastMulAvxKernel : ExecutionProviderExtensions {
  CastMulAvxKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  )
    : ort_(ort_api) {
    const auto info_obj = Ort::ConstKernelInfo(info);
    to_ = static_cast<ONNXTensorElementDataType>(
      info_obj.GetAttribute<int64_t>("to")
    );
  }

  // cast(input[0]) * input[1] operation
  void Compute(OrtKernelContext* context) {
    auto ctx = Ort::KernelContext(context);

    const auto input_tensor = ctx.GetInput(0);
    const auto mul_tensor = ctx.GetInput(1);

    const auto input_dtype =
      input_tensor.GetTensorTypeAndShapeInfo().GetElementType();
    auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const size_t input_elements =
      input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    auto mul_shape = mul_tensor.GetTensorTypeAndShapeInfo().GetShape();

    // Data pointers and broadcast buffers
    const auto* input_data = input_tensor.GetTensorRawData();
    const void* input_data_ptr = input_data;
    const auto* mul_data = mul_tensor.GetTensorRawData();
    const void* mul_data_ptr = mul_data;

    std::vector<uint8_t> input_data_expanded_raw;  // Raw buffer for any type
    std::vector<uint8_t> mul_data_expanded_raw;    // Raw buffer for any type

    // Broadcast information
    broadcast_utils::BroadcastInfo broadcast_info;
    bool use_optimized_broadcast = false;

    // Output shape defaults to input shape, may be updated by broadcasting
    auto output_shape = input_shape;
    size_t output_elements = input_elements;

    if (input_shape != mul_shape) {
      // Compute broadcast information
      broadcast_info =
        broadcast_utils::compute_broadcast_info(input_shape, mul_shape);
      output_shape = broadcast_info.output_shape;
      output_elements = broadcast_info.output_size;

      // Check if we can use optimized broadcast path
      use_optimized_broadcast =
        (broadcast_info.b_pattern ==
           broadcast_utils::BroadcastPattern::SCALAR ||
         broadcast_info.b_pattern ==
           broadcast_utils::BroadcastPattern::INNERMOST_DIM) &&
        (broadcast_info.a_pattern == broadcast_utils::BroadcastPattern::NONE);

      if (!use_optimized_broadcast) {
        // Get element size based on target type (mul_data type == to_)
        size_t target_element_size = 0;
        switch (to_) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            target_element_size = sizeof(float);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            target_element_size = sizeof(uint16_t);
            break;
          default:
            throw std::invalid_argument(
              "Unsupported target data type for broadcasting"
            );
        }

        // Get element size based on input data type
        size_t input_element_size = 0;
        switch (input_dtype) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            input_element_size = sizeof(float);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            input_element_size = sizeof(uint16_t);
            break;
          default:
            throw std::invalid_argument(
              "Unsupported input data type for broadcasting"
            );
        }

        // Expand multiplier if needed (mul_data is target type)
        bool mul_needs_broadcast = (output_shape != mul_shape);
        if (mul_needs_broadcast) {
          mul_data_expanded_raw.resize(output_elements * target_element_size);

          if (to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* mul_fp32 = reinterpret_cast<const float*>(mul_data);
            float* expanded_fp32 =
              reinterpret_cast<float*>(mul_data_expanded_raw.data());
            for (size_t i = 0; i < output_elements; ++i) {
              auto coords = broadcast_utils::index_to_coords(i, output_shape);
              size_t mul_idx = broadcast_utils::compute_offset(
                coords, broadcast_info.b_strides
              );
              expanded_fp32[i] = mul_fp32[mul_idx];
            }
          } else {
            // FP16 or BF16
            const uint16_t* mul_fp16 =
              reinterpret_cast<const uint16_t*>(mul_data);
            uint16_t* expanded_fp16 =
              reinterpret_cast<uint16_t*>(mul_data_expanded_raw.data());
            for (size_t i = 0; i < output_elements; ++i) {
              auto coords = broadcast_utils::index_to_coords(i, output_shape);
              size_t mul_idx = broadcast_utils::compute_offset(
                coords, broadcast_info.b_strides
              );
              expanded_fp16[i] = mul_fp16[mul_idx];
            }
          }
          mul_data_ptr = mul_data_expanded_raw.data();
        }

        // Expand first input if needed
        bool input_needs_broadcast = (output_shape != input_shape);
        if (input_needs_broadcast) {
          input_data_expanded_raw.resize(output_elements * input_element_size);

          if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* input_fp32 =
              reinterpret_cast<const float*>(input_data);
            float* expanded_fp32 =
              reinterpret_cast<float*>(input_data_expanded_raw.data());
            for (size_t i = 0; i < output_elements; ++i) {
              auto coords = broadcast_utils::index_to_coords(i, output_shape);
              size_t input_idx = broadcast_utils::compute_offset(
                coords, broadcast_info.a_strides
              );
              expanded_fp32[i] = input_fp32[input_idx];
            }
          } else {
            // FP16 or BF16
            const uint16_t* input_fp16 =
              reinterpret_cast<const uint16_t*>(input_data);
            uint16_t* expanded_fp16 =
              reinterpret_cast<uint16_t*>(input_data_expanded_raw.data());
            for (size_t i = 0; i < output_elements; ++i) {
              auto coords = broadcast_utils::index_to_coords(i, output_shape);
              size_t input_idx = broadcast_utils::compute_offset(
                coords, broadcast_info.a_strides
              );
              expanded_fp16[i] = input_fp16[input_idx];
            }
          }
          input_data_ptr = input_data_expanded_raw.data();
        }
      }
    }

    // Create output tensor with computed shape
    auto output_tensor = ctx.GetOutput(0, output_shape);
    auto* output_data = output_tensor.GetTensorMutableRawData();

    // ========================================================================
    // Float32 conversions
    // ========================================================================

    if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      // Float32 -> BFloat16
      RecordDuration(Metric::Casting, [&]() {
        const float* fp32_input =
          reinterpret_cast<const float*>(input_data_ptr);
        const uint16_t* bf16_mul =
          reinterpret_cast<const uint16_t*>(mul_data_ptr);

        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::
              float_to_bfloat16_mul_scalar_broadcast_bf16_avx(
                fp32_input, output_elements, bf16_mul[0],
                reinterpret_cast<uint16_t*>(output_data)
              );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              float_to_bfloat16_mul_innermost_broadcast_avx(
                fp32_input, bf16_mul, outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<uint16_t*>(output_data)
              );
          }
        } else {
          // General case: cast input to BF16, multiply with BF16, output BF16
          cast_mul_avx_impl::float_to_bfloat16_mul_vec_avx(
            fp32_input, bf16_mul, output_elements,
            reinterpret_cast<uint16_t*>(output_data)
          );
        }
      });

    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      // Float32 -> Float16
      RecordDuration(Metric::Casting, [&]() {
        const float* fp32_input =
          reinterpret_cast<const float*>(input_data_ptr);
        const uint16_t* fp16_mul =
          reinterpret_cast<const uint16_t*>(mul_data_ptr);

        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::
              float_to_float16_mul_scalar_broadcast_fp16_avx(
                fp32_input, output_elements, fp16_mul[0],
                reinterpret_cast<uint16_t*>(output_data)
              );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              float_to_float16_mul_innermost_broadcast_fp16_avx(
                fp32_input, fp16_mul, outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<uint16_t*>(output_data)
              );
          }
        } else {
          // General case: FP32 * FP16 -> FP16 (only 2 conversions!)
          cast_mul_avx_impl::float_to_float16_mul_vec_fp16_avx(
            fp32_input, fp16_mul, output_elements,
            reinterpret_cast<uint16_t*>(output_data)
          );
        }
      });

    }

    // ========================================================================
    // BFloat16 conversions
    // ========================================================================

    else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
             to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      // BFloat16 -> Float32
      RecordDuration(Metric::Casting, [&]() {
        // input_data is bfloat16, mul_data_ptr is float
        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::bfloat16_to_float_mul_scalar_broadcast_avx(
              reinterpret_cast<const uint16_t*>(input_data_ptr),
              output_elements, reinterpret_cast<const float*>(mul_data_ptr)[0],
              reinterpret_cast<float*>(output_data)
            );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              bfloat16_to_float_mul_innermost_broadcast_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                reinterpret_cast<const float*>(mul_data_ptr), outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<float*>(output_data)
              );
          }
        } else {
          // General case: BFloat16 * Float32 -> Float32
          cast_mul_avx_impl::bfloat16_to_float_mul_vec_fp32_avx(
            reinterpret_cast<const uint16_t*>(input_data_ptr),
            reinterpret_cast<const float*>(mul_data_ptr), output_elements,
            reinterpret_cast<float*>(output_data)
          );
        }
      });

    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      // BFloat16 -> Float16
      RecordDuration(Metric::Casting, [&]() {
        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::
              bfloat16_to_float16_mul_scalar_broadcast_fp16_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                output_elements,
                reinterpret_cast<const uint16_t*>(mul_data_ptr)[0],
                reinterpret_cast<uint16_t*>(output_data)
              );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              bfloat16_to_float16_mul_innermost_broadcast_fp16_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                reinterpret_cast<const uint16_t*>(mul_data_ptr), outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<uint16_t*>(output_data)
              );
          }
        } else {
          // General case: BFloat16 * Float16 -> Float16
          cast_mul_avx_impl::bfloat16_to_float16_mul_vec_fp16_avx(
            reinterpret_cast<const uint16_t*>(input_data_ptr),
            reinterpret_cast<const uint16_t*>(mul_data_ptr), output_elements,
            reinterpret_cast<uint16_t*>(output_data)
          );
        }
      });

      // ========================================================================
      // Float16 conversions
      // ========================================================================

    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      // Float16 -> Float32
      RecordDuration(Metric::Casting, [&]() {
        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::float16_to_float_mul_scalar_broadcast_avx(
              reinterpret_cast<const uint16_t*>(input_data_ptr),
              output_elements, reinterpret_cast<const float*>(mul_data_ptr)[0],
              reinterpret_cast<float*>(output_data)
            );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              float16_to_float_mul_innermost_broadcast_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                reinterpret_cast<const float*>(mul_data_ptr), outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<float*>(output_data)
              );
          }
        } else {
          // General case: Float16 * Float32 -> Float32
          cast_mul_avx_impl::float16_to_float_mul_vec_avx(
            reinterpret_cast<const uint16_t*>(input_data_ptr),
            reinterpret_cast<const float*>(mul_data_ptr), output_elements,
            reinterpret_cast<float*>(output_data)
          );
        }
      });

    } else if (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
               to_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      // Float16 -> BFloat16
      RecordDuration(Metric::Casting, [&]() {
        if (use_optimized_broadcast) {
          if (broadcast_info.b_pattern ==
              broadcast_utils::BroadcastPattern::SCALAR) {
            cast_mul_avx_broadcast::
              float16_to_bfloat16_mul_scalar_broadcast_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                output_elements,
                reinterpret_cast<const uint16_t*>(mul_data_ptr)[0],
                reinterpret_cast<uint16_t*>(output_data)
              );
          } else if (broadcast_info.b_pattern ==
                     broadcast_utils::BroadcastPattern::INNERMOST_DIM) {
            size_t outer_size =
              output_elements / broadcast_info.innermost_dim_size;
            cast_mul_avx_broadcast::
              float16_to_bfloat16_mul_innermost_broadcast_avx(
                reinterpret_cast<const uint16_t*>(input_data_ptr),
                reinterpret_cast<const uint16_t*>(mul_data_ptr), outer_size,
                broadcast_info.innermost_dim_size,
                reinterpret_cast<uint16_t*>(output_data)
              );
          }
        } else {
          // General case: Float16 * FP16 -> BFloat16
          cast_mul_avx_impl::float16_to_bfloat16_mul_vec_fp16_avx(
            reinterpret_cast<const uint16_t*>(input_data_ptr),
            reinterpret_cast<const uint16_t*>(mul_data_ptr), output_elements,
            reinterpret_cast<uint16_t*>(output_data)
          );
        }
      });

    } else {
      throw std::invalid_argument(
        "Unsupported CastMulAvx conversion from " +
        std::string(type_to_str(input_dtype)) + " to " +
        std::string(type_to_str(to_))
      );
    }
  }

 private:
  OrtApi ort_{};
  ONNXTensorElementDataType to_;
  float scalar_;
};

// Define the custom operator
static const char kCastMulAvx[] = "CastMulAvx";
struct CastMulAvx : Operator<CastMulAvxKernel, kCastMulAvx> {
  using Operator::Operator;

  size_t GetInputTypeCount() const noexcept override { return 2; }
  size_t GetOutputTypeCount() const noexcept override { return 1; }

  // Override to mark inputs as required (not variadic)
  // ONNX Runtime requires only the last input can be variadic
  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /* index */) const noexcept override {
    return INPUT_OUTPUT_REQUIRED;
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_CAST_MUL_AVX
