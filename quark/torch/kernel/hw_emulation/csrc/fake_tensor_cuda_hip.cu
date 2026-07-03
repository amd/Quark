//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable ABI version using raw CUDA kernels instead of ATen internal APIs,
// so the extension can be pre-compiled and shipped without user-side build.
//

#ifdef USE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cmath>
#include <stdexcept>

#include "device_guard.h"
#include "gpu_stream.h"

namespace quark {
namespace hw_emulation {

__device__ __forceinline__ float to_float(__half val) {
  return __half2float(val);
}
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ __half from_float(__half, float val) {
  return __float2half(val);
}
__device__ __forceinline__ float from_float(float, float val) { return val; }

// Quantize-then-dequantize, bit-exact against the legacy pybind11 kernel.
// Two precision details matter for that bit-exactness:
//   1. Mode 2 uses ``+ 0.5`` (double) + ``std::floor`` so the sum is
//      promoted to double before flooring; ``+ 0.5f`` / ``floorf`` would
//      disagree near tie boundaries.
//   2. Saturation clamps in float, not int64 — int64 clamping would diverge
//      for inputs overflowing float's 24-bit integer range.
template <typename T>
__device__ __forceinline__ T fake_quantize_apply(
  T input_raw, T scale_raw, int32_t zp_val, int64_t quant_min,
  int64_t quant_max, int64_t round_mode
) {
  const float scale_val = to_float(scale_raw);
  const float inv_scale = 1.0f / scale_val;
  const float input_val = to_float(input_raw);

  int64_t qval;
  switch (round_mode) {
    case 2:
      // ``0.5`` must remain a double literal; see file-level comment.
      qval =
        static_cast<int64_t>(std::floor(input_val * inv_scale + 0.5) + zp_val);
      break;
    case 3:
      qval = static_cast<int64_t>(std::round(input_val * inv_scale) + zp_val);
      break;
    case 8:
      qval =
        static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + zp_val);
      break;
    default:
      // Unreachable: ``launch_fake_quantize_kernel`` rejects unknown modes.
      qval = 0;
      break;
  }

  const float clamped = fminf(
    static_cast<float>(quant_max),
    fmaxf(static_cast<float>(quant_min), static_cast<float>(qval))
  );
  return from_float(T{}, (clamped - zp_val) * scale_val);
}

template <typename T>
__global__ void fake_quantize_kernel(
  T* output, const T* input, const T* scale, const int32_t* zero_point,
  int64_t quant_min, int64_t quant_max, int64_t round_mode, int64_t num_elements
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = fake_quantize_apply<T>(
      input[idx], *scale, *zero_point, quant_min, quant_max, round_mode
    );
  }
}

template <typename T>
void launch_fake_quantize_kernel(
  T* output, const T* input, const T* scale, const int32_t* zero_point,
  int64_t quant_min, int64_t quant_max, int64_t round_mode,
  int64_t num_elements, cudaStream_t stream
) {
  if (round_mode != 2 && round_mode != 3 && round_mode != 8) {
    throw std::runtime_error("Unknown round_mode");
  }
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;
  fake_quantize_kernel<T><<<num_blocks, block_size, 0, stream>>>(
    output, input, scale, zero_point, quant_min, quant_max, round_mode,
    num_elements
  );
}

torch::stable::Tensor fake_quantize_per_tensor_affine_impl(
  const torch::stable::Tensor& input, const torch::stable::Tensor& scale,
  const torch::stable::Tensor& zero_point, int64_t quant_min, int64_t quant_max,
  int64_t round_mode
) {
  quark::OptionalDeviceGuard guard(input);

  STD_TORCH_CHECK(
    quant_min <= quant_max,
    "`quant_min` should be less than or equal to `quant_max`."
  );

  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  STD_TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  STD_TORCH_CHECK(zero_point.is_cuda(), "zero_point must be a CUDA tensor");
  STD_TORCH_CHECK(zero_point.is_contiguous(), "zero_point must be contiguous");

  torch::stable::Tensor output = torch::stable::empty_like(input);

  int64_t num_elements = input.numel();

  const cudaStream_t stream = getCurrentStream();

  auto dtype = input.scalar_type();
  if (dtype == torch::headeronly::ScalarType::Half) {
    launch_fake_quantize_kernel<__half>(
      static_cast<__half*>(output.data_ptr()),
      static_cast<const __half*>(input.data_ptr()),
      static_cast<const __half*>(scale.data_ptr()),
      static_cast<const int32_t*>(zero_point.data_ptr()), quant_min, quant_max,
      round_mode, num_elements, stream
    );
  } else if (dtype == torch::headeronly::ScalarType::Float) {
    launch_fake_quantize_kernel<float>(
      static_cast<float*>(output.data_ptr()),
      static_cast<const float*>(input.data_ptr()),
      static_cast<const float*>(scale.data_ptr()),
      static_cast<const int32_t*>(zero_point.data_ptr()), quant_min, quant_max,
      round_mode, num_elements, stream
    );
  } else {
    throw std::runtime_error("Unsupported input data type");
  }

  return output;
}

}  // namespace hw_emulation
}  // namespace quark

#endif  // USE_CUDA
