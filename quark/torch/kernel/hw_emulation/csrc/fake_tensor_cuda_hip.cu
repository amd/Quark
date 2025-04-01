//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/ATen.h>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename T>
void fake_quantize_tensor_kernel_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t round_mode) {
    T * scale_ptr = scale.data_ptr<T>();
    int32_t* zp_ptr = zero_point.data_ptr<int32_t>();
    auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(output)
    .add_input(input)
    .build();


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fake_quantize_tensor_kernel_types", [&] {
    switch(round_mode){
    case 2:
        at::native::gpu_kernel(
            iter,
            [=] GPU_LAMBDA (scalar_t input_val) -> scalar_t {
                float inv_scale = 1.0f / (*scale_ptr);
                const auto qval = static_cast<int64_t>(std::floor(input_val * inv_scale + 0.5) + (*zp_ptr));
                return  (fminf(quant_max, fmaxf(quant_min, qval)) - (*zp_ptr)) * (*scale_ptr);
            }
        );
        break;
    case 3:
        at::native::gpu_kernel(
            iter,
            [=] GPU_LAMBDA (scalar_t input_val) -> scalar_t {
                float inv_scale = 1.0f / (*scale_ptr);
                const auto qval = static_cast<int64_t>(std::round(input_val * inv_scale) + (*zp_ptr));
                return  (fminf(quant_max, fmaxf(quant_min, qval)) - (*zp_ptr)) * (*scale_ptr);
            }
        );
        break;
    case 8:
        at::native::gpu_kernel(
          iter,
          [=] GPU_LAMBDA (scalar_t input_val) -> scalar_t {
            float inv_scale = 1.0f / (*scale_ptr);
            const auto qval = static_cast<int64_t>(std::nearbyint(input_val * inv_scale) + (*zp_ptr));
            return  (fminf(quant_max, fmaxf(quant_min, qval)) - (*zp_ptr)) * (*scale_ptr);
          }
        );
        break;
    default:
        TORCH_CHECK(false, "Unknown round_method");
        break;
    }
  });
}

torch::Tensor fake_quantize_per_tensor_affine(const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& zero_point, int64_t quant_min, int64_t quant_max, int64_t round_mode){

    TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    CHECK_INPUT(zero_point);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto Y = at::empty_like(input, input.options(), c10::MemoryFormat::Preserve);

    switch(input.scalar_type()){
      case at::ScalarType::Half:
        fake_quantize_tensor_kernel_cuda<at::Half>(Y , input, scale, zero_point, quant_min, quant_max, round_mode);
        break;
      // case at::ScalarType::BFloat16:
      //   fake_quantize_tensor_kernel_cuda<at::BFloat16>(Y , input, scale, zero_point, quant_min, quant_max, round_mode);
      //   break;
      case at::ScalarType::Float:
        fake_quantize_tensor_kernel_cuda<float>(Y , input, scale, zero_point, quant_min, quant_max, round_mode);
        break;
      default:
        std::cerr << "input data type not support!" << std::endl;
        abort();
    }
    return Y;
}
