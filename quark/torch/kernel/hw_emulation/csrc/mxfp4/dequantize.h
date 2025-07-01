#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <math.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#endif

#ifndef func_defined_dq
#define func_defined_dq


#ifdef USE_CUDA
template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__device__ float_type upcast_fp4_to_fp16_or_bf16(uint8_t val);

template<typename float_type, typename scale_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__global__ void dq_uint8_mxfp4_to_half_kernel(uint8_t* inp, scale_type* scales, float_type* out);

void dq_uint8_mxfp4_to_half(torch::Tensor inp, torch::Tensor scales, torch::Tensor out, int group_size);

#else

void dq_uint8_mxfp4_to_half(torch::Tensor inp, torch::Tensor scales, torch::Tensor out, int group_size) {
    TORCH_CHECK(false, "dq_uint8_mxfp4_to_half is only implemented in CUDA devices! Please check your installation.")
}
#endif
#endif
