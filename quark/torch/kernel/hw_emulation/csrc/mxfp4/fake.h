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

#ifndef func_defined_qdq
#define func_defined_qdq

#ifdef USE_CUDA
template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__device__ float_type fp16_to_fp4_simulate(float_type* val);

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias, uint16_t val_to_add, uint16_t sign_exponent_mask>
__global__ void qdq_mxfp4_kernel(float_type* inp, float_type* out);

// in place
void qdq_mxfp4_(torch::Tensor a, int group_size);

// out of place
torch::Tensor qdq_mxfp4(torch::Tensor a, int group_size);

#else
void qdq_mxfp4_(torch::Tensor a, int group_size) {
    TORCH_CHECK(false, "qdq_mxfp4_ is only implemented in CUDA devices! Please check your installation.")
}

torch::Tensor qdq_mxfp4(torch::Tensor a, int group_size) {
    TORCH_CHECK(false, "qdq_mxfp4 is only implemented in CUDA devices! Please check your installation.")
}
#endif

#endif
