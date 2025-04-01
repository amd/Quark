//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifdef USE_CUDA
#include "bfp/cuda/bfp_kernel.h"
#include "mx/cuda/mx_kernel.h"
#else
#include "bfp/cpu/bfp_kernel.h"
#include "mx/cpu/mx_kernel.h"
#endif

#include <torch/extension.h>


#ifdef USE_CUDA

torch::Tensor bfp_kernel(torch::Tensor& tensor, int bit_width, int block_size, int rounding_mode, int kernel_version) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCUDA);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in, torch::kCUDA).to(device);
    float* output = tensor_out.data_ptr<float>();

    std::vector<int64_t> dimensions = tensor.sizes().vec();
    int axis_size = dimensions[dimensions.size() - 1];

    LaunchBFPCUDAKernel(input, output, element_count, axis_size, bit_width, block_size, rounding_mode, kernel_version);

    return tensor_out;
}

torch::Tensor bfp_prime_kernel(torch::Tensor& tensor, int bit_width, int block_size,
                               int sub_block_size, int sub_block_shift_bits, int rounding_mode) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCUDA);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in, torch::kCUDA).to(device);
    float* output = tensor_out.data_ptr<float>();

    std::vector<int64_t> dimensions = tensor.sizes().vec();
    int axis_size = dimensions[dimensions.size() - 1];

    LaunchBFPPrimeCUDAKernel(input, output, element_count, axis_size, bit_width, block_size, sub_block_size,
                             sub_block_shift_bits, rounding_mode);

    return tensor_out;
}

torch::Tensor mx_kernel(torch::Tensor& tensor, int block_size, int ebits, int mbits, int emax, float max_norm, float min_norm, int rounding_mode) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCUDA);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in, torch::kCUDA).to(device);
    float* output = tensor_out.data_ptr<float>();

    std::vector<int64_t> dimensions = tensor.sizes().vec();
    int axis_size = dimensions[dimensions.size() - 1];

    LaunchMXCUDAKernel(input, output, element_count, axis_size, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode);

    return tensor_out;
}

#else

torch::Tensor bfp_kernel(torch::Tensor& tensor, int bit_width, int block_size, int rounding_mode,
		int kernel_version) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCPU);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in);
    float* output = tensor_out.data_ptr<float>();

    LaunchBFPCPUKernel(input, output, element_count, bit_width, block_size,
		                         rounding_mode, kernel_version);

    return tensor_out.to(device);
}

torch::Tensor bfp_prime_kernel(torch::Tensor& tensor, int bit_width, int block_size,
		int sub_block_size, int sub_block_shift_bits, int rounding_mode) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCPU);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in);
    float* output = tensor_out.data_ptr<float>();

    LaunchBFPPrimeCPUKernel(input, output, element_count, bit_width, block_size,
		    sub_block_size, sub_block_shift_bits, rounding_mode);

    return tensor_out.to(device);
}

torch::Tensor mx_kernel(torch::Tensor& tensor, int block_size, int ebits, int mbits, int emax, float max_norm, float min_norm, int rounding_mode) {
    auto device = tensor.device();

    torch::Tensor tensor_in = tensor.to(torch::kCPU);
    float* input = tensor_in.data_ptr<float>();
    int element_count = tensor_in.numel();

    torch::Tensor tensor_out = torch::empty_like(tensor_in);
    float* output = tensor_out.data_ptr<float>();

    LaunchMXCPUKernel(input, output, element_count, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode);

    return tensor_out.to(device);
}

#endif


torch::Tensor bfp(torch::Tensor& tensor, int bit_width, int block_size, int rounding_mode,
		int kernel_version) {
    return bfp_kernel(tensor, bit_width, block_size, rounding_mode, kernel_version);
}


torch::Tensor bfp_prime(torch::Tensor& tensor, int bit_width, int block_size,
		int sub_block_size, int sub_block_shift_bits, int rounding_mode) {
    return bfp_prime_kernel(tensor, bit_width, block_size, sub_block_size,
                            sub_block_shift_bits, rounding_mode);
}


torch::Tensor mx(torch::Tensor& tensor, int block_size, int ebits, int mbits, int emax, float max_norm, float min_norm, int rounding_mode) {
    return mx_kernel(tensor, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode);
}


// Create python module via PYBIND11_MODULE macro
#ifdef _WIN32
PYBIND11_MODULE(custom_ops, m) {
    m.def("bfp", &bfp, "A function that quantizes a tensor in BFP format");
    m.def("bfp_prime", &bfp_prime, "A function that quantizes a tensor in BFP format by prime method");
    m.def("mx", &mx, "A function that quantizes a tensor in MX format");
}
PYBIND11_MODULE(libcustom_ops, m) {
    m.def("bfp", &bfp, "A function that quantizes a tensor in BFP format");
    m.def("bfp_prime", &bfp_prime, "A function that quantizes a tensor in BFP format by prime method");
    m.def("mx", &mx, "A function that quantizes a tensor in MX format");
}
#else
#ifdef USE_CUDA
PYBIND11_MODULE(libcustom_ops_gpu, m) {
    m.def("bfp", &bfp, "A function that quantizes a tensor in BFP format");
    m.def("bfp_prime", &bfp_prime, "A function that quantizes a tensor in BFP format by prime method");
    m.def("mx", &mx, "A function that quantizes a tensor in MX format");
}
#else
PYBIND11_MODULE(libcustom_ops, m) {
    m.def("bfp", &bfp, "A function that quantizes a tensor in BFP format");
    m.def("bfp_prime", &bfp_prime, "A function that quantizes a tensor in BFP format by prime method");
    m.def("mx", &mx, "A function that quantizes a tensor in MX format");
}
#endif
#endif
