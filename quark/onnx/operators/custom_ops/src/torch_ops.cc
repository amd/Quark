//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// PyTorch Stable ABI custom ops (torch >= 2.10). Dispatches on the input
// tensor's runtime device, so a single build services CPU and CUDA inputs.
//

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <vector>

// Forward-declared rather than #include'd: the CPU and CUDA kernel headers
// both define ``enum rounding_mode_enum`` at file scope, which would trip
// an ODR violation when pulled in together.
void LaunchBFPCPUKernel(
  const float* input, float* output, int n, int bit_width, int block_size,
  int rounding_mode, int use_compiler_version_cpu_kernel
);
void LaunchBFPPrimeCPUKernel(
  const float* input, float* output, const int n, const int bit_width,
  const int block_size, const int sub_block_size,
  const int sub_block_shift_bits, const int rounding_mode
);
void LaunchMXCPUKernel(
  const float* input, float* output, const int n, const int block_size,
  const int ebits, const int mbits, const int emax, const float max_norm,
  const float min_norm, const int rounding_mode
);

#ifdef USE_CUDA
void LaunchBFPCUDAKernel(
  const float* input, float* output, const int n, const int axis_size,
  const int bit_width, const int block_size, const int rounding_mode,
  int use_compiler_version_cpu_kernel
);
void LaunchBFPPrimeCUDAKernel(
  const float* input, float* output, const int n, const int axis_size,
  const int bit_width, const int block_size, const int sub_block_size,
  const int sub_block_shift_bits, const int rounding_mode
);
void LaunchMXCUDAKernel(
  const float* input, float* output, const int n, const int axis_size,
  const int block_size, const int ebits, const int mbits, const int emax,
  const float max_norm, const float min_norm, const int rounding_mode
);
#endif

namespace quark {
namespace custom_ops {

namespace {

int last_axis_size(const torch::stable::Tensor& tensor) {
  auto sizes = tensor.sizes();
  return static_cast<int>(sizes[sizes.size() - 1]);
}

}  // namespace

torch::stable::Tensor bfp_kernel(
  const torch::stable::Tensor& tensor, int64_t bit_width, int64_t block_size,
  int64_t rounding_mode, int64_t kernel_version
) {
  auto device = tensor.device();
  int element_count = static_cast<int>(tensor.numel());

#ifdef USE_CUDA
  bool on_cuda = tensor.is_cuda();
#else
  constexpr bool on_cuda = false;
#endif

  torch::stable::Tensor tensor_in = torch::stable::to(
    tensor, on_cuda ? torch::headeronly::kCUDA : torch::headeronly::kCPU
  );
  torch::stable::Tensor tensor_out = torch::stable::empty_like(tensor_in);

  // Typed accessor enforces float32; kernels assume IEEE-754 binary32 lanes.
  float* input = tensor_in.mutable_data_ptr<float>();
  float* output = tensor_out.mutable_data_ptr<float>();

#ifdef USE_CUDA
  if (on_cuda) {
    LaunchBFPCUDAKernel(
      input, output, element_count, last_axis_size(tensor),
      static_cast<int>(bit_width), static_cast<int>(block_size),
      static_cast<int>(rounding_mode), static_cast<int>(kernel_version)
    );
  } else
#endif
  {
    LaunchBFPCPUKernel(
      input, output, element_count, static_cast<int>(bit_width),
      static_cast<int>(block_size), static_cast<int>(rounding_mode),
      static_cast<int>(kernel_version)
    );
  }

  return torch::stable::to(tensor_out, device);
}

torch::stable::Tensor bfp_prime_kernel(
  const torch::stable::Tensor& tensor, int64_t bit_width, int64_t block_size,
  int64_t sub_block_size, int64_t sub_block_shift_bits, int64_t rounding_mode
) {
  auto device = tensor.device();
  int element_count = static_cast<int>(tensor.numel());

#ifdef USE_CUDA
  bool on_cuda = tensor.is_cuda();
#else
  constexpr bool on_cuda = false;
#endif

  torch::stable::Tensor tensor_in = torch::stable::to(
    tensor, on_cuda ? torch::headeronly::kCUDA : torch::headeronly::kCPU
  );
  torch::stable::Tensor tensor_out = torch::stable::empty_like(tensor_in);

  float* input = tensor_in.mutable_data_ptr<float>();
  float* output = tensor_out.mutable_data_ptr<float>();

#ifdef USE_CUDA
  if (on_cuda) {
    LaunchBFPPrimeCUDAKernel(
      input, output, element_count, last_axis_size(tensor),
      static_cast<int>(bit_width), static_cast<int>(block_size),
      static_cast<int>(sub_block_size), static_cast<int>(sub_block_shift_bits),
      static_cast<int>(rounding_mode)
    );
  } else
#endif
  {
    LaunchBFPPrimeCPUKernel(
      input, output, element_count, static_cast<int>(bit_width),
      static_cast<int>(block_size), static_cast<int>(sub_block_size),
      static_cast<int>(sub_block_shift_bits), static_cast<int>(rounding_mode)
    );
  }

  return torch::stable::to(tensor_out, device);
}

torch::stable::Tensor mx_kernel(
  const torch::stable::Tensor& tensor, int64_t block_size, int64_t ebits,
  int64_t mbits, int64_t emax, double max_norm, double min_norm,
  int64_t rounding_mode
) {
  auto device = tensor.device();
  int element_count = static_cast<int>(tensor.numel());

#ifdef USE_CUDA
  bool on_cuda = tensor.is_cuda();
#else
  constexpr bool on_cuda = false;
#endif

  torch::stable::Tensor tensor_in = torch::stable::to(
    tensor, on_cuda ? torch::headeronly::kCUDA : torch::headeronly::kCPU
  );
  torch::stable::Tensor tensor_out = torch::stable::empty_like(tensor_in);

  float* input = tensor_in.mutable_data_ptr<float>();
  float* output = tensor_out.mutable_data_ptr<float>();

#ifdef USE_CUDA
  if (on_cuda) {
    LaunchMXCUDAKernel(
      input, output, element_count, last_axis_size(tensor),
      static_cast<int>(block_size), static_cast<int>(ebits),
      static_cast<int>(mbits), static_cast<int>(emax),
      static_cast<float>(max_norm), static_cast<float>(min_norm),
      static_cast<int>(rounding_mode)
    );
  } else
#endif
  {
    LaunchMXCPUKernel(
      input, output, element_count, static_cast<int>(block_size),
      static_cast<int>(ebits), static_cast<int>(mbits), static_cast<int>(emax),
      static_cast<float>(max_norm), static_cast<float>(min_norm),
      static_cast<int>(rounding_mode)
    );
  }

  return torch::stable::to(tensor_out, device);
}

torch::stable::Tensor bfp(
  const torch::stable::Tensor& tensor, int64_t bit_width, int64_t block_size,
  int64_t rounding_mode, int64_t kernel_version
) {
  return bfp_kernel(
    tensor, bit_width, block_size, rounding_mode, kernel_version
  );
}

torch::stable::Tensor bfp_prime(
  const torch::stable::Tensor& tensor, int64_t bit_width, int64_t block_size,
  int64_t sub_block_size, int64_t sub_block_shift_bits, int64_t rounding_mode
) {
  return bfp_prime_kernel(
    tensor, bit_width, block_size, sub_block_size, sub_block_shift_bits,
    rounding_mode
  );
}

torch::stable::Tensor mx(
  const torch::stable::Tensor& tensor, int64_t block_size, int64_t ebits,
  int64_t mbits, int64_t emax, double max_norm, double min_norm,
  int64_t rounding_mode
) {
  return mx_kernel(
    tensor, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode
  );
}

}  // namespace custom_ops
}  // namespace quark

STABLE_TORCH_LIBRARY(quark_custom_ops, m) {
  m.def(
    "bfp(Tensor tensor, int bit_width, int block_size, int rounding_mode, int "
    "kernel_version) -> Tensor"
  );
  m.def(
    "bfp_prime(Tensor tensor, int bit_width, int block_size, int "
    "sub_block_size, int sub_block_shift_bits, int rounding_mode) -> Tensor"
  );
  m.def(
    "mx(Tensor tensor, int block_size, int ebits, int mbits, int emax, float "
    "max_norm, float min_norm, int rounding_mode) -> Tensor"
  );
}

STABLE_TORCH_LIBRARY_IMPL(quark_custom_ops, CompositeExplicitAutograd, m) {
  m.impl("bfp", TORCH_BOX(&quark::custom_ops::bfp));
  m.impl("bfp_prime", TORCH_BOX(&quark::custom_ops::bfp_prime));
  m.impl("mx", TORCH_BOX(&quark::custom_ops::mx));
}
