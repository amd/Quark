//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include "mx/funcs.cuh"

#ifdef USE_CUDA
torch::Tensor fake_quantize_per_tensor_affine(const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& zero_point, int64_t quant_min, int64_t quant_max, int64_t round_mode);
#elif ! defined(USE_CUDA)
torch::Tensor fake_quantize_per_tensor_affine(const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& zero_point, int64_t quant_min, int64_t quant_max, int64_t round_mode) {
  throw std::runtime_error("fake_quantize_per_tensor_affine is not implemented on non CUDA-devices");
}
#endif

torch::Tensor fake_quantize_to_low_precision_fp(
        torch::Tensor &input,
        int ebits,
        int mbits,
        float max_norm,
        uint32_t round_mode
);

std::vector<at::Tensor> tqt_backward(
    at::Tensor& x, at::Tensor& scale, at::Tensor& quant_max,
    at::Tensor& quant_min, at::Tensor& logt, at::Tensor& grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_quantize_per_tensor_affine", &fake_quantize_per_tensor_affine, "fake_quantize_per_tensor_affine function",
        py::arg("inputs"),
        py::arg("scale"),
        py::arg("zero_point"),
        py::arg("quant_min"),
        py::arg("quant_max"),
        py::arg("round_mode")
  );
  m.def("fake_quantize_to_low_precision_fp", &fake_quantize_to_low_precision_fp, "fake_quantize_to_low_precision_fp",
        py::arg("input"),
        py::arg("ebits"),
        py::arg("mbits"),
        py::arg("max_norm"),
        py::arg("round_mode")
  );

  m.def("tqt_backward", &tqt_backward, "tqt_backward function",
	py::arg("x"),
	py::arg("scale"),
	py::arg("quant_max"),
	py::arg("quant_min"),
	py::arg("logt"),
	py::arg("grad_output")
  );
}
