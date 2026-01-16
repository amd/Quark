// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "matmul.hpp"

#include <cassert>
#include <iostream>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtMatMul::construct(const Ort::ConstKernelInfo& info) {
  std::vector<Ort::OpAttr> attrs;

  createOp(
    info, "MatMul", "", 13, {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},
    std::move(attrs), 2,
    1
  );  // empty attributes, 2 inputs, 1 output
}

void OrtMatMul::execute(
  OrtKernelContext* context, float* activation_ptr,
  std::vector<int64_t> act_dim, float* weights_ptr,
  std::vector<int64_t> wts_dim, float* output_ptr, std::vector<int64_t> out_dim
) {
  if (!isInitialized()) {
    throw std::runtime_error("MatMul operator is not initialized");
  }

  auto act_size =
    std::accumulate(act_dim.begin(), act_dim.end(), 1ULL, std::multiplies<>());

  auto wts_size =
    std::accumulate(wts_dim.begin(), wts_dim.end(), 1ULL, std::multiplies<>());

  auto out_size =
    std::accumulate(out_dim.begin(), out_dim.end(), 1ULL, std::multiplies<>());

  // Create single Ort tensor
  Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value activation = Ort::Value::CreateTensor<float>(
    info, activation_ptr, act_size, act_dim.data(), act_dim.size()
  );

  Ort::Value weights = Ort::Value::CreateTensor<float>(
    info, weights_ptr, wts_size, wts_dim.data(), wts_dim.size()
  );

  Ort::Value out = Ort::Value::CreateTensor<float>(
    info, output_ptr, out_size, out_dim.data(), out_dim.size()
  );

  try {
    invokeOp(context, {activation, weights}, {out});

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    throw;
  }
}

}  // namespace ryzenai::onnx_utils
