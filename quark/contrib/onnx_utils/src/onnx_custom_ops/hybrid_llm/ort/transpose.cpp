// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "transpose.hpp"

#include <cassert>
#include <numeric>

namespace ryzenai::onnx_utils {

template <typename T>
void OrtTranspose<T>::construct(
  const Ort::ConstKernelInfo& info, std::vector<int64_t> perm
) {
  perm_ = std::move(perm);

  // std::vector<int64_t> perm_vec{0, 2, 1, 3};
  auto perm_attr = Ort::OpAttr(
    "perm", perm_.data(), perm_.size(), OrtOpAttrType::ORT_OP_ATTR_INTS
  );

  std::vector<Ort::OpAttr> attrs;

  attrs.push_back(std::move(perm_attr));

  createOp(
    info, "Transpose", "ai.onnx", 21, {{"T", Ort::TypeToTensorType<T>().type}},
    std::move(attrs), 1,
    1
  );  // 1 attributes, 1 input, 1 output
}

template <typename T>
void OrtTranspose<T>::execute(
  T* output_data, T* input_data, const std::vector<int64_t>& input_shape,
  OrtKernelContext* context
) {
  if (!isInitialized()) {
    throw std::runtime_error("Transpose operator is not initialized");
  }
  if (input_shape.size() != perm_.size()) {
    throw std::runtime_error(
      "Input shape size does not match permutation size"
    );
  }
  auto shape_count = input_shape.size();
  auto tensor_size = std::accumulate(
    input_shape.begin(), input_shape.end(), 1ULL, std::multiplies<>()
  );
  std::vector<int64_t> output_shape{input_shape};
  for (size_t i = 0; i < perm_.size(); ++i) {
    output_shape[i] = input_shape[perm_[i]];
  }
  //   int64_t input_shape[4] = {D0, D1, D2, D3};
  //   int64_t output_shape[4] = {D0, D2, D1, D3};

  Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto bf16_enum = Ort::TypeToTensorType<T>().type;
  //   OrtValue* input = nullptr;
  auto input = Ort::Value::CreateTensor<T>(
    info, input_data, tensor_size * sizeof(T), input_shape.data(), shape_count
  );
  auto output = Ort::Value::CreateTensor<T>(
    info, output_data, tensor_size * sizeof(T), output_shape.data(), shape_count
  );

  invokeOp(context, {input}, {output});
}

template class OrtTranspose<Ort::BFloat16_t>;
template class OrtTranspose<float>;

}  // namespace ryzenai::onnx_utils
