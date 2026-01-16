// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "cast.hpp"

#include <numeric>

namespace ryzenai::onnx_utils {

template <typename kFrom, typename kTo>
void OrtCast<kFrom, kTo>::construct(const Ort::ConstKernelInfo& info) {
  std::vector<Ort::OpAttr> attrs1;
  attrs1.reserve(1);
  auto to = Ort::TypeToTensorType<kTo>().type;
  attrs1.emplace_back("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  createOp(
    info, "Cast", "", 21, {{"T1", Ort::TypeToTensorType<kFrom>().type}},
    std::move(attrs1), 1, 1
  );
}

template <typename kFrom, typename kTo>
void OrtCast<kFrom, kTo>::execute(
  kTo* output_data, kFrom* input_data, const std::vector<int64_t>& input_shape,
  OrtKernelContext* context
) {
  auto tensor_size = std::accumulate(
    input_shape.begin(), input_shape.end(), 1ULL, std::multiplies<>()
  );
  auto input_tensor = Ort::Value::CreateTensor<kFrom>(
    memory_info_, input_data, tensor_size, input_shape.data(),
    input_shape.size()
  );

  execute(output_data, input_tensor.GetConst(), context);
}

template <typename kFrom, typename kTo>
void OrtCast<kFrom, kTo>::execute(
  kTo* output_data, const Ort::ConstValue& input_tensor,
  OrtKernelContext* context
) {
  if (!isInitialized()) {
    throw std::runtime_error("Cast operator is not initialized");
  }

  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto tensor_size = input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

  Ort::Value output_tensor = Ort::Value::CreateTensor<kTo>(
    memory_info_, output_data, tensor_size, input_shape.data(),
    input_shape.size()
  );

  invokeOp(context, {input_tensor}, {output_tensor});
}

template class OrtCast<Ort::BFloat16_t, Ort::Float16_t>;
template class OrtCast<Ort::Float16_t, Ort::BFloat16_t>;
template class OrtCast<Ort::BFloat16_t, float>;
template class OrtCast<float, Ort::BFloat16_t>;
template class OrtCast<Ort::Float16_t, float>;
template class OrtCast<float, Ort::Float16_t>;

}  // namespace ryzenai::onnx_utils
