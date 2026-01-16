// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "skip_simplified_layer_norm.hpp"

#include <numeric>

#include "hybrid_llm/npu/common.hpp"

namespace ryzenai::onnx_utils {

void OrtSkipSimplifiedLayerNorm::construct(const Ort::ConstKernelInfo& info) {
  auto epsilon = info.GetAttribute<float>("epsilon");

  Ort::OpAttr attr_epsilon(
    "epsilon", &epsilon, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT
  );

  std::vector<Ort::OpAttr> attrs;

  attrs.push_back(std::move(attr_epsilon));

  createOp(
    info, "SkipSimplifiedLayerNormalization", "com.microsoft", 1,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}}, std::move(attrs), 3,
    4
  );  // 1 attributes, 3 inputs, 2 outputs
}

void OrtSkipSimplifiedLayerNorm::execute(
  float* output_1_data, float* output_2_data, float* input_a_data,
  float* input_b_data, const std::vector<int64_t>& input_shape,
  float* weights_data, const std::vector<int64_t>& weights_shape,
  OrtKernelContext* context
) {
  if (!isInitialized()) {
    throw std::runtime_error(
      "SkipSimplifiedLayerNorm operator is not initialized"
    );
  }
  if (input_shape.size() != 3) {
    throw std::runtime_error(
      "Input shape size must be 3 for SimplifiedLayerNorm"
    );
  }

  auto num_elements = std::accumulate(
    input_shape.begin(), input_shape.end(), 1ULL, std::multiplies<>()
  );
  auto num_weights = std::accumulate(
    weights_shape.begin(), weights_shape.end(), 1ULL, std::multiplies<>()
  );

  Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
    memory_info_, input_a_data, num_elements, input_shape.data(),
    input_shape.size()
  );
  Ort::Value input_tensor_b = Ort::Value::CreateTensor<float>(
    memory_info_, input_b_data, num_elements, input_shape.data(),
    input_shape.size()
  );
  Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
    memory_info_, output_1_data, num_elements, input_shape.data(),
    input_shape.size()
  );
  Ort::Value output_tensor_2 = Ort::Value::CreateTensor<float>(
    memory_info_, output_2_data, num_elements, input_shape.data(),
    input_shape.size()
  );
  Ort::Value ort_wts_val = Ort::Value::CreateTensor<float>(
    memory_info_, weights_data, num_weights, weights_shape.data(),
    weights_shape.size()
  );

  invokeOp(
    context, {input_tensor_a, input_tensor_b, ort_wts_val.GetConst()},
    {output_tensor_1, nullptr, nullptr, output_tensor_2}
  );
}

}  // namespace ryzenai::onnx_utils
