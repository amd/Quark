// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "rotary_embedding.hpp"

#include <cassert>
#include <iostream>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtRotaryEmbedding::construct(
  const Ort::ConstKernelInfo& info, float scale, int64_t rotary_interleaved,
  int64_t num_heads, int64_t rotary_embedding_dim
) {
  // scale
  // auto val_scale_rope_q = scale_;
  auto attr_scale_rope =
    Ort::OpAttr("scale", &scale, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
  // interleaved
  // int64_t val_interleaved_q = rotary_interleaved_;
  auto attr_interleaved_q = Ort::OpAttr(
    "interleaved", &rotary_interleaved, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );

  // num_heads for q
  // int64_t val_num_heads_q = num_heads_;
  auto attr_num_heads_q =
    Ort::OpAttr("num_heads", &num_heads, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  // int64_t val_rotary_embedding_dim_q = rotary_embedding_dim;
  auto attr_rotary_embedding_dim_q = Ort::OpAttr(
    "rotary_embedding_dim", &rotary_embedding_dim, 1,
    OrtOpAttrType::ORT_OP_ATTR_INT
  );

  std::vector<Ort::OpAttr> attrs;

  attrs.push_back(std::move(attr_scale_rope));
  attrs.push_back(std::move(attr_interleaved_q));
  attrs.push_back(std::move(attr_num_heads_q));
  attrs.push_back(std::move(attr_rotary_embedding_dim_q));

  createOp(
    info.Copy(), "RotaryEmbedding", "com.microsoft", 1,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
     {"M", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}},
    std::move(attrs), 4,
    1
  );  // 4 attributes, 4 inputs, 1 outputs
}

void OrtRotaryEmbedding::execute(
  float* output_data, float* input_data, int64_t* pos_ids_data,
  const OrtValue* cos_cache, const OrtValue* sin_cache,
  const std::vector<int64_t>& input_shape, OrtKernelContext* context,
  int rewind_pos
) {
  if (!isInitialized()) {
    throw std::runtime_error("Rotary Embedding operator is not initialized");
  }
  if (input_shape.size() != 4) {
    throw std::runtime_error("Input shape size does not equal 4");
  }

  auto tensor_size = std::accumulate(
    input_shape.begin(), input_shape.end(), 1ULL, std::multiplies<>()
  );
  const auto& batch = input_shape[0];
  const auto& kv_num_heads = input_shape[1];
  const auto& total_seq_len = input_shape[2];
  const auto& head_size = input_shape[3];

  std::vector<int64_t> pos_ids_shape{batch};
  Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    info, input_data, tensor_size, input_shape.data(), input_shape.size()
  );
  Ort::Value pos_ids = Ort::Value::CreateTensor<int64_t>(
    info, pos_ids_data, rewind_pos ? total_seq_len : batch,
    pos_ids_shape.data(), pos_ids_shape.size()
  );
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
    info, output_data, tensor_size, input_shape.data(), input_shape.size()
  );
  try {
    invokeOp(
      context, {input_tensor, pos_ids, cos_cache, sin_cache}, {output_tensor}
    );
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    throw;
  }
}

}  // namespace ryzenai::onnx_utils
