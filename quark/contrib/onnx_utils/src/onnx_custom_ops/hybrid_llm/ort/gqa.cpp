// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "gqa.hpp"

#include <cassert>
#include <iostream>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtGQA::construct(
  const Ort::ConstKernelInfo& info, int64_t do_rotary, float scale,
  int64_t rotary_interleaved, int64_t num_heads, int64_t kv_num_heads,
  int64_t rotary_embedding_dim, float softcap, int64_t local_window_size,
  bool has_head_sink
) {
  auto attr_do_rotary =
    Ort::OpAttr("do_rotary", &do_rotary, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  auto attr_kv_num_heads = Ort::OpAttr(
    "kv_num_heads", &kv_num_heads, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  auto attr_num_heads =
    Ort::OpAttr("num_heads", &num_heads, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  auto attr_rotary_interleave = Ort::OpAttr(
    "rotary_interleaved", &rotary_interleaved, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  auto attr_scale =
    Ort::OpAttr("scale", &scale, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);

  auto attr_local_window_size = Ort::OpAttr(
    "local_window_size", &local_window_size, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );

  auto attr_softcap =
    Ort::OpAttr("softcap", &softcap, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);

  std::vector<Ort::OpAttr> attrs;

  attrs.push_back(std::move(attr_do_rotary));
  attrs.push_back(std::move(attr_kv_num_heads));
  attrs.push_back(std::move(attr_num_heads));
  attrs.push_back(std::move(attr_rotary_interleave));
  attrs.push_back(std::move(attr_scale));
  attrs.push_back(std::move(attr_softcap));
  attrs.push_back(std::move(attr_local_window_size));

  has_head_sink_ = has_head_sink;
  const auto num_inputs = has_head_sink ? (12) : 9;

  createOp(
    info.Copy(), "GroupQueryAttention", "com.microsoft", 1,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}}, std::move(attrs), num_inputs,
    3
  );  // 7 attributes, 9 inputs by default followed by [position_ids,
      // attention_bias, head_sink], 3 output
}

void OrtGQA::execute(
  float* output_data, float* present_k_data, float* present_v_data,
  float* q_data, float* k_data, float* v_data, float* past_k_data,
  float* past_v_data, const std::vector<int64_t>& q_shape,
  const std::vector<int64_t>& kv_shape,
  const std::vector<int64_t>& past_k_shape,
  const std::vector<int64_t>& past_v_shape,
  const std::vector<int64_t>& present_k_shape,
  const std::vector<int64_t>& present_v_shape, int64_t seq_len,
  const Ort::ConstValue& cos_cache, const Ort::ConstValue& sin_cache,
  std::vector<float>& head_sink, OrtKernelContext* context
) {
  if (!isInitialized()) {
    throw std::runtime_error("GQA operator is not initialized");
  }
  if (q_shape.size() != 3) {
    throw std::runtime_error("q_shape size does not equal 3");
  }
  if (kv_shape.size() != 3) {
    throw std::runtime_error("kv_shape size does not equal 3");
  }
  if (past_k_shape.size() != 4) {
    throw std::runtime_error("past_k_shape size does not equal 4");
  }
  if (past_v_shape.size() != 4) {
    throw std::runtime_error("past_v_shape size does not equal 4");
  }
  if (present_k_shape.size() != 4) {
    throw std::runtime_error("present_k_shape size does not equal 4");
  }
  if (present_v_shape.size() != 4) {
    throw std::runtime_error("present_v_shape size does not equal 4");
  }

  auto q_size =
    std::accumulate(q_shape.begin(), q_shape.end(), 1ULL, std::multiplies<>());
  auto kv_size = std::accumulate(
    kv_shape.begin(), kv_shape.end(), 1ULL, std::multiplies<>()
  );
  auto past_k_size = std::accumulate(
    past_k_shape.begin(), past_k_shape.end(), 1ULL, std::multiplies<>()
  );
  auto past_v_size = std::accumulate(
    past_v_shape.begin(), past_v_shape.end(), 1ULL, std::multiplies<>()
  );
  auto present_k_size = std::accumulate(
    present_k_shape.begin(), present_k_shape.end(), 1ULL, std::multiplies<>()
  );
  auto present_v_size = std::accumulate(
    present_v_shape.begin(), present_v_shape.end(), 1ULL, std::multiplies<>()
  );
  // const auto& batch = past_k_shape[0];
  // const auto& num_heads = q_shape[1];
  // const auto& kv_num_heads = kv_shape[1];
  // const auto& total_seq_len = past_k_shape[2];
  // const auto& head_size = past_k_shape[3];

  const std::vector<int64_t>& output_shape = q_shape;
  auto output_size = std::accumulate(
    output_shape.begin(), output_shape.end(), 1ULL, std::multiplies<>()
  );

  Ort::MemoryInfo info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Create single Ort tensors
  auto t = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  // std::vector<int64_t> q_shape{B, S, N_q * H};
  // std::vector<int64_t> k_shape{B, S, N_kv * H};
  // std::vector<int64_t> v_shape{B, S, N_kv * H};
  Ort::Value fp_q = Ort::Value::CreateTensor<float>(
    info, q_data, q_size, q_shape.data(), q_shape.size()
  );
  Ort::Value fp_k = Ort::Value::CreateTensor<float>(
    info, k_data, kv_size, kv_shape.data(), kv_shape.size()
  );
  Ort::Value fp_v = Ort::Value::CreateTensor<float>(
    info, v_data, kv_size, kv_shape.data(), kv_shape.size()
  );

  Ort::Value fp_p_k = Ort::Value::CreateTensor<float>(
    info, past_k_data, past_k_size, past_k_shape.data(), past_k_shape.size()
  );
  Ort::Value fp_p_v = Ort::Value::CreateTensor<float>(
    info, past_v_data, past_v_size, past_v_shape.data(), past_v_shape.size()
  );

  Ort::KernelContext ctx(context);
  auto seqlens_k = ctx.GetInput(5);
  auto total_seqlen = ctx.GetInput(6);
  //__TIC__(ORTKernelCreateOutputOrtTensorValue)
  Ort::Value fp_out = Ort::Value::CreateTensor<float>(
    info, output_data, output_size, output_shape.data(), output_shape.size()
  );
  Ort::Value fp_present_key = Ort::Value::CreateTensor<float>(
    info, present_k_data, present_k_size, present_k_shape.data(),
    present_k_shape.size()
  );
  Ort::Value fp_present_value = Ort::Value::CreateTensor<float>(
    info, present_v_data, present_v_size, present_v_shape.data(),
    present_v_shape.size()
  );

  Ort::Value head_sink_value;

  const std::vector<int64_t> head_sink_shape = {(int64_t)head_sink.size()};

  if (has_head_sink_) {
    head_sink_value = Ort::Value::CreateTensor<float>(
      info, head_sink.data(), head_sink.size(), head_sink_shape.data(),
      head_sink_shape.size()
    );
  }
  //__TOC__(ORTKernelCreateOutputOrtTensorValue)

  try {
    if (has_head_sink_) {
      invokeOp(
        context,
        {fp_q, fp_k, fp_v, fp_p_k, fp_p_v, seqlens_k, total_seqlen, cos_cache,
         sin_cache, nullptr, nullptr, head_sink_value},
        {fp_out, fp_present_key, fp_present_value}
      );
    } else {
      invokeOp(
        context,
        {fp_q, fp_k, fp_v, fp_p_k, fp_p_v, seqlens_k, total_seqlen, cos_cache,
         sin_cache},
        {fp_out, fp_present_key, fp_present_value}
      );
    }

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    throw;
  }
}

}  // namespace ryzenai::onnx_utils
