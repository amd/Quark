// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "simplified_layer_norm.hpp"

#include "hybrid_llm/npu/common.hpp"
#include "ops/ops_common/dtype_utils.h"
#include "ort.hpp"
#include "ryzenai/ryzen_mm.h"

namespace ryzenai::onnx_utils {

void OrtSimplifiedLayerNorm::construct(const Ort::ConstKernelInfo& info) {
  // default values come from LayerNormalization spec
  auto epsilon = getAttribute<float>(info, "epsilon", 1e-5);
  auto axis = getAttribute<int64_t>(info, "axis", -1);
  auto stash_type = getAttribute<int64_t>(info, "stash_type", 1);

  auto attr_epsilon =
    Ort::OpAttr("epsilon", &epsilon, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
  auto attr_axis =
    Ort::OpAttr("axis", &axis, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  auto attr_stash_type =
    Ort::OpAttr("stash_type", &stash_type, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  std::vector<Ort::OpAttr> attrs;

  attrs.push_back(std::move(attr_axis));
  attrs.push_back(std::move(attr_epsilon));
  attrs.push_back(std::move(attr_stash_type));

  createOp(
    info, "SimplifiedLayerNormalization", "ai.onnx", 1,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}}, std::move(attrs), 2,
    1
  );  // 3 attributes, 2 inputs, 1 outputs
}

template <typename Wts, typename Out>
void OrtSimplifiedLayerNorm::execute(
  Out* output_data, Ort::BFloat16_t* input_data,
  const std::vector<int64_t>& input_shape, Wts* weights_data,
  const std::vector<int64_t>& weights_shape,
  const RyzenMM::Allocator& allocator, OrtKernelContext* context
) {
  if (!isInitialized()) {
    throw std::runtime_error("SimplifiedLayerNorm operator is not initialized");
  }
  if (input_shape.size() != 3) {
    throw std::runtime_error(
      "Input shape size must be 3 for SimplifiedLayerNorm"
    );
  }

  auto batch_size = input_shape[0];
  auto seq_length = input_shape[1];
  auto hidden_size = input_shape[2];

  Ort::KernelContext ctx(context);

  const auto& dimensions_out = input_shape;
  auto output = ctx.GetOutput(0, dimensions_out);  // Output activation

  size_t num_elements = batch_size * seq_length * hidden_size;

  RyzenMM::BufferRef input_a =
    allocator.AllocateBuffer(num_elements * sizeof(float));
  RyzenMM::BufferRef wts_a;
  RyzenMM::BufferRef output_1 =
    allocator.AllocateBuffer(num_elements * sizeof(float));

  float* wts_ptr;

  bfloat16_buffer_to_float(
    (uint16_t*)input_data, num_elements, input_a.Data<float>()
  );

  if constexpr (std::is_same_v<Wts, Ort::BFloat16_t>) {
    // If weights are in bfloat16, convert them to float
    wts_a = allocator.AllocateBuffer(num_elements * sizeof(float));
    wts_ptr = wts_a.Data<float>();
    bfloat16_buffer_to_float((uint16_t*)weights_data, num_elements, wts_ptr);

  } else if constexpr (std::is_same_v<Wts, Ort::Float16_t>) {
    wts_a = allocator.AllocateBuffer(num_elements * sizeof(float));
    wts_ptr = wts_a.Data<float>();
    float16_buffer_to_float((uint16_t*)weights_data, num_elements, wts_ptr);
  } else if constexpr (std::is_same_v<Wts, float>) {
    wts_ptr = weights_data;
  } else {
    static_assert(
      !sizeof(Wts),
      "Unsupported weights type. Only bfloat16, float16 or float "
      "are supported."
    );
  }

  // Create input tensor
  Ort::Value input_tensor_a = Ort::Value::CreateTensor<float>(
    memory_info_, input_a.Data<float>(), num_elements, input_shape.data(),
    input_shape.size()
  );

  Ort::Value wts_tensor_a = Ort::Value::CreateTensor<float>(
    memory_info_, wts_ptr, num_elements, weights_shape.data(),
    weights_shape.size()
  );

  Ort::Value output_tensor_1 = Ort::Value::CreateTensor<float>(
    memory_info_, output_1.Data<float>(), num_elements, input_shape.data(),
    input_shape.size()
  );

  invokeOp(context, {input_tensor_a, wts_tensor_a}, {output_tensor_1});

  if constexpr (std::is_same_v<Out, Ort::BFloat16_t>) {
    float_buffer_to_bfloat16(
      output_1.Data<float>(), batch_size * seq_length * hidden_size,
      (uint16_t*)output_data
    );
  } else if constexpr (std::is_same_v<Out, Ort::Float16_t>) {
    float_buffer_to_float16(
      output_1.Data<float>(), batch_size * seq_length * hidden_size,
      (uint16_t*)output_data
    );
  } else {
    static_assert(
      !sizeof(Out),
      "Unsupported output type. Only bfloat16 or float16 are supported."
    );
  }
}

#define INSTANTIATE(WtsType, OutType)                                         \
  template void OrtSimplifiedLayerNorm::execute<WtsType, OutType>(            \
    OutType*, Ort::BFloat16_t*, const std::vector<int64_t>&, WtsType*,        \
    const std::vector<int64_t>&, const RyzenMM::Allocator&, OrtKernelContext* \
  )

// Explicit template instantiations
INSTANTIATE(Ort::BFloat16_t, Ort::BFloat16_t);
INSTANTIATE(float, Ort::BFloat16_t);
INSTANTIATE(Ort::Float16_t, Ort::BFloat16_t);
INSTANTIATE(Ort::Float16_t, Ort::Float16_t);

}  // namespace ryzenai::onnx_utils
