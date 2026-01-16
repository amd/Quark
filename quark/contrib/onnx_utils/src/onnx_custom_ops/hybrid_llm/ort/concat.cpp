// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "concat.hpp"

#include <cassert>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtConcat::construct(const Ort::ConstKernelInfo& info, int64_t axis) {
  std::vector<Ort::OpAttr> attrs{};

  attrs.emplace_back("axis", &axis, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  createOp(
    info.Copy(), "Concat", "", 13,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16}}, std::move(attrs), 2, 1
  );
}

void OrtConcat::execute(
  const Ort::ConstValue& inp, const Ort::ConstValue& inp1,
  Ort::UnownedValue& out, OrtKernelContext* context
) {
  const auto inp_type = inp.GetTensorTypeAndShapeInfo().GetElementType();
  const auto inp1_type = inp1.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out_type = out.GetTensorTypeAndShapeInfo().GetElementType();

  if (!(inp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Concat: unsupported input type"};
  }
  if (!(inp1_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Concat: unsupported input1 type"};
  }
  if (!(out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Concat: unsupported output type"};
  }

  invokeOp(context, {inp, inp1}, {out});
}

}  // namespace ryzenai::onnx_utils
