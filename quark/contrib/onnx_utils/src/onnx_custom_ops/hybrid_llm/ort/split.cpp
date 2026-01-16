// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "split.hpp"

namespace ryzenai::onnx_utils {

void OrtSplit::construct(
  const Ort::ConstKernelInfo& info, int64_t axis, int64_t num_outputs
) {
  std::vector<Ort::OpAttr> attrs{};

  attrs.emplace_back("axis", &axis, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  // 'num_outputs' attr and 'split' input should not be provided at same time.
  /*attrs.emplace_back(
    "num_outputs", &num_outputs, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );*/

  createOp(
    info.Copy(), "Split", "", 18, {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16}},
    std::move(attrs), 2, num_outputs
  );
}

void OrtSplit::execute(
  const Ort::ConstValue& inp, const Ort::ConstValue& split_sizes,
  Ort::UnownedValue& out0, Ort::UnownedValue& out1, Ort::UnownedValue& out2,
  OrtKernelContext* context
) {
  const auto inp_type = inp.GetTensorTypeAndShapeInfo().GetElementType();
  const auto split_sizes_type =
    split_sizes.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out0_type = out0.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out1_type = out1.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out2_type = out2.GetTensorTypeAndShapeInfo().GetElementType();
  if (!(inp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Split: unsupported input type"};
  }
  if (!(split_sizes_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    throw std::runtime_error{"Split: unsupported input split type"};
  }
  if (!(out0_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Split: unsupported out0 type"};
  }
  if (!(out1_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Split: unsupported out1 type"};
  }
  if (!(out2_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Split: unsupported out2 type"};
  }
  invokeOp(context, {inp, split_sizes}, {out0, out1, out2});
}

}  // namespace ryzenai::onnx_utils
