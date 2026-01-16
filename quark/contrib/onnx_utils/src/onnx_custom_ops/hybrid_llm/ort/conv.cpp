// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "conv.hpp"

#include <cassert>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtConv::construct(
  const Ort::ConstKernelInfo& info, int64_t group,
  std::vector<int64_t> kernel_shape, std::vector<int64_t> pads
) {
  group_ = group;
  kernel_shape_ = std::move(kernel_shape);
  pads_ = std::move(pads);

  std::vector<Ort::OpAttr> attrs;

  attrs.emplace_back("group", &group_, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  attrs.emplace_back(
    "kernel_shape", kernel_shape_.data(), kernel_shape_.size(),
    OrtOpAttrType::ORT_OP_ATTR_INTS
  );
  attrs.emplace_back(
    "pads", pads_.data(), pads_.size(), OrtOpAttrType::ORT_OP_ATTR_INTS
  );

  createOp(
    info.Copy(), "Conv", "", 22, {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}},
    std::move(attrs), 2, 1
  );
}

void OrtConv::execute(
  Ort::ConstValue inp, Ort::ConstValue wts, Ort::UnownedValue out,
  OrtKernelContext* context
) {
  const auto inp_type = inp.GetTensorTypeAndShapeInfo().GetElementType();
  const auto wts_type = wts.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out_type = out.GetTensorTypeAndShapeInfo().GetElementType();

  if (!(inp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    throw std::runtime_error{"Conv: unsupported input type"};
  }

  if (!(wts_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    throw std::runtime_error{"Conv: unsupported weights type"};
  }

  if (!(out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    throw std::runtime_error{"Conv: unsupported output type"};
  }

  invokeOp(context, {inp, wts}, {out});
}

}  // namespace ryzenai::onnx_utils
