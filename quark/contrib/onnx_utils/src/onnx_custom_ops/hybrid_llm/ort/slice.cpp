// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "slice.hpp"

#include <cassert>
#include <numeric>

namespace ryzenai::onnx_utils {

void OrtSlice::construct(const Ort::ConstKernelInfo& info) {
  std::vector<Ort::OpAttr> attrs{};

  createOp(
    info.Copy(), "Slice", "", 13,  // 11,
    {{"T", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16},
     {"Tind", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}},
    std::move(attrs), 4, 1
  );

  /*onnx::TensorProto* steps = graph->add_initializer();
  steps->set_name("steps");
  steps->set_data_type(onnx::TensorProto_DataType_INT64);
  steps->add_dims(1);
  steps->add_int64_data(1);*/
}

void OrtSlice::execute(
  Ort::ConstValue inp, Ort::ConstValue starts, Ort::ConstValue ends,
  Ort::ConstValue axes, Ort::UnownedValue out, OrtKernelContext* context
) {
  const auto inp_type = inp.GetTensorTypeAndShapeInfo().GetElementType();
  const auto starts_type = starts.GetTensorTypeAndShapeInfo().GetElementType();
  const auto ends_type = ends.GetTensorTypeAndShapeInfo().GetElementType();
  const auto axes_type = axes.GetTensorTypeAndShapeInfo().GetElementType();
  const auto out_type = out.GetTensorTypeAndShapeInfo().GetElementType();

  if (!(inp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Slice: unsupported input type"};
  }
  if (!(starts_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    throw std::runtime_error{"Slice: unsupported starts type"};
  }
  if (!(ends_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    throw std::runtime_error{"Slice: unsupported ends type"};
  }
  if (!(axes_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
    throw std::runtime_error{"Slice: unsupported axes type"};
  }
  if (!(out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
    throw std::runtime_error{"Slice: unsupported output type"};
  }

  invokeOp(context, {inp, starts, ends, axes}, {out});
}

}  // namespace ryzenai::onnx_utils
