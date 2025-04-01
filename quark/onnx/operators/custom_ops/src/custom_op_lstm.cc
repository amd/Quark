//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_lstm.h"

#define LEN_SEQ    80
#define LEN_INPUT  64
#define LEN_HIDDEN 128
#define YLEN (LEN_SEQ    * LEN_HIDDEN    )
#define XLEN (LEN_SEQ    * LEN_INPUT     )
#define WLEN (LEN_INPUT  * LEN_HIDDEN * 4)
#define RLEN (LEN_HIDDEN * LEN_HIDDEN * 4)
#define BLEN (LEN_HIDDEN * 4             )
#include "lstm/cpu/lstm.h"

#include "qdq/cpu/quantize_linear.h"

#include <vector>
#include <cmath>

namespace vai_q {

KernelCustomLSTM::KernelCustomLSTM(const OrtApi& api,
		                   const OrtKernelInfo* info) : api_(api) {
  Ort::ConstKernelInfo const_info{info};
  info_ = const_info.Copy();

  size_t size = 16;
  direction_.resize(size);
  auto status = api_.KernelInfoGetAttribute_string(info_, "direction", &direction_[0], &size);
#if 0
  if (direction_ != "bidirectional") {
    ORT_CXX_API_THROW("KernelCustomLSTM supports attribute direction = 'bidirectional' only.",
                      OrtErrorCode::ORT_NOT_IMPLEMENTED);
  }
  num_directions_ = (direction_ == "bidirectional") ? 2 : 1;
#else
  // Supports bidirectional only
  num_directions_ = 2;
#endif

  status = api_.KernelInfoGetAttribute_int64(info_, "hidden_size", &hidden_size_);
  if (status != nullptr || hidden_size_ <= 0) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'hidden_size'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }

  status = api_.KernelInfoGetAttribute_int64(info_, "input_forget", &input_forget_);
  if (input_forget_ != 0) {
    ORT_CXX_API_THROW("KernelCustomLSTM supports attribute 'input_forget' = 0 only.",
                      OrtErrorCode::ORT_NOT_IMPLEMENTED);
  }

  status = api_.KernelInfoGetAttribute_int64(info_, "layout", &layout_);
  if (layout_ != 0) {
    ORT_CXX_API_THROW("KernelCustomLSTM supports attribute 'layout' = 0 only.",
                      OrtErrorCode::ORT_NOT_IMPLEMENTED);
  }

  status = api_.KernelInfoGetAttribute_float(info_, "x_scale", &x_scale_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'x_scale'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_int64(info_, "x_zero_point", &x_zero_point_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'x_zero_point'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_float(info_, "w_scale", &w_scale_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'w_scale'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_int64(info_, "w_zero_point", &w_zero_point_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'w_zero_point'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_float(info_, "r_scale", &r_scale_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'r_scale'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_int64(info_, "r_zero_point", &r_zero_point_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'r_zero_point'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_float(info_, "b_scale", &b_scale_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'b_scale'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_int64(info_, "b_zero_point", &b_zero_point_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'b_zero_point'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_float(info_, "y_scale", &y_scale_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'y_scale'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
  status = api_.KernelInfoGetAttribute_int64(info_, "y_zero_point", &y_zero_point_);
  if (status != nullptr) {
    ORT_CXX_API_THROW("KernelCustomLSTM needs attribute 'y_zero_point'.",
                      OrtErrorCode::ORT_INVALID_GRAPH);
  }
};

KernelCustomLSTM::~KernelCustomLSTM() {
};


QUANTIZE_LINEAR_APPLY(float)

template <typename T>
static void QuantizeData(Ort::ConstValue& input, float scale, T zp, T* &quantized_data) {
  ONNXTensorElementDataType type = input.GetTensorTypeAndShapeInfo().GetElementType();
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      ORT_CXX_API_THROW("KernelCustomLSTM supports float input only.",
                        OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  const float* data = input.GetTensorData<float>();
  int data_len = input.GetTensorTypeAndShapeInfo().GetElementCount();
  quantized_data = (T*)malloc(data_len * sizeof(T));
  if (!quantized_data) {
      ORT_CXX_API_THROW("KernelCustomLSTM supports float input only.",
                        OrtErrorCode::ORT_ENGINE_ERROR);
  }

  // Only supports per-tensor quantization
  int block_count = 1;
  int broadcast_dim = 1;
  int block_size = data_len;

  QuantizeLinearApply<T>().op(nullptr, block_count, broadcast_dim, block_size,
                              data, &scale, quantized_data, &zp);
};

DEQUANTIZE_LINEAR_APPLY(float)

template <typename T>
static void DequantizeData(T* quantized_data, int data_len, float scale, T zp, float* data) {
  // Only supports per-tensor quantization
  int block_count = 1;
  int broadcast_dim = 1;
  int block_size = data_len;

  DequantizeLinearApply<T>().op(nullptr, block_count, broadcast_dim, block_size,
                              quantized_data, &scale, data, &zp);
};

void KernelCustomLSTM::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  // Quantize X
  auto x = ctx.GetInput(0);
  int x_data_len = x.GetTensorTypeAndShapeInfo().GetElementCount();

  uint16_t* x_quantized_data = nullptr;
  QuantizeData<uint16_t>(x, x_scale_, (uint16_t)x_zero_point_, x_quantized_data);

  // Now update the seq_len_ and input_size_
  std::vector<int64_t> input_shape = x.GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() < 3) {
      ORT_CXX_API_THROW("KernelCustomLSTM supports 3 dims at least.",
                        OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  seq_len_ = (layout_ == 1) ? input_shape[1] : input_shape[0];
  input_size_ = input_shape[2];

  // Quantize W
  auto w = ctx.GetInput(1);
  int w_data_len = w.GetTensorTypeAndShapeInfo().GetElementCount();

  uint16_t* w_quantized_data = nullptr;
  QuantizeData<uint16_t>(w, w_scale_, (uint16_t)w_zero_point_, w_quantized_data);

  // Quantize R
  auto r = ctx.GetInput(2);
  int r_data_len = r.GetTensorTypeAndShapeInfo().GetElementCount();

  uint16_t* r_quantized_data = nullptr;
  QuantizeData<uint16_t>(r, r_scale_, (uint16_t)r_zero_point_, r_quantized_data);

  // Quantize B
  auto b = ctx.GetInput(3);
  int b_data_len = b.GetTensorTypeAndShapeInfo().GetElementCount();

  uint16_t* b_quantized_data = nullptr;
  QuantizeData<uint16_t>(b, b_scale_, (uint16_t)b_zero_point_, b_quantized_data);

  // Execute lstm
  std::vector<int64_t> output_shape = {input_shape[0], num_directions_, input_shape[1], hidden_size_};
  auto y = ctx.GetOutput(0, output_shape);

  float* y_data = y.GetTensorMutableData<float>();
  int y_data_len = y.GetTensorTypeAndShapeInfo().GetElementCount();
  uint16_t* y_quantized_data = (uint16_t*)malloc(y_data_len * sizeof(uint16_t));
  if (!y_quantized_data) {
      ORT_CXX_API_THROW("KernelCustomLSTM supports float input only.",
                        OrtErrorCode::ORT_ENGINE_ERROR);
  }

  lstm(y_quantized_data, x_quantized_data, w_quantized_data,
       r_quantized_data, b_quantized_data,
       x_scale_, w_scale_, r_scale_, b_scale_, y_scale_,
       (unsigned short) x_zero_point_,
       (unsigned short) w_zero_point_,
       (unsigned short) r_zero_point_,
       (unsigned short) b_zero_point_,
       (unsigned short) y_zero_point_,
       (int)seq_len_, (int)input_size_, (int)hidden_size_);

  // Output float
  DequantizeData<uint16_t>(y_quantized_data, y_data_len, y_scale_, (uint16_t)y_zero_point_, y_data);

  // Free the memories
  if (x_quantized_data) {
    free(x_quantized_data);
    x_quantized_data = nullptr;
  }
  if (w_quantized_data) {
    free(w_quantized_data);
    w_quantized_data = nullptr;
  }
  if (r_quantized_data) {
    free(r_quantized_data);
    r_quantized_data = nullptr;
  }
  if (b_quantized_data) {
    free(b_quantized_data);
    b_quantized_data = nullptr;
  }
  if (y_quantized_data) {
    free(y_quantized_data);
    y_quantized_data = nullptr;
  }
};

}  // namespace vai_q
