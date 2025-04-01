//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "custom_op_bfp.h"
#include "custom_op_mx.h"

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "mx/mx.h"

#ifdef _WIN32
#include <stdint.h>
#endif

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif


void ParseElementDataTypeString(std::string& dtype,
                         std::vector<int>& bits,
			 std::vector<float>& range) {
    int ebits = 0; // bit numbers of exponent
    int mbits = 0; // bit numbers of mantissa
    int emax = 0;  // max exponent value

    float max_norm = 0;
    float min_norm = 0;

    if (dtype == "fp8_e5m2") {
        ebits = 5;
        mbits = 2;
	emax = 15;  // float8e5m2 has Inf

	max_norm = 57344;
	min_norm = -57344;
    }
    else if (dtype == "fp8_e4m3") {
        ebits = 4;
        mbits = 3;
	emax = 8;

	max_norm = 448;
	min_norm = -448;
    }
    else if (dtype == "fp6_e3m2") {
        ebits = 3;
        mbits = 2;
	emax = 4;

	max_norm = 28.0;
	min_norm = -28.0;
    }
    else if (dtype == "fp6_e2m3") {
        ebits = 2;
        mbits = 3;
	emax = 2;

	max_norm = 7.5;
	min_norm = -7.5;
    }
    else if (dtype == "fp4_e2m1") {
        ebits = 2;
        mbits = 1;
	emax = 2;

	max_norm = 6.0;
	min_norm = -6.0;
    }
    else if (dtype == "int8") {
        ebits = 0;
        mbits = 8;
	emax = 0;  // int8 has a implicit scale 2^-6

	max_norm = 127;
	min_norm = -128;
    }

    bits.clear();
    bits.push_back(ebits);
    bits.push_back(mbits);
    bits.push_back(emax);

    range.clear();
    range.push_back(max_norm);
    range.push_back(min_norm);
}

MXFixNeuronKernel::MXFixNeuronKernel(
    const OrtApi& ort_api, 
    const OrtKernelInfo* k_info,
    std::string& scale_dtype,
    std::string& element_dtype,
    int64_t axis,
    int64_t block_size, 
    int64_t rounding_mode) : 
      ort_(ort_api), 
      scale_dtype_(scale_dtype),
      element_dtype_(element_dtype),
      axis_(axis),
      block_size_(block_size),
      rounding_mode_(rounding_mode) {
  Ort::ConstKernelInfo info{k_info};
  info_copy_ = info.Copy();
}

Ort::Value MXFixNeuronKernel::do_mx(Ort::Value &input) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);
  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());
  to_mx(input, scale_dtype_, element_dtype_,
        block_size_, rounding_mode_, output);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void MXFixNeuronKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  auto input_value = ctx.GetInput(0);
  std::vector<int64_t> dimensions = input_value.GetTensorTypeAndShapeInfo().GetShape();
  auto input_tensor = Ort::Value::CreateTensor<float>(
    input_value.GetTensorMemoryInfo(),
    const_cast<float*>(input_value.GetTensorData<float>()), 
    input_value.GetTensorTypeAndShapeInfo().GetElementCount(), 
    dimensions.data(), dimensions.size());

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  Ort::Value ret{nullptr};
  if (input_tensor.GetTensorTypeAndShapeInfo().GetElementCount() == 1) {
    ret = Ort::Value::CreateTensor<float>(
        input_value.GetTensorMemoryInfo(),
        const_cast<float*>(input_value.GetTensorData<float>()), 
        input_value.GetTensorTypeAndShapeInfo().GetElementCount(), 
        dimensions.data(), dimensions.size());
  } else if (dimensions.size() == 1) {
    auto padded_tensor = pad(context, input_tensor, block_size_);
    auto mx_tensor = do_mx(padded_tensor);
    ret = slice(context, mx_tensor, dimensions[0]);
  } else {
    auto transposed_tensor = transpose(context, input_tensor, axis_, dimensions.size() - 1);
    auto padded_tensor = pad(context, transposed_tensor, block_size_);
    auto mx_tensor = do_mx(padded_tensor);
    // auto mx_tensor = std::move(padded_tensor);
    auto sliced_tensor = slice(context, mx_tensor, dimensions[axis_]);
    ret = transpose(context, sliced_tensor, axis_, dimensions.size() - 1);
  }
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  auto output = ctx.GetOutput(0, ret.GetTensorTypeAndShapeInfo().GetShape());

#ifdef USE_CUDA
  cudaMemcpy(
    output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(), 
    output.GetTensorTypeAndShapeInfo().GetElementCount() * 4, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
#else
  memcpy(output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(), 
    output.GetTensorTypeAndShapeInfo().GetElementCount() * 4);
#endif

  for (Buffer b : tmp_buffers_) {
    b.release();
  }
  tmp_buffers_.clear();
}

MXFixNeuronKernel::~MXFixNeuronKernel() {
}

void MXFixNeuronKernel::create_pad_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  std::string mode_str = "constant";
  auto mode = Ort::OpAttr("mode", mode_str.c_str(), 1, OrtOpAttrType::ORT_OP_ATTR_STRING);
  Ort::OpAttr attrs[1] = {std::move(mode)};
  op_pad_ = Ort::Op::Create(info_copy_, "Pad", "", 18,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            3, attrs, 1, 3, 1);
  op_pad_init_ = true;
}

Ort::Value MXFixNeuronKernel::pad(OrtKernelContext* context, Ort::Value &input, int block_size) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  if (!op_pad_init_) {
    create_pad_op();
  }
  int channels_to_pad = dimensions[dimensions.size() - 1] % block_size == 0 ? 
    0 : block_size - dimensions[dimensions.size() - 1] % block_size;
  int64_t pad_size = 2 * dimensions.size();
  std::vector<int64_t> pad(pad_size);
  for (int i = 0; i < pad_size - 1; i++) pad[i] = 0;
  pad[2 * dimensions.size() - 1] = channels_to_pad;
  auto pad_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &pad[0], pad_size, &pad_size, 1);
  
  int64_t const_value_size = 1;
  float const_value = 0;
  auto const_value_tensor = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), &const_value, 1, &const_value_size, 1);
  
  dimensions[dimensions.size() - 1] += channels_to_pad;
  size_t element_count = 1;
  for (int i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());
  
  const OrtValue* inputs[3] = {input, pad_tensor, const_value_tensor};
  OrtValue* outputs[1] = {output};
  op_pad_.Invoke(context, inputs, 3, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void MXFixNeuronKernel::create_transpose_op(size_t num_dims, int from, int to) {
  const char* add_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  std::vector<int64_t> perm(num_dims);
  for (int i = 0; i < num_dims; i++) {
    perm[i] = i;
  }
  int64_t tmp = perm[from];
  perm[from] = perm[to];
  perm[to] = tmp;
  auto perm_attr = Ort::OpAttr("perm", &perm[0], num_dims, OrtOpAttrType::ORT_OP_ATTR_INTS);
  Ort::OpAttr attrs[1] = {std::move(perm_attr)};

  op_transpose_ = Ort::Op::Create(info_copy_, "Transpose", "", 13,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            1, attrs, 1, 1, 1);
  op_transpose_init_ = true;
}

Ort::Value MXFixNeuronKernel::transpose(OrtKernelContext* context, Ort::Value &input, int from, int to) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  if (!op_transpose_init_) {
    create_transpose_op(dimensions.size(), from, to);
  }
  int64_t tmp = dimensions[from];
  dimensions[from] = dimensions[to];
  dimensions[to] = tmp;

  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());

  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};
  op_transpose_.Invoke(context, inputs, 1, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void MXFixNeuronKernel::create_slice_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  op_slice_ = Ort::Op::Create(info_copy_, "Slice", "", 13,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            5, nullptr, 0, 5, 1);
  op_slice_init_ = true;
}

Ort::Value MXFixNeuronKernel::slice(OrtKernelContext* context, Ort::Value &input, size_t last_dim) {
  if (!op_slice_init_) {
    create_slice_op();
  }
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();

  int64_t num_dims = dimensions.size();
  dimensions[num_dims - 1] = last_dim;
  std::vector<int64_t> start(num_dims);
  std::vector<int64_t> axes(num_dims);
  std::vector<int64_t> steps(num_dims);
  for (int i = 0; i < num_dims; i++) {
    start[i] = 0;
    axes[i] = i;
    steps[i] = 1;
  }
  auto start_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &start[0], num_dims, &num_dims, 1);
  auto end_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &dimensions[0], num_dims, &num_dims, 1);
  auto axes_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &axes[0], num_dims, &num_dims, 1);
  auto steps_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &steps[0], num_dims, &num_dims, 1);
  
  size_t element_count = 1;
  for (auto i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());

  const OrtValue* inputs[] = {input, start_tensor, end_tensor, axes_tensor, steps_tensor};
  OrtValue* outputs[] = {output};
  op_slice_.Invoke(context, inputs, 5, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}
