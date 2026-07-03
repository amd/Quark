//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_bfp.h"

#include <stdint.h>

#include <cmath>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfp/bfp.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#endif

Buffer::Buffer(size_t size) {
#ifdef USE_CUDA
  cudaMalloc(&data_, size);
#else
  data_ = new char[size];
#endif
}

void* Buffer::get_data_ptr() { return data_; }

void Buffer::release() {
#ifdef USE_CUDA
  cudaFree(data_);
#else
  delete[] data_;
#endif
}

int64_t HandleNegativeBlockAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

BFPFixNeuronKernel::BFPFixNeuronKernel(
  const OrtApi& ort_api, const OrtKernelInfo* k_info, std::string bfp_method,
  int64_t axis, int64_t bit_width, int64_t block_size, int64_t rounding_mode,
  int64_t sub_block_size, int64_t sub_block_shift_bits,
  int64_t convert_to_bfloat_before_bfp, int64_t use_compiler_version_cpu_kernel
)
  : ort_(ort_api),
    bfp_method_(bfp_method),
    axis_(axis),
    bit_width_(bit_width),
    block_size_(block_size),
    rounding_mode_(rounding_mode),
    sub_block_size_(sub_block_size),
    sub_block_shift_bits_(sub_block_shift_bits),
    convert_to_bfloat_before_bfp_(convert_to_bfloat_before_bfp),
    use_compiler_version_cpu_kernel_(use_compiler_version_cpu_kernel) {
  Ort::ConstKernelInfo info{k_info};
  info_copy_ = info.Copy();
}

Ort::Value BFPFixNeuronKernel::do_bfp(Ort::Value& input) {
  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();
  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);
  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count,
    dimensions.data(), dimensions.size()
  );

  if (bfp_method_ == "to_bfp") {
    to_bfp(
      input, bit_width_, block_size_, rounding_mode_, output,
      use_compiler_version_cpu_kernel_
    );
    //  Here we try to be compatible with the old method name
    //  'to_bfp_prime_shared'
  } else if (bfp_method_ == "to_bfp_prime" ||
             bfp_method_ == "to_bfp_prime_shared") {
    to_bfp_prime(
      input, bit_width_, block_size_, sub_block_size_, sub_block_shift_bits_,
      rounding_mode_, output
    );
  } else {
    throw std::invalid_argument(
      "Invalid bfp_method, valid bfp_method should be one of [\"to_bfp\", "
      "\"to_bfp_prime\"], current bfp_method is " +
      bfp_method_
    );
  }

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

Ort::Value BFPFixNeuronKernel::cast_to_fp32(
  OrtKernelContext* context, Ort::Value& input
) {
  if (!op_cast_to_fp32_init_) {
    int64_t to = static_cast<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Ort::OpAttr attr =
      Ort::OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    const char* type_constraint_names[] = {"T1", "T2"};
    ONNXTensorElementDataType type_constraint_values[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    };
    op_cast_to_fp32_ = Ort::Op::Create(
      info_copy_, "Cast", "", 13, type_constraint_names, type_constraint_values,
      2, &attr, 1, 1, 1
    );
    op_cast_to_fp32_init_ = true;
  }

  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();
  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * sizeof(float));
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count,
    dimensions.data(), dimensions.size()
  );

  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};
  op_cast_to_fp32_.Invoke(context, inputs, 1, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

Ort::Value BFPFixNeuronKernel::cast_to_fp16(
  OrtKernelContext* context, Ort::Value& input
) {
  if (!op_cast_to_fp16_init_) {
    int64_t to = static_cast<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    Ort::OpAttr attr =
      Ort::OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    const char* type_constraint_names[] = {"T1", "T2"};
    ONNXTensorElementDataType type_constraint_values[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    };
    op_cast_to_fp16_ = Ort::Op::Create(
      info_copy_, "Cast", "", 13, type_constraint_names, type_constraint_values,
      2, &attr, 1, 1, 1
    );
    op_cast_to_fp16_init_ = true;
  }

  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();
  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 2);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor(
    input.GetTensorMemoryInfo(), b.get_data_ptr(), element_count * 2,
    dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
  );

  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};
  op_cast_to_fp16_.Invoke(context, inputs, 1, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void BFPFixNeuronKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  auto input_value = ctx.GetInput(0);
  std::vector<int64_t> dimensions =
    input_value.GetTensorTypeAndShapeInfo().GetShape();

  auto input_dtype = input_value.GetTensorTypeAndShapeInfo().GetElementType();
  is_fp16_input_ = (input_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  Ort::Value fp32_input{nullptr};
  if (is_fp16_input_) {
    size_t elem_count =
      input_value.GetTensorTypeAndShapeInfo().GetElementCount();
    auto fp16_tensor = Ort::Value::CreateTensor(
      input_value.GetTensorMemoryInfo(),
      const_cast<void*>(
        static_cast<const void*>(input_value.GetTensorRawData())
      ),
      elem_count * 2, dimensions.data(), dimensions.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    );
    fp32_input = cast_to_fp32(context, fp16_tensor);
  }

  auto input_tensor = [&]() -> Ort::Value {
    if (is_fp16_input_) {
      return Ort::Value::CreateTensor<float>(
        fp32_input.GetTensorMemoryInfo(),
        const_cast<float*>(fp32_input.GetTensorData<float>()),
        fp32_input.GetTensorTypeAndShapeInfo().GetElementCount(),
        dimensions.data(), dimensions.size()
      );
    } else {
      return Ort::Value::CreateTensor<float>(
        input_value.GetTensorMemoryInfo(),
        const_cast<float*>(input_value.GetTensorData<float>()),
        input_value.GetTensorTypeAndShapeInfo().GetElementCount(),
        dimensions.data(), dimensions.size()
      );
    }
  }();

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif

  if (convert_to_bfloat_before_bfp_ != 0) {
    to_bfloat(input_tensor);
  }

  Ort::Value ret{nullptr};
  if (input_tensor.GetTensorTypeAndShapeInfo().GetElementCount() == 1) {
    if (is_fp16_input_) {
      ret = Ort::Value::CreateTensor<float>(
        fp32_input.GetTensorMemoryInfo(),
        const_cast<float*>(fp32_input.GetTensorData<float>()),
        fp32_input.GetTensorTypeAndShapeInfo().GetElementCount(),
        dimensions.data(), dimensions.size()
      );
    } else {
      ret = Ort::Value::CreateTensor<float>(
        input_value.GetTensorMemoryInfo(),
        const_cast<float*>(input_value.GetTensorData<float>()),
        input_value.GetTensorTypeAndShapeInfo().GetElementCount(),
        dimensions.data(), dimensions.size()
      );
    }
  } else if (dimensions.size() == 1) {
    auto padded_tensor = pad(context, input_tensor, block_size_);
    auto bfp_tensor = do_bfp(padded_tensor);
    ret = slice(context, bfp_tensor, dimensions[0]);
  } else {
    int64_t tensor_rank = static_cast<int64_t>(dimensions.size());
    if ((axis_ >= -tensor_rank && axis_ <= tensor_rank - 1) == false)
      ORT_CXX_API_THROW(
        "The axis of BFP should be in the valid range.",
        OrtErrorCode::ORT_INVALID_GRAPH
      );
    const int64_t axis_no_neg = HandleNegativeBlockAxis(axis_, tensor_rank);

    auto transposed_tensor =
      transpose(context, input_tensor, axis_no_neg, dimensions.size() - 1);
    auto padded_tensor = pad(context, transposed_tensor, block_size_);
    auto bfp_tensor = do_bfp(padded_tensor);
    auto sliced_tensor = slice(context, bfp_tensor, dimensions[axis_no_neg]);
    ret = transpose(context, sliced_tensor, axis_no_neg, dimensions.size() - 1);
  }
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif

  if (is_fp16_input_) {
    auto fp16_ret = cast_to_fp16(context, ret);
    auto output =
      ctx.GetOutput(0, fp16_ret.GetTensorTypeAndShapeInfo().GetShape());
#ifdef USE_CUDA
    cudaMemcpy(
      output.GetTensorMutableRawData(), fp16_ret.GetTensorMutableRawData(),
      output.GetTensorTypeAndShapeInfo().GetElementCount() * 2,
      cudaMemcpyKind::cudaMemcpyDeviceToDevice
    );
#else
    memcpy(
      output.GetTensorMutableRawData(), fp16_ret.GetTensorMutableRawData(),
      output.GetTensorTypeAndShapeInfo().GetElementCount() * 2
    );
#endif
  } else {
    auto output = ctx.GetOutput(0, ret.GetTensorTypeAndShapeInfo().GetShape());
#ifdef USE_CUDA
    cudaMemcpy(
      output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(),
      output.GetTensorTypeAndShapeInfo().GetElementCount() * 4,
      cudaMemcpyKind::cudaMemcpyDeviceToDevice
    );
#else
    memcpy(
      output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(),
      output.GetTensorTypeAndShapeInfo().GetElementCount() * 4
    );
#endif
  }

  for (Buffer b : tmp_buffers_) {
    b.release();
  }
  tmp_buffers_.clear();
}

BFPFixNeuronKernel::~BFPFixNeuronKernel() {}

void BFPFixNeuronKernel::create_pad_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
  };
#ifdef USE_CUDA
  int64_t opset = 13;  // Op of this opset have a CUDA implementation
#else
  int64_t opset = 18;
#endif

  op_pad_ = Ort::Op::Create(
    info_copy_, "Pad", "", opset, add_type_constraint_names,
    add_type_constraint_values, 3, nullptr, 0, 3, 1
  );
  op_pad_init_ = true;
}

Ort::Value BFPFixNeuronKernel::pad(
  OrtKernelContext* context, Ort::Value& input, int block_size
) {
  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();
  if (!op_pad_init_) {
    create_pad_op();
  }
  int channels_to_pad =
    dimensions[dimensions.size() - 1] % block_size == 0
      ? 0
      : block_size - dimensions[dimensions.size() - 1] % block_size;
  int64_t pad_size = 2 * dimensions.size();
  std::vector<int64_t> pad(pad_size);
  for (int i = 0; i < pad_size - 1; i++) pad[i] = 0;
  pad[2 * dimensions.size() - 1] = channels_to_pad;
  auto pad_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &pad[0], pad_size, &pad_size, 1
  );

  int64_t const_value_size = 1;
  float const_value = 0;
  auto const_value_tensor = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), &const_value, 1, &const_value_size, 1
  );

  dimensions[dimensions.size() - 1] += channels_to_pad;
  size_t element_count = 1;
  for (int i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count,
    dimensions.data(), dimensions.size()
  );

  const OrtValue* inputs[3] = {input, pad_tensor, const_value_tensor};
  OrtValue* outputs[1] = {output};
  op_pad_.Invoke(context, inputs, 3, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void BFPFixNeuronKernel::create_transpose_op(
  size_t num_dims, int from, int to
) {
  const char* add_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[1] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
  };
  std::vector<int64_t> perm(num_dims);
  for (int i = 0; i < num_dims; i++) {
    perm[i] = i;
  }
  int64_t tmp = perm[from];
  perm[from] = perm[to];
  perm[to] = tmp;
  auto perm_attr =
    Ort::OpAttr("perm", &perm[0], num_dims, OrtOpAttrType::ORT_OP_ATTR_INTS);
  Ort::OpAttr attrs[1] = {std::move(perm_attr)};

  op_transpose_ = Ort::Op::Create(
    info_copy_, "Transpose", "", 13, add_type_constraint_names,
    add_type_constraint_values, 1, attrs, 1, 1, 1
  );
  op_transpose_init_ = true;
}

Ort::Value BFPFixNeuronKernel::transpose(
  OrtKernelContext* context, Ort::Value& input, int from, int to
) {
  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();
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
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count,
    dimensions.data(), dimensions.size()
  );

  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};
  op_transpose_.Invoke(context, inputs, 1, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}

void BFPFixNeuronKernel::create_slice_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
  };
  op_slice_ = Ort::Op::Create(
    info_copy_, "Slice", "", 13, add_type_constraint_names,
    add_type_constraint_values, 5, nullptr, 0, 5, 1
  );
  op_slice_init_ = true;
}

Ort::Value BFPFixNeuronKernel::slice(
  OrtKernelContext* context, Ort::Value& input, size_t last_dim
) {
  if (!op_slice_init_) {
    create_slice_op();
  }
  std::vector<int64_t> dimensions =
    input.GetTensorTypeAndShapeInfo().GetShape();

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
    input.GetTensorMemoryInfo(), &start[0], num_dims, &num_dims, 1
  );
  auto end_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &dimensions[0], num_dims, &num_dims, 1
  );
  auto axes_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &axes[0], num_dims, &num_dims, 1
  );
  auto steps_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &steps[0], num_dims, &num_dims, 1
  );

  size_t element_count = 1;
  for (auto i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count,
    dimensions.data(), dimensions.size()
  );

  const OrtValue* inputs[] = {
    input, start_tensor, end_tensor, axes_tensor, steps_tensor
  };
  OrtValue* outputs[] = {output};
  op_slice_.Invoke(context, inputs, 5, outputs, 1);
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  return output;
}
