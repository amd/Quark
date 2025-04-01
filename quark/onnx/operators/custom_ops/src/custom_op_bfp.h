//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Buffer {
  Buffer(size_t size);
  void* get_data_ptr();
  void release();

private:
  char* data_;
};


struct BFPFixNeuronKernel {
  BFPFixNeuronKernel(
    const OrtApi& ort_api, 
    const OrtKernelInfo* info, 
    std::string bfp_method,
    int64_t axis,
    int64_t bit_width, 
    int64_t block_size, 
    int64_t rounding_mode,
    int64_t sub_block_size,
    int64_t sub_block_shift_bits,
    int64_t convert_to_bfloat_before_bfp,
    int64_t use_compiler_version_cpu_kernel);
  ~BFPFixNeuronKernel();
  void Compute(OrtKernelContext* context);
#if ORT_API_VERSION >= 17
  // This is for adapting to onnxruntime_cxx_api.h in ORT 1.17.0 (and higher)
  OrtStatusPtr ComputeV2(OrtKernelContext* context) { return nullptr; }
#endif

  Ort::Value transpose(OrtKernelContext* context, Ort::Value &input, int from, int to);

  void create_transpose_op(size_t num_dims, int from, int to);

  Ort::Value pad(OrtKernelContext* context, Ort::Value &input, int block_size);

  void create_pad_op();

  Ort::Value slice(OrtKernelContext* context, Ort::Value &input, size_t last_dim);

  void create_slice_op();

  Ort::Value do_bfp(Ort::Value &input);

 private:

  const OrtApi& ort_;
  Ort::KernelInfo info_copy_{nullptr};
  Ort::Op op_transpose_{nullptr};
  bool op_transpose_init_ = false;
  Ort::Op op_pad_{nullptr};
  bool op_pad_init_ = false;
  Ort::Op op_slice_{nullptr};
  bool op_slice_init_ = false;
  std::vector<Buffer> tmp_buffers_;

  std::string bfp_method_ = "to_bfp";
  int64_t axis_ = 1;
  int64_t bit_width_ = 16;
  int64_t block_size_ = 8;
  int64_t rounding_mode_ = 0;
  int64_t sub_block_size_ = 2;
  int64_t sub_block_shift_bits_ = 1;
  int64_t convert_to_bfloat_before_bfp_ = 0;
  int64_t use_compiler_version_cpu_kernel_ = 0;
};

struct BFPFixNeuron : Ort::CustomOpBase<BFPFixNeuron, BFPFixNeuronKernel> {
  explicit BFPFixNeuron() {}

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
    Ort::InitApi(&api);

    std::string bfp_method;
    int64_t axis;
    int64_t bit_width;
    int64_t block_size;
    int64_t rounding_mode;
    int64_t sub_block_size;
    int64_t sub_block_shift_bits;
    int64_t convert_to_bfloat_before_bfp;
    int64_t use_compiler_version_cpu_kernel;

    // It has two steps to get attribute with string datatype
    // Step1. Get the string length
    const char* attribute_name = "bfp_method";
    size_t str_len;
    auto status = api.KernelInfoGetAttribute_string(info, attribute_name, nullptr, &str_len);
    // Step2. Get the string content
    char* str_ptr = (char*)malloc(str_len);
    status = api.KernelInfoGetAttribute_string(info, attribute_name, str_ptr, &str_len);
    bfp_method = (status != nullptr) ? "to_bfp" : str_ptr;
    free(str_ptr); // Free the memory

    status = api.KernelInfoGetAttribute_int64(info, "axis", &axis);
    if (status != nullptr) axis = 1;
    status = api.KernelInfoGetAttribute_int64(info, "bit_width", &bit_width);
    if (status != nullptr) bit_width = 16;
    status = api.KernelInfoGetAttribute_int64(info, "block_size", &block_size);
    if (status != nullptr) block_size = 8;
    status = api.KernelInfoGetAttribute_int64(info, "rounding_mode", &rounding_mode);
    if (status != nullptr) rounding_mode = 0;
    status = api.KernelInfoGetAttribute_int64(info, "sub_block_size", &sub_block_size);
    if (status != nullptr) sub_block_size = 2;
    status = api.KernelInfoGetAttribute_int64(info, "sub_block_shift_bits", &sub_block_shift_bits);
    if (status != nullptr) sub_block_shift_bits = 1;
    status = api.KernelInfoGetAttribute_int64(info, "convert_to_bfloat_before_bfp", &convert_to_bfloat_before_bfp);
    if (status != nullptr) convert_to_bfloat_before_bfp = 0;
    status = api.KernelInfoGetAttribute_int64(info, "use_compiler_version_cpu_kernel", &use_compiler_version_cpu_kernel);
    if (status != nullptr) use_compiler_version_cpu_kernel = 0;

    return new BFPFixNeuronKernel(
      api, info, bfp_method, axis, bit_width, 
      block_size, rounding_mode, sub_block_size, 
      sub_block_shift_bits, convert_to_bfloat_before_bfp, 
      use_compiler_version_cpu_kernel);
  };

#if ORT_API_VERSION >= 17
  // This is for adapting to onnxruntime_cxx_api.h in ORT 1.17.0 (and higher)
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** op_kernel) const {
    return nullptr;
  };
  OrtStatusPtr KernelComputeV2(OrtKernelContext* context) const {
    return nullptr;
  };
#endif

  const char* GetName() const { return "BFPFixNeuron"; };

  const char* GetExecutionProviderType() const { 
  #ifdef USE_CUDA
    return "CUDAExecutionProvider"; 
  #else
    return "CPUExecutionProvider";
  #endif
  };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

#if ORT_API_VERSION >= 17
  // A function that will be called by SetShapeInferFn to get shape info
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count < 1) {
      return Ort::Status("input count should be greater than 0", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    //Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    //ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  };
#endif
};

#ifdef __cplusplus
}
#endif

