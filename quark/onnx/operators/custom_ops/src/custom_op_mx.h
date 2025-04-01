//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#endif

#include "custom_op_mx.h"

#ifdef __cplusplus
extern "C" {
#endif


struct MXFixNeuronKernel {
  MXFixNeuronKernel(
    const OrtApi& ort_api, 
    const OrtKernelInfo* info, 
    std::string& scale_dtype,
    std::string& element_dtype,
    int64_t axis,
    int64_t block_size,
    int64_t rounding_mode);
  ~MXFixNeuronKernel();
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

  Ort::Value do_mx(Ort::Value &input);

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

  std::string scale_dtype_ = "e8m0";
  std::string element_dtype_ = "int8";
  int64_t axis_ = 1;
  int64_t block_size_ = 32;
  int64_t rounding_mode_ = 0;
};

struct MXFixNeuron : Ort::CustomOpBase<MXFixNeuron, MXFixNeuronKernel> {
  explicit MXFixNeuron() {}

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
    Ort::InitApi(&api);

    std::string scale_dtype = "e8m0";
    std::string element_dtype;
    int64_t axis;
    int64_t block_size;
    int64_t rounding_mode;

    // It has two steps to get attribute with string datatype
    const char* attribute_name = "element_dtype";
    // Step1. Get the string length
    size_t str_len;
    auto status = api.KernelInfoGetAttribute_string(info, attribute_name, nullptr, &str_len);
    // Step2. Get the string content
    char* str_ptr = (char*)malloc(str_len);
    status = api.KernelInfoGetAttribute_string(info, attribute_name, str_ptr, &str_len);
    element_dtype = (status != nullptr) ? "int8" : str_ptr;
    free(str_ptr); // Free the memory

    status = api.KernelInfoGetAttribute_int64(info, "axis", &axis);
    if (status != nullptr) axis = 1;
    status = api.KernelInfoGetAttribute_int64(info, "block_size", &block_size);
    if (status != nullptr) block_size = 32;
    status = api.KernelInfoGetAttribute_int64(info, "rounding_mode", &rounding_mode);
    if (status != nullptr) rounding_mode = 0;

    return new MXFixNeuronKernel(
      api, info, scale_dtype, element_dtype, axis, block_size, rounding_mode);
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

  const char* GetName() const { return "MXFixNeuron"; };

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
