//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

#include "custom_op_qdq.h"
#include "custom_op_in.h"
#include "custom_op_lstm.h"
#include "custom_op_bfp.h"
#include "custom_op_mx.h"

#define ORT_TRY try
#define ORT_CATCH(x) catch (x)
#define ORT_RETHROW throw;

#define ORT_HANDLE_EXCEPTION(func) func()


// For downward compatibility
static const char* c_OpDomain_deprecated = "com.vai.quantize";
static const char c_OpName1_deprecated[] = "VitisQuantizeLinear";
static const char c_OpName2_deprecated[] = "VitisDequantizeLinear";
static const char c_OpName3_deprecated[] = "VitisInstanceNormalization";
static const char c_OpName4_deprecated[] = "VitisLSTM";
static const char c_OpName5_deprecated[] = "BFPFixNeuron";
static const char c_OpName6_deprecated[] = "MXFixNeuron";

static const char* c_OpDomain = "com.amd.quark";
static const char c_OpName1[] = "ExtendedQuantizeLinear";
static const char c_OpName2[] = "ExtendedDequantizeLinear";
static const char c_OpName3[] = "ExtendedInstanceNormalization";
static const char c_OpName4[] = "ExtendedLSTM";
static const char c_OpName5[] = "BFPQuantizeDequantize";
static const char c_OpName6[] = "MXQuantizeDequantize";

static const int c_OpVersion = 1;


template <const char* OpName, int OpVersion>
struct CustomQuantizeLinear : Ort::CustomOpBase<CustomQuantizeLinear<OpName, OpVersion>, quark_onnx::KernelCustomQuantizeLinear> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<quark_onnx::KernelCustomQuantizeLinear>(api, info).release();
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

  const char* GetName() const { return OpName; };
  int GetVersion() const { return OpVersion; };

  const char* GetExecutionProviderType() const {
  #ifdef NO_GPU
    return "CPUExecutionProvider";
  #elif defined(USE_ROCM)
    return "ROCMExecutionProvider";
  #elif defined(USE_CUDA)
    return "CUDAExecutionProvider";
  #else
    return "CPUExecutionProvider";
  #endif
  };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
#if 0
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The third input (index == 2) is optional
    if (index == 2)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  };
#endif
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };
#if 0
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
#endif

#if ORT_API_VERSION >= 17
  // A function that will be called by SetShapeInferFn to get shape info
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count <= 1) {
      return Ort::Status("input count should be greater than 1", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    //Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    //ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  };
#endif
};

template <const char* OpName, int OpVersion>
struct CustomDequantizeLinear : Ort::CustomOpBase<CustomDequantizeLinear<OpName, OpVersion>, quark_onnx::KernelCustomDequantizeLinear> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<quark_onnx::KernelCustomDequantizeLinear>(api, info).release();
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

  const char* GetName() const { return OpName; };
  int GetVersion() const { return OpVersion; };

  const char* GetExecutionProviderType() const {
  #ifdef NO_GPU
    return "CPUExecutionProvider";
  #elif defined(USE_ROCM)
    return "ROCMExecutionProvider";
  #elif defined(USE_CUDA)
    return "CUDAExecutionProvider";
  #else
    return "CPUExecutionProvider";
  #endif
  };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
#if 0
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The third input (index == 2) is optional
    if (index == 2)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  };
#endif
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
#if 0
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
#endif

#if ORT_API_VERSION >= 17
  // A function that will be called by SetShapeInferFn to get shape info
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count <= 1) {
      return Ort::Status("input count should be greater than 1", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    //Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    //ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  };
#endif
};

template <const char* OpName, int OpVersion>
struct CustomInstanceNormalization : Ort::CustomOpBase<CustomInstanceNormalization<OpName, OpVersion>, quark_onnx::KernelCustomInstanceNormalization> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<quark_onnx::KernelCustomInstanceNormalization>(api, info).release();
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

  const char* GetName() const { return OpName; };
  int GetVersion() const { return OpVersion; };

  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

#if ORT_API_VERSION >= 17
  // A function that will be called by SetShapeInferFn to get shape info
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count != 3) {
      return Ort::Status("input count should be 3", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    //Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    //ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  };
#endif
};

template <const char* OpName, int OpVersion>
struct CustomLSTM : Ort::CustomOpBase<CustomLSTM<OpName, OpVersion>, quark_onnx::KernelCustomLSTM> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<quark_onnx::KernelCustomLSTM>(api, info).release();
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

  const char* GetName() const { return OpName; };
  int GetVersion() const { return OpVersion; };

  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const { return 4; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index <= 3)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

#if ORT_API_VERSION >= 17
  // A function that will be called by SetShapeInferFn to get shape info
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count <= 3) {
      return Ort::Status("input count should be greater than 3", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    //Ort::ShapeInferContext::Shape shape = ctx.GetInputShape(0);
    //ctx.SetOutputShape(0, shape);
    return Ort::Status(nullptr);
  };
#endif
};

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CustomQuantizeLinear<c_OpName1_deprecated, c_OpVersion> c_CustomQ;
  static const CustomDequantizeLinear<c_OpName2_deprecated, c_OpVersion> c_CustomDQ;
  static const CustomInstanceNormalization<c_OpName3_deprecated, c_OpVersion> c_CustomIN;
  static const CustomLSTM<c_OpName4_deprecated, c_OpVersion> c_CustomLSTM;
  static const BFPFixNeuron<c_OpName5_deprecated, c_OpVersion> c_BFPFixNeuron;
  static const MXFixNeuron<c_OpName6_deprecated, c_OpVersion> c_MXFixNeuron;

  static const CustomQuantizeLinear<c_OpName1, c_OpVersion> c_ExtendedQ;
  static const CustomDequantizeLinear<c_OpName2, c_OpVersion> c_ExtendedDQ;
  static const CustomInstanceNormalization<c_OpName3, c_OpVersion> c_ExtendedIN;
  static const CustomLSTM<c_OpName4, c_OpVersion> c_ExtendedLSTM;
  static const BFPFixNeuron<c_OpName5, c_OpVersion> c_BFP;
  static const MXFixNeuron<c_OpName6, c_OpVersion> c_MX;

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain_deprecated{c_OpDomain_deprecated};
    domain_deprecated.Add(&c_CustomQ);
    domain_deprecated.Add(&c_CustomDQ);
    domain_deprecated.Add(&c_CustomIN);
    domain_deprecated.Add(&c_CustomLSTM);
    domain_deprecated.Add(&c_BFPFixNeuron);
    domain_deprecated.Add(&c_MXFixNeuron);

    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_ExtendedQ);
    domain.Add(&c_ExtendedDQ);
    domain.Add(&c_ExtendedIN);
    domain.Add(&c_ExtendedLSTM);
    domain.Add(&c_BFP);
    domain.Add(&c_MX);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain_deprecated);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain_deprecated));
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
