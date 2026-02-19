// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_CUSTOM_OPS
#define GUARD_ONNX_CUSTOM_OPS_CUSTOM_OPS

#include <onnxruntime_c_api.h>

#include <atomic>

#ifndef ONNX_UTILS_EXPORT
#ifdef _WIN32
#define ONNX_UTILS_EXPORT __declspec(dllexport)
#else
#define ONNX_UTILS_EXPORT
#endif
#endif

namespace ryzenai {
using RegisterCustomOpsFunction = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*,
                                                            const OrtApiBase*);
extern std::atomic<RegisterCustomOpsFunction> RegisterCustomOpsOverride;
}  // namespace ryzenai

#ifdef __cplusplus
extern "C" {
#endif

ONNX_UTILS_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base);

ONNX_UTILS_EXPORT
void LoraCompute(const char* lora_name);

ONNX_UTILS_EXPORT
void LoraComputeFromBuffer(
  const char* lora_name, const void* prefill_proto_ptr,
  size_t prefill_proto_size, const void* prefill_bin_ptr,
  size_t prefill_bin_size, const void* token_proto_ptr, size_t token_proto_size,
  const void* token_bin_ptr, size_t token_bin_size
);

ONNX_UTILS_EXPORT OrtStatus* ORT_API_CALL RyzenAI_RegisterCustomOps(
  OrtSessionOptions* options, const OrtApiBase* api_base
);

#ifdef __cplusplus
}
#endif

#endif  // GUARD_ONNX_CUSTOM_OPS_CUSTOM_OPS
