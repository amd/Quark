// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "custom_ops.hpp"

#include <mutex>
#include <utility>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#ifdef ONNX_UTILS_ENABLE_PROJECT_DD
#include "dynamic_dispatch/main.hpp"
#endif
#ifdef ONNX_UTILS_ENABLE_CORELIB
#include "corelib/corelib.hpp"
#endif
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM
#include "hybrid_llm/main.hpp"
#endif

std::atomic<ryzenai::RegisterCustomOpsFunction>
  ryzenai::RegisterCustomOpsOverride = nullptr;

// This function shows one way of keeping domains alive until the library is
// unloaded.
static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

namespace ryzenai::onnx_utils {
// by default it's set for CPU-mode, in EP-mode it's gonna be dynamically switch
// to EP-specific name
const char* ExecutionProvider = kOnnxUtilsEp;
}  // namespace ryzenai::onnx_utils

// Called by ONNX Runtime to register the library's custom operators with the
// provided session options.
OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  if (auto func = ryzenai::RegisterCustomOpsOverride.load())
    return func(options, api);

  // Manually initialize the OrtApi to enable C++ API classes and functions.
  Ort::InitApi(api->GetApi(ORT_API_VERSION));

  Ort::ConstSessionOptions options_obj{options};

  Ort::CustomOpDomain domain{"com.ryzenai"};

#ifdef ONNX_UTILS_ENABLE_PROJECT_DD
  auto dynamic_dispatch_ops =
    ryzenai::onnx_utils::create_dynamic_dispatch_ops(options_obj);
  for (const auto* op : dynamic_dispatch_ops) {
    domain.Add(op);
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_DD

#ifdef ONNX_UTILS_ENABLE_CORELIB
  auto corelib_ops = ryzenai::onnx_utils::create_corelib_ops(options_obj);
  for (const auto* op : corelib_ops) {
    domain.Add(op);
  }
#endif  // ONNX_UTILS_ENABLE_CORELIB

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM
  auto hybrid_llm_ops = ryzenai::onnx_utils::create_hybrid_llm_ops(options_obj);
  for (const auto* op : hybrid_llm_ops) {
    domain.Add(op);
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM

  OrtStatus* result = nullptr;

  try {
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }
  return result;
}

#ifdef ONNX_UTILS_USE_RYZENAI_EP
OrtStatus* ORT_API_CALL RyzenAI_RegisterCustomOps(
  OrtSessionOptions* options, const OrtApiBase* api_base
) {
  return RegisterCustomOps(options, api_base);
}
#endif
