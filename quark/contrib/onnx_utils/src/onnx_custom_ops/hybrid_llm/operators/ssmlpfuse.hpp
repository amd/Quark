// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once
#include "hybrid_kernel.hpp"
#include "hybrid_operator.hpp"
#include "opUtils.h"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/ssmlpfuse.hpp"
#endif

namespace ryzenai::onnx_utils {
class SSMlpFuseKernel : public HybridKernel {
 public:
  SSMlpFuseKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  ~SSMlpFuseKernel();

  void Compute(OrtKernelContext* context);

 private:
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  std::unique_ptr<AMDSSMlpFuseKernel> npu_ssmlpfuse_;
#endif
};

static const char kSSMlpFuse[] = "SSMLP_FUSE";

struct SSMlpFuse : HybridOperator<SSMlpFuseKernel, kSSMlpFuse> {
  explicit SSMlpFuse(const Ort::ConstSessionOptions& session_options)
    : HybridOperator<SSMlpFuseKernel, kSSMlpFuse>(
        session_options, GetSessionConfigKeys()
      ) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const { return {}; }
};

}  // namespace ryzenai::onnx_utils
