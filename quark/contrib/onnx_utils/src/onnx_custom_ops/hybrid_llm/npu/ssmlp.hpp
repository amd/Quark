// Copyright (c) 2023 Advanced Micro Devices, Inc.

#pragma once
#include <onnxruntime_cxx_api.h>

#include <string>
#include <unordered_map>

#include "jit_node.hpp"
#include "ssmlp_base.hpp"

namespace ryzenai::onnx_utils {

class AMDSSMLPKernel : public SSMLPBase<AMDSSMLPKernel> {
 public:
  AMDSSMLPKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
};

extern template class JitNode<JitNode<AMDSSMLPKernel>>;
extern template class SSMLPBase<AMDSSMLPKernel>;
extern template class SSMLPBase<JitNode<AMDSSMLPKernel>>;
extern template class JitNode<AMDSSMLPKernel>;
}  // namespace ryzenai::onnx_utils
