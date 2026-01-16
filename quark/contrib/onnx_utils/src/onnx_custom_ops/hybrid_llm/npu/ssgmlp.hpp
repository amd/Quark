// Copyright (c) 2023 Advanced Micro Devices, Inc.

#pragma once
#include <onnxruntime_cxx_api.h>

#include <string>
#include <unordered_map>

#include "jit_node.hpp"
#include "ssmlp_base.hpp"

namespace ryzenai::onnx_utils {

class AMDSSGMLPKernel : public SSMLPBase<AMDSSGMLPKernel> {
 public:
  AMDSSGMLPKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
};

extern template class JitNode<JitNode<AMDSSGMLPKernel>>;
extern template class SSMLPBase<JitNode<AMDSSGMLPKernel>>;
extern template class SSMLPBase<AMDSSGMLPKernel>;
extern template class JitNode<AMDSSGMLPKernel>;
}  // namespace ryzenai::onnx_utils
