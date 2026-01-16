// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "ssgmlp.hpp"

#include "jit_node_impl.hpp"

namespace ryzenai::onnx_utils {

AMDSSGMLPKernel::AMDSSGMLPKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : SSMLPBase(this, k_info, session_configs, true) {}

template class JitNode<AMDSSGMLPKernel>;
template class JitNode<JitNode<AMDSSGMLPKernel>>;

}  // namespace ryzenai::onnx_utils
