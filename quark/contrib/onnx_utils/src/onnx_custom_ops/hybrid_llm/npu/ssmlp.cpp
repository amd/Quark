// Copyright (C) 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "ssmlp.hpp"

#include "jit_node_impl.hpp"

namespace ryzenai::onnx_utils {

using NPUTensor = ::Tensor;

AMDSSMLPKernel::AMDSSMLPKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : SSMLPBase(this, k_info, session_configs, false) {}

template class JitNode<AMDSSMLPKernel>;
template class JitNode<JitNode<AMDSSMLPKernel>>;

}  // namespace ryzenai::onnx_utils
