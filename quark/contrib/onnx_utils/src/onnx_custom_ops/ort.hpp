// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <optional>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

template <typename R>
R getAttribute(
  const Ort::ConstKernelInfo& kernel_info, const char* name,
  std::optional<R> default_value = std::nullopt
) {
  try {
    return kernel_info.GetAttribute<R>(name);
  } catch (const Ort::Exception&) {
    if (default_value) return default_value.value();
    throw;
  }
}

template <typename R>
std::vector<R> getAttributes(
  const Ort::ConstKernelInfo& kernel_info, const char* name,
  std::optional<std::vector<R>> default_value = std::nullopt
) {
  try {
    return kernel_info.GetAttributes<R>(name);
  } catch (const Ort::Exception&) {
    if (default_value) return default_value.value();
    throw;
  }
}

}  // namespace ryzenai::onnx_utils
