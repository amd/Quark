//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable-ABI replacements for ATen device guards:
//   quark::DeviceGuard         — replaces at::DeviceGuard
//   quark::OptionalDeviceGuard — replaces at::(cuda::)?OptionalCUDAGuard
//

#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

namespace quark {

// Sets the current device to the tensor's device on construction; restores
// on destruction. Tensor MUST be on an accelerator device (CUDA/HIP).
class DeviceGuard {
 public:
  explicit DeviceGuard(const torch::stable::Tensor& tensor)
    : guard_(static_cast<int32_t>(tensor.get_device())) {}

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&&) = delete;
  DeviceGuard& operator=(DeviceGuard&&) = delete;

 private:
  torch::stable::accelerator::DeviceGuard guard_;
};

// No-op when the tensor is on CPU; sets the device otherwise.
class OptionalDeviceGuard {
 public:
  explicit OptionalDeviceGuard(const torch::stable::Tensor& tensor) {
    if (tensor.is_cuda()) {
      guard_.emplace(static_cast<int32_t>(tensor.get_device()));
    }
  }

  OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard(OptionalDeviceGuard&&) = delete;
  OptionalDeviceGuard& operator=(OptionalDeviceGuard&&) = delete;

 private:
  std::optional<torch::stable::accelerator::DeviceGuard> guard_;
};

}  // namespace quark
