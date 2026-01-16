// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications: Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once
#include <memory>

#include "onnxruntime_c_api.h"

namespace ryzenai::onnx_utils {

class CustomAllocator : public OrtAllocator {
 public:
  CustomAllocator(bool enable);
  ~CustomAllocator() noexcept;

  inline const auto Info() const { return OrtAllocator::Info(this); }
  inline const auto& GetInfo() const { return *OrtAllocator::Info(this); }

  inline void ReleaseMemory() {}

 private:
  struct Data;
  std::unique_ptr<Data> data_;
};

}  // namespace ryzenai::onnx_utils
