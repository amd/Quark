// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

namespace ryzenai::onnx_utils {

class LoraOpInterface {
 public:
  virtual void LoadLora() = 0;
};

}  // namespace ryzenai::onnx_utils
