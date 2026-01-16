// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

struct SharedWeightsInfo {
  std::string key = "";
  size_t addr = 0;
};

class SharedWeights {
 public:
  bool isEnabled() const;
  bool ready() const;

  void setModelKey(std::string key);

  void setWeightsCount(size_t count);
  void setWeightKey(int index, std::string key);
  void setWeightKey(
    int index, const Ort::ConstKernelInfo& info, const char* attr_name
  );

  void setWeightAddr(int index);
  size_t weightAddr(int index) const;

 private:
  std::string model_key_;
  std::vector<SharedWeightsInfo> weights_;
};

}  // namespace ryzenai::onnx_utils
