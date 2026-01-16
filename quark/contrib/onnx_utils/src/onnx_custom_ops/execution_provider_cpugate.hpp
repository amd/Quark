// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <map>
#include <memory>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::CPUGate {
struct OpParams {
  std::string op_name;
  std::string domain;
  int version = -1;
  std::map<std::string, ONNXTensorElementDataType> type_constraint;
  std::vector<Ort::OpAttr> attrs;
  size_t input_count = 0;
  size_t output_count = 0;
};

struct Op {
  virtual ~Op() = default;

  virtual void Invoke(
    std::vector<const OrtValue*> input, std::vector<OrtValue*> output
  ) = 0;
};

struct Interface {
  virtual ~Interface() = default;

  virtual std::shared_ptr<Op> CreateOp(OpParams params) = 0;

  virtual void Initialize() = 0;
};

std::shared_ptr<Interface> CreateInstance();
}  // namespace ryzenai::CPUGate
