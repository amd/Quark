// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "ryzenai/onnx_utils/string.hpp"

void test_single_matmul(
  const std::string& golden_onnx_model, const std::string& test_onnx_model,
  const std::string& dll_path
) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_single_matmul");
  Ort::SessionOptions session_options;
  Ort::CustomOpConfigs custom_op_configs;

  auto dll_path_wide = ryzenai::onnx_utils::toWideString(dll_path);
  auto golden_model_wide = ryzenai::onnx_utils::toWideString(golden_onnx_model);

  session_options.RegisterCustomOpsLibrary(
    dll_path_wide.c_str(), custom_op_configs
  );
  auto session = Ort::Session(env, golden_model_wide.c_str(), session_options);

  // TODO(varunsh): run the models
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: test_single_matmul golden_onnx_model test_onnx_model "
                 "dll_path\n";
    return -1;
  }

  auto golden_onnx_model = std::string{argv[1]};
  auto test_onnx_model = std::string{argv[2]};
  auto dll_path = std::string{argv[3]};

  test_single_matmul(golden_onnx_model, test_onnx_model, dll_path);

  std::cout << "Success\n";
  return 0;
}
