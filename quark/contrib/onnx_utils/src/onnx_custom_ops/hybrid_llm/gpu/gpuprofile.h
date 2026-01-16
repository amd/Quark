// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// #define ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING

#include <chrono>
#include <iostream>
#include <sstream>
#include <tuple>

namespace ryzenai::onnx_utils {

using Clock = std::chrono::steady_clock;

using Duration = Clock::duration;

using MillisecondsFp =
  std::chrono::duration<float, std::chrono::milliseconds::period>;

enum GPUEventID : int {
  MATMULNBITS_UPLOAD_ID = 0,
  MATMULNBITS_EXECUTE_ID = 1,
  MATMULNBITS_DOWNLOAD_ID = 2,
  SS_MLP_UPLOAD_ID = 3,
  SS_MLP_EXECUTE_ID = 4,
  SS_MLP_DOWNLOAD_ID = 5,
  GQO_DYNAMIC_INIT_ID = 6,
  GQO_UPLOAD_ID = 7,
  GQO_EXECUTE_ID = 8,
  GQO_DOWNLOAD_ID = 9,
};
}  // namespace ryzenai::onnx_utils

#endif
