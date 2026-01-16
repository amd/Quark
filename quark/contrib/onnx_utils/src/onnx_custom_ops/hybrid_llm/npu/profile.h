// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// #define ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING

#include <psapi.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <tuple>

namespace ryzenai::onnx_utils {

using Clock = std::chrono::steady_clock;

using Duration = Clock::duration;

using MillisecondsFp =
  std::chrono::duration<float, std::chrono::milliseconds::period>;

enum EventID : int {
  CONFIG_ID = 0,
  SETUP_ID = 1,
  INPUT_FORMAT_ID = 2,
  OUTPUT_FORMAT_ID = 3,
  MATMUL_EXECUTE_ID = 4,
  LRN0_EXECUTE_ID = 4,
  MLP_EXECUTE_ID = 5,
  LRN1_EXECUTE_ID = 6,
  GQO_EXECUTE_ID = 4,
  OPROJ_EXECUTE_ID = 5,
  MAX_EVENTS = 8,
};

static inline size_t GetWorkingSetSizeInBytes() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    throw std::runtime_error(
      "GetProcessMemoryInfo failed with error code " +
      std::to_string(GetLastError())
    );
  }

  return pmc.WorkingSetSize;
}

using ProfileInfo = std::tuple<EventID, Duration>;
}  // namespace ryzenai::onnx_utils

#endif
