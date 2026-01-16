// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#ifdef _WIN32
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <cassert>

#include "onnxruntime_cxx_api.h"
#include "ops/ops_common/dtype_utils.h"

namespace ryzenai::onnx_utils {

float bfloat_to_float_2(uint16_t x) {
  float i = 0;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy uint16_t to float (msb)
  std::memcpy(tmp + 2, src, sizeof(uint16_t));
  return i;
}

float bfloat16_to_float_single(uint16_t v) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = (uint32_t(v)) << 16;
  return u.f;
}

uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t* src = (uint8_t*)&x;
  uint8_t* tmp = (uint8_t*)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

uint16_t float_to_bfloat16_2(float f) {
  uint32_t val = *reinterpret_cast<uint32_t*>(&f);

  return val >> 16;
}

}  // namespace ryzenai::onnx_utils
