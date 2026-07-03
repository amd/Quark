//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// BFloat16 conversion utility shared between BFP kernels and instance norm.
//

#pragma once

#include <stdint.h>

namespace quark_onnx {

typedef union value_convert {
  uint32_t u;
  int32_t i;
  float f;
} value_convert_t;

static inline uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

static inline float u_to_f(uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

static inline int32_t f_to_i(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.i;
}

static inline float i_to_f(int32_t data) {
  value_convert_t vc{};
  vc.i = data;
  return vc.f;
}

// fp32->bf16 round-to-nearest-even constants.
constexpr uint32_t kBf16RoundingBit = 0x00008000u;
constexpr uint32_t kBf16LowMask = 0x0000FFFFu;
constexpr uint32_t kBf16Lsb = 0x00010000u;
constexpr uint32_t kBf16HighMask = 0xFFFF0000u;

static inline float float2bfloat_cpu(const float x) {
  uint32_t itmp = f_to_u(x);
  if ((itmp & kBf16RoundingBit) == kBf16RoundingBit) {
    if ((itmp & kBf16LowMask) > kBf16RoundingBit ||
        (((itmp & kBf16LowMask) == kBf16RoundingBit) &&
         ((itmp & kBf16Lsb) == kBf16Lsb))) {
      itmp += kBf16Lsb;
    }
  }
  itmp &= kBf16HighMask;
  return u_to_f(itmp);
}

}  // namespace quark_onnx
