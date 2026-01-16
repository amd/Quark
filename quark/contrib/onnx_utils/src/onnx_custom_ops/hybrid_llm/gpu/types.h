// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Don't define min/max macros in the Windows headers.
#define NOMINMAX

#include <winrt/base.h>

#include <cstdint>
#include <exception>
#include <memory>

// Require DML version 6.3 or higher
// #define DML_TARGET_VERSION 0x6300

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

using winrt::hstring;

#if NUGET_BUILD == 0
// This is a workaround for a bug in the MS SDK in Visual Studio 2019.
// https://github.com/microsoft/Windows.UI.Composition-Win32-Samples/issues/47#issuecomment-672696574
namespace winrt::impl {
template <typename Async>
auto wait_for(Async const& async, Windows::Foundation::TimeSpan const& timeout);
}
#endif

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Determines the maximum of two numbers.
template <typename T>
constexpr T Max(T value1, T value2) {
  return ((value1 > value2) ? value1 : value2);
}

// =====================================================================================================================
// Determines the minimum of two numbers.
template <typename T>
constexpr T Min(T value1, T value2) {
  return ((value1 < value2) ? value1 : value2);
}

// =====================================================================================================================
// Computes the absolute value of a number.
template <typename T>
constexpr T Abs(T value) {
  return (value < 0) ? -value : value;
}

// =====================================================================================================================
// Increments a const pointer by nBytes by first casting it to a const uint8*.
template <typename T>
constexpr const void* VoidPtrInc(const void* p, T numBytes) {
  return (static_cast<const uint8*>(p) + numBytes);
}

// =====================================================================================================================
// Increments a pointer by nBytes by first casting it to a uint8*.
template <typename T>
constexpr void* VoidPtrInc(void* p, T numBytes) {
  return (static_cast<uint8*>(p) + numBytes);
}

// =====================================================================================================================
// Rounds the specified uint 'value' up to the nearest value meeting the
// specified 'alignment'. Only power of 2 alignments are supported by this
// function.
template <typename T>
T Pow2Align(T value, uint64 alignment) {
  return (
    (value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1)
  );
}

// =====================================================================================================================
// Implements an alternative version of integer division in which the quotient
// is rounded up instead of down.
template <typename T>
constexpr T RoundUpQuotient(T dividend, T divisor) {
  return ((dividend + (divisor - 1)) / divisor);
}

// =====================================================================================================================
// An IEEE 754 float-32 split into its three components.
struct Float32 {
  static constexpr uint32 FracBits =
    23;  // The number of bits in the fraction (significand).
  static constexpr uint32 ExpBits =
    8;  // The number of bits in the exponent term.

  static constexpr uint32 ExpInf =
    (1 << ExpBits) - 1;  // Indicates infinity or NaN.
  static constexpr int32 ExpBias =
    (1 << (ExpBits - 1)) - 1;  // 2^0 is at this exponent value.

  Float32() {}
  Float32(float val) : f32(val) {}

  operator float() const { return f32; }

  union {
    struct {
      uint32 frac : FracBits;  // The fraction (significand) bits.
      uint32 exp : ExpBits;    // The exponent term.
      uint32 sign : 1;         // Set to 1 if the number is negative.
    } bits;  // This struct split into its binary representation.

    float f32;  // This struct as a native float.
  };
};

static_assert(sizeof(Float32) == sizeof(uint32), "Float32 must be 32 bits!");

// =====================================================================================================================
// An IEEE 754 float-16 split into its three components.
struct Float16 {
  static constexpr uint32 FracBits =
    10;  // The number of bits in the fraction (significand).
  static constexpr uint32 ExpBits =
    5;  // The number of bits in the exponent term.

  static constexpr uint32 ExpInf =
    (1 << ExpBits) - 1;  // Indicates infinity or NaN.
  static constexpr int32 ExpBias =
    (1 << (ExpBits - 1)) - 1;  // 2^0 is at this exponent value.

  Float16() {}
  Float16(float val);

  operator float() const;

  union {
    struct {
      uint16 frac : FracBits;  // The fraction (significand) bits.
      uint16 exp : ExpBits;    // The exponent term.
      uint16 sign : 1;         // Set to 1 if the number is negative.
    } bits;  // This struct split into its binary representation.

    uint16 u16;  // This struct as a single 16-bit integer.
  };
};

static_assert(sizeof(Float16) == sizeof(uint16), "Float16 must be 16 bits!");

}  // namespace ryzenai::onnx_utils
