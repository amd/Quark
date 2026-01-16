// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "types.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Constructs a Float16 from a native float.
Float16::Float16(float val) {
  const Float32 fp32(val);

  // Always copy the sign bit.
  bits.sign = fp32.bits.sign;

  if (fp32.bits.exp == 0x0) {
    // Convert zero and fp32 subnornals to zero. The fp32 subnormals are way,
    // way out of the fp16 range.
    bits.exp = 0x0;
    bits.frac = 0x0;
  } else {
    // Adjust the exponent bias and check the four boundary special cases.
    const int32 adjExp = fp32.bits.exp - Float32::ExpBias + ExpBias;

    bits.exp = (fp32.bits.exp == 0)                 ? 0
               : (fp32.bits.exp == Float32::ExpInf) ? ExpInf
               : (adjExp <= 0x1)                    ? 0x1
               : (adjExp >= Float32::ExpInf - 1)    ? ExpInf - 1
                                                    : adjExp;

    // Strip off the lower fraction bits that we can't fit.
    bits.frac = fp32.bits.frac >> (Float32::FracBits - FracBits);
  }
}

// =====================================================================================================================
// Converts float-16 data to float-32.
Float16::operator float() const {
  // Note that we only need to copy the sign to covert zero values.
  Float32 fp32 = {};
  fp32.bits.sign = bits.sign;

  if ((bits.exp == 0x0) && (bits.frac != 0x0)) {
    // Convert an fp16 subnormal to an fp32 normal with a negative exponent.
    // Find the highest set fraction bit, it must become the new "1.xxx" in our
    // normalized value.
    int32 hiPos = 0;
    for (int32 pos = 0; pos < FracBits; ++pos) {
      if (((1 << pos) & bits.frac) != 0) {
        hiPos = pos;
      }
    }

    // We must subtract this many places from the exponent and shift up this
    // many places.
    const int32 distance = FracBits - hiPos;

    // Compute normalized values in fp16 notation, the biased exp might go
    // negative.
    const int32 normExp = 1 - distance;
    const uint32 normFrac = (bits.frac << distance) & ((1 << FracBits) - 1);

    // Convert to the proper fp32 notation.
    fp32.bits.exp = normExp + Float32::ExpBias - ExpBias;
    fp32.bits.frac = normFrac << (Float32::FracBits - FracBits);
  } else if (bits.exp != 0x0) {
    // A non-zero exp means we have an Inf/NaN or fp16 normal which is easy to
    // convert.
    fp32.bits.exp = (bits.exp == ExpInf)
                      ? Float32::ExpInf
                      : (bits.exp + Float32::ExpBias - ExpBias);
    fp32.bits.frac = bits.frac << (Float32::FracBits - FracBits);
  }

  return fp32.f32;
}

}  // namespace ryzenai::onnx_utils
