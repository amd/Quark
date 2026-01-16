// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <stdint.h>

#include <fstream>
#include <string>
#include <unordered_map>

namespace ryzenai::onnx_utils {

template <typename T, typename U = T>
void writeToFile(
  const std::string& filename, const T* data, size_t size, bool as_binary
) {
  std::ofstream ofs{filename, as_binary ? std::ios::binary : std::ios::out};

  if (!ofs) {
    throw std::ios_base::failure("Failed to open file");
  }

  if (as_binary) {
    ofs.write((char*)(data), size * sizeof(T));
  } else {
    for (size_t i = 0; i < size; ++i) {
      ofs << static_cast<U>(data[i]) << "\n";
    }
  }

  if (!ofs) {
    throw std::ios_base::failure("Failed to write data to file");
  }
  ofs.close();
}

float bfloat_to_float_2(uint16_t x);
uint16_t float_to_bfloat16_2(float x);
float bfloat16_to_float_single(uint16_t v);
uint16_t float_to_bfloat16(float x);
uint16_t float_to_bfloat16_2(float f);

}  // namespace ryzenai::onnx_utils
