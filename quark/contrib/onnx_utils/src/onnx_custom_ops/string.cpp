// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications: Copyright (c) 2024 Advanced Micro Devices, Inc.

#include "ryzenai/onnx_utils/string.hpp"

#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
// clang-format off
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>  // NOLINT(misc-include-cleaner)
#include <stringapiset.h>
#include <WinNls.h>
// clang-format on
#endif

namespace ryzenai::onnx_utils {

#ifdef _WIN32
std::string toUtf8String(std::wstring_view str) {
  if (str.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("length overflow");
  }

  const auto src_len = static_cast<int>(str.size() + 1);
  // this check requires the size to be passed when using .data() method of a
  // string_view object. This is being done but clang-tidy cannot detect this
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  const int len = WideCharToMultiByte(
    CP_UTF8, 0, str.data(), src_len, nullptr, 0, nullptr, nullptr
  );
  assert(len > 0);
  std::string new_str(static_cast<size_t>(len) - 1, '\0');
  [[maybe_unused]] const int ret = WideCharToMultiByte(
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    CP_UTF8, 0, str.data(), src_len, new_str.data(), len, nullptr, nullptr
  );
  assert(len == ret);
  return new_str;
}

std::wstring toWideString(std::string_view str) {
  if (str.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("length overflow");
  }

  const auto src_len = static_cast<int>(str.size() + 1);
  const int len =
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    MultiByteToWideChar(CP_UTF8, 0, str.data(), src_len, nullptr, 0);
  assert(len > 0);
  std::wstring new_str(static_cast<size_t>(len) - 1, '\0');
  [[maybe_unused]] const int ret =
    // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
    MultiByteToWideChar(CP_UTF8, 0, str.data(), src_len, new_str.data(), len);
  assert(len == ret);
  return new_str;
}
#endif  // #ifdef _WIN32

bool endsWith(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

}  // namespace ryzenai::onnx_utils
