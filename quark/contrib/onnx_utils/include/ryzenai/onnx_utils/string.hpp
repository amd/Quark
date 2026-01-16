// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <string>

namespace ryzenai::onnx_utils {

#ifdef _WIN32
std::string toUtf8String(std::wstring_view str);
std::wstring toWideString(std::string_view str);
#endif

bool endsWith(const std::string& str, const std::string& suffix);

}  // namespace ryzenai::onnx_utils
