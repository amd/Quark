//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"
#undef ORT_API_MANUAL_INIT


void ParseElementDataTypeString(std::string& dtype,
                                std::vector<int>& bits,
                                std::vector<float>& range);

void to_mx(const Ort::Value& tensor,
		         std::string& scale_dtype,
			 std::string& element_dtype,
                         int64_t block_size,
                         int64_t rounding_mode,
                         Ort::Value& out);
