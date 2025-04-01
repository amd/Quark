//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"
#undef ORT_API_MANUAL_INIT

void to_bfp(const Ort::Value& tensor,
                         int64_t bit_width,
                         int64_t block_size,
                         int64_t rounding_mode,
                         Ort::Value& out,
                         int64_t use_compiler_version_cpu_kernel);

void to_bfloat(Ort::Value& tensor);

void to_bfp_prime(const Ort::Value& tensor,
                                   int64_t bit_width,
                                   int64_t block_size,
                                   int64_t sub_block_size,
                                   int64_t sub_block_shift_bits,
                                   int64_t rounding_mode,
                                   Ort::Value& out);
