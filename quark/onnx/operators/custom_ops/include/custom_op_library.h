//
// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once
#include "onnxruntime_c_api.h"

// Force-export the ORT entry points. ORT's own ``ORT_EXPORT`` macro is
// empty on Linux (only set under ``__APPLE__``), and bare
// ``__attribute__((visibility("default")))`` doesn't compile on MSVC
// (Windows uses ``__declspec(dllexport)`` for the equivalent semantic).
// Without explicit export, the ``register_custom_ops_library`` ->
// ``dlsym("RegisterCustomOps")`` lookup returns null on a wheel built
// with ``-fvisibility=hidden`` (the manylinux gcc-toolset default), and
// every ``com.amd.quark`` kernel fails with ``NOT_IMPLEMENTED`` at
// session run time.
#if defined(_MSC_VER)
#define QUARK_ORT_EXPORT __declspec(dllexport)
#else
#define QUARK_ORT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QUARK_ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

// alternative name to test registration by function name
QUARK_ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif
