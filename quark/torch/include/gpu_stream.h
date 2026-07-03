//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

// Compile-time stream-accessor select aligned with the 2.10 stable-ABI floor
// (``torch_supports_stable_abi`` in Python).
//   * Stable-ABI build (``-DTORCH_TARGET_VERSION=...``) — always >= 2.10 by the
//     Python-side gate; use the AOTI shim. Note ``<torch/version.h>`` itself
//     ``#error``s when ``TORCH_TARGET_VERSION`` is defined, so we cannot read
//     it on this branch.
//   * Legacy JIT build, torch >= 2.10 — also uses the AOTI shim (same surface).
//   * Legacy JIT build, torch <  2.10 — ATen accessor; the stable-ABI headers
//     (``shim_utils.h`` / the AOTI shim) aren't shipped on the rocm/vllm 2.9
//     wheels (issue #5616).
#if !defined(TORCH_TARGET_VERSION)
#include <torch/version.h>
#if !defined(TORCH_VERSION_MAJOR) || !defined(TORCH_VERSION_MINOR)
#error "<torch/version.h> did not define TORCH_VERSION_MAJOR/MINOR"
#endif
#endif

#if defined(TORCH_TARGET_VERSION) || TORCH_VERSION_MAJOR > 2 || \
  (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

// Returns the current CUDA/HIP stream for the active device. Caller must have
// already set the correct device (e.g. via quark::(Optional)DeviceGuard).
inline cudaStream_t getCurrentStream() {
  int device_index;
  cudaGetDevice(&device_index);
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
  );
  return static_cast<cudaStream_t>(stream_ptr);
}

#else  // legacy JIT, torch < 2.10.

#include <ATen/cuda/CUDAContext.h>

inline cudaStream_t getCurrentStream() {
  return at::cuda::getCurrentCUDAStream();
}

#endif
