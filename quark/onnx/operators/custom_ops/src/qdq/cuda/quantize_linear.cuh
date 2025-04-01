//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cuda_runtime.h>


template <class OutT, class InT>
bool CudaQuantizeLinearStdInt(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element);
template <class OutT, class InT>
bool CudaQuantizeLinearStdFp(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element);
template <class OutT, class InT>
bool CudaQuantizeLinearStdBf(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element);


template <class InT, class OutT>
bool CudaDequantizeLinearStdInt(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element);
template <class InT, class OutT>
bool CudaDequantizeLinearStdFp(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element);
template <class InT, class OutT>
bool CudaDequantizeLinearStdBf(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element);
