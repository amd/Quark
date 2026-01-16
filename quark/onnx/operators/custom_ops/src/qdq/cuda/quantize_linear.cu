//
// Modifications Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include <limits>

#include "core/framework/float16.h"
#include "qdq/cuda/quantize_linear.cuh"

#include <cuda_fp16.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif


// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};


template <typename InT, typename OutT>
struct RoundStd;

template <>
struct RoundStd<float, int8_t> {
  __device__ __forceinline__ int8_t operator()(float v, float scale, int8_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<int8_t>::lowest()) value = std::numeric_limits<int8_t>::lowest();
    if (value > std::numeric_limits<int8_t>::max()) value = std::numeric_limits<int8_t>::max();
    return static_cast<int8_t>(value);
  }
};

template <>
struct RoundStd<float, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(float v, float scale, uint8_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<uint8_t>::lowest()) value = std::numeric_limits<uint8_t>::lowest();
    if (value > std::numeric_limits<uint8_t>::max()) value = std::numeric_limits<uint8_t>::max();
    return static_cast<uint8_t>(value);
  }
};

template <>
struct RoundStd<float, int16_t> {
  __device__ __forceinline__ int16_t operator()(float v, float scale, int16_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<int16_t>::lowest()) value = std::numeric_limits<int16_t>::lowest();
    if (value > std::numeric_limits<int16_t>::max()) value = std::numeric_limits<int16_t>::max();
    return static_cast<int16_t>(value);
  }
};

template <>
struct RoundStd<float, uint16_t> {
  __device__ __forceinline__ uint16_t operator()(float v, float scale, uint16_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<uint16_t>::lowest()) value = std::numeric_limits<uint16_t>::lowest();
    if (value > std::numeric_limits<uint16_t>::max()) value = std::numeric_limits<uint16_t>::max();
    return static_cast<uint16_t>(value);
  }
};

template <>
struct RoundStd<float, int32_t> {
  __device__ __forceinline__ int32_t operator()(float v, float scale, int32_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<int32_t>::lowest()) value = std::numeric_limits<int32_t>::lowest();
    if (value > std::numeric_limits<int32_t>::max()) value = std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(value);
  }
};

template <>
struct RoundStd<float, uint32_t> {
  __device__ __forceinline__ uint32_t operator()(float v, float scale, uint32_t zero_point) const {
    int64_t value = static_cast<int64_t>(__float2int_rn(v / scale));
    value = value + zero_point;
    if (value < std::numeric_limits<uint32_t>::lowest()) value = std::numeric_limits<uint32_t>::lowest();
    if (value > std::numeric_limits<uint32_t>::max()) value = std::numeric_limits<uint32_t>::max();
    return static_cast<uint32_t>(value);
  }
};

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelStdInt(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N, RoundStd<InT, OutT> round) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  InT scale = *scale_ptr;
  OutT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : static_cast<OutT>(0);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = round(input[id], scale, zero_point);
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
bool CudaQuantizeLinearStdInt(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelStdInt<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      RoundStd<InT, OutT>());
  return true;
}

template bool CudaQuantizeLinearStdInt<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdInt<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdInt<int16_t, float>(cudaStream_t stream, const float* input, int16_t* output, const float* scale, const int16_t* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdInt<uint16_t, float>(cudaStream_t stream, const float* input, uint16_t* output, const float* scale, const uint16_t* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdInt<int32_t, float>(cudaStream_t stream, const float* input, int32_t* output, const float* scale, const int32_t* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdInt<uint32_t, float>(cudaStream_t stream, const float* input, uint32_t* output, const float* scale, const uint32_t* zero_point, size_t num_of_element);


template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelStdFp(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  InT scale = *scale_ptr;
  OutT zero_point = *zero_point_ptr;  // Do not support asymmetric
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      float float_value = input[id] / scale;
      __half half_value = __float2half(float_value);
      output->val = *reinterpret_cast<uint16_t*>(&half_value);
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
bool CudaQuantizeLinearStdFp(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelStdFp<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));

  cudaDeviceSynchronize();

  return true;
}

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelStdBf(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  InT scale = *scale_ptr;
  OutT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : OutT(0.0f);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = OutT(input[id] / scale + zero_point.ToFloat());
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
bool CudaQuantizeLinearStdBf(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelStdBf<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));

  cudaDeviceSynchronize();

  return true;
}

// Note that onnxruntime::MLFloat16 has CPU implementation only, while onnxruntime::BFloat16 has both CPU and GPU implementation
template bool CudaQuantizeLinearStdFp<onnxruntime::MLFloat16, float>(cudaStream_t stream, const float* input, onnxruntime::MLFloat16* output, const float* scale, const onnxruntime::MLFloat16* zero_point, size_t num_of_element);
template bool CudaQuantizeLinearStdBf<onnxruntime::BFloat16, float>(cudaStream_t stream, const float* input, onnxruntime::BFloat16* output, const float* scale, const onnxruntime::BFloat16* zero_point, size_t num_of_element);


template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelStdInt(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  OutT scale = *scale_ptr;
  InT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : static_cast<InT>(0);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = static_cast<OutT>(input[id] - zero_point) * scale;
      id += NumThreadsPerBlock;
    }
  }
}

template <class InT, class OutT>
bool CudaDequantizeLinearStdInt(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelStdInt<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));
  return true;
}

template bool CudaDequantizeLinearStdInt<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdInt<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdInt<int16_t, float>(cudaStream_t stream, const int16_t* input, float* output, const float* scale, const int16_t* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdInt<uint16_t, float>(cudaStream_t stream, const uint16_t* input, float* output, const float* scale, const uint16_t* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdInt<int32_t, float>(cudaStream_t stream, const int32_t* input, float* output, const float* scale, const int32_t* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdInt<uint32_t, float>(cudaStream_t stream, const uint32_t* input, float* output, const float* scale, const uint32_t* zero_point, size_t num_of_element);


template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelStdFp(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  OutT scale = *scale_ptr;
  InT zero_point = *zero_point_ptr;  // Do not support asymmetric
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      uint16_t half_bits = input[id].val;
      __half half_value = *reinterpret_cast<__half*>(&half_bits);
      float float_value = __half2float(half_value);
      output[id] = OutT(float_value * scale);
      id += NumThreadsPerBlock;
    }
  }
}

template <class InT, class OutT>
bool CudaDequantizeLinearStdFp(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelStdFp<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));

  cudaDeviceSynchronize();

  return true;
}

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelStdBf(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  OutT scale = *scale_ptr;
  InT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : InT(0.0f);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = (input[id].ToFloat() - zero_point.ToFloat()) * scale;
      id += NumThreadsPerBlock;
    }
  }
}

template <class InT, class OutT>
bool CudaDequantizeLinearStdBf(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return false;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelStdBf<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));

  cudaDeviceSynchronize();

  return true;
}

// Note that onnxruntime::MLFloat16 has CPU implementation only, while onnxruntime::BFloat16 has both CPU and GPU implementation
template bool CudaDequantizeLinearStdFp<onnxruntime::MLFloat16, float>(cudaStream_t stream, const onnxruntime::MLFloat16* input, float* output, const float* scale, const onnxruntime::MLFloat16* zero_point, size_t num_of_element);
template bool CudaDequantizeLinearStdBf<onnxruntime::BFloat16, float>(cudaStream_t stream, const onnxruntime::BFloat16* input, float* output, const float* scale, const onnxruntime::BFloat16* zero_point, size_t num_of_element);
