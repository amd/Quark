// Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <atlbase.h>
#include <d3d12.h>
#include <ryzenai/ryzen_mm.h>

#include "types.h"

namespace ryzenai::onnx_utils {
struct OnnxTensorInfo {
  int index;
  std::string name;
  std::vector<int64_t> shape;
  hstring dataType;
  uint64_t offsetInBytes;
  int64_t elementCount;
  uint32_t heapId;
  RyzenMM::BufferRef buffer;
  CComPtr<ID3D12Resource> d3dResource;
  void* pCpuMappedD3DResc;
  void* pExternalBuffer;
  bool isExternalBufferConstant;
  bool rebindD3DResc;
  bool isConstForJit;
  bool isDynamicTensor;

  const auto& UseRMMBuffer(RyzenMM::BufferRef inp) {
    buffer = std::move(inp);
    d3dResource.Attach(
      RyzenMM::Platform::DX::GetUnderlyingD3D12Resource(buffer)
    );
    pCpuMappedD3DResc = buffer.Data();
    return buffer;
  }

  void UseRMMAllocatedMemory(const void* ptr) {
    d3dResource.Attach(
      RyzenMM::Platform::DX::GetUnderlyingD3D12Resource(
        RyzenMM::NonResidentBufferRef::FromUnmanagedBuffer(
          RyzenMM::UnmanagedBufferPtr(ptr), offsetInBytes
        )
      )
    );

    void* ptr_w_offset = const_cast<void*>(ptr);
    std::uint8_t* base_ptr = (std::uint8_t*)ptr_w_offset - offsetInBytes;
    pCpuMappedD3DResc = (void*)base_ptr;
  }

  OnnxTensorInfo() {
    name = "";
    shape = {};
    dataType = L"";
    offsetInBytes = 0;
    elementCount = 0;
    heapId = 0;
    buffer = {};
    d3dResource = nullptr;
    pCpuMappedD3DResc = nullptr;
    pExternalBuffer = nullptr;
    isExternalBufferConstant = false;
    rebindD3DResc = false;
    isConstForJit = false;
    isDynamicTensor = false;
  }
};

struct AttributeForSSMLP {
  int64 gateBits;
  int64 upBits;
  int64 downBits;

  int64 gateBlockSize;
  int64 upBlockSize;
  int64 downBlockSize;

  int64 gate_K;
  int64 down_K;
  int64 up_K;

  int64 gate_N;
  int64 down_N;
  int64 up_N;

  int64 has_gelu;
  float epsilon;
};

struct AttributeForGQO {
  int64 do_rotary;
  int64 head_size;
  int64 kv_num_heads;
  int64 local_window_size = -1;
  int64 num_heads;
  int64 o_proj_bits;
  int64 o_proj_block_size;
  int64 o_proj_K;
  int64 o_proj_N;
  int64 rotary_interleaved = 0;
  int64 rotary_embedding_dim = 0;
  float scale;
  float softcap;
};

struct AttributeForSlrn {
  float epsilon;
  std::vector<int64> shapeIn;
  std::vector<int64> shapeOut;
};

static ULONG GetRefCount(ID3D12Resource* resc) {
  if (!resc) {
    return 0;
  }
  ULONG refCount = resc->AddRef();
  refCount = resc->Release();
  return refCount;
}
}  // namespace ryzenai::onnx_utils
