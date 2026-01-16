// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <atlbase.h>
#include <d3d12.h>

#include <ostream>
#include <random>
#include <vector>

#include "DirectML.h"
#include "types.h"

namespace ryzenai::onnx_utils {

class Context;
class Tensor;
struct TensorDesc;

// We will use these vectors frequently so create some typedefs. They need to
// store pointers so that we can put holes in the vector where our Context must
// insert "none" DirectML bindings.
typedef std::vector<std::shared_ptr<TensorDesc>> TensorDescVector;
typedef std::vector<std::shared_ptr<Tensor>> TensorVector;

// All information needed to describe a single dimension in a tensor.
struct TensorDim {
  int64 size;  // The logical size of this dimension of the tensor in elements.
  int64 stride;  // The absolute distance between elements in this dimension in
                 // elements.
  int32 order;  // Where this dimension is in the tensor's memory packing order.
                // Zero means this is the outer most dimension, one is the next
                // largest, etc. An NCHW tensor has orders of [0, 1, 2, 3].
};

// All information needed to describe a tensor.
struct TensorDesc {
  hstring name;  // The name of the tensor.
  std::vector<TensorDim>
    dims;  // The size, stride, etc. of each dimension of the tensor.
  DML_TENSOR_DATA_TYPE dataType;    // The element data type.
  int64 totalBytes;                 // The total size of the tensor in bytes.
  bool isConst;                     // If the tensor is constant.
  bool isOutputToNextNode = false;  // If the tensor is output to next node.
  bool isInputFromPrevNode =
    false;  // If the tensor is input from previous node.
  bool createResource = false;
  bool isLastDimOdd = false;  // If the tensor needs repacking
};

// =====================================================================================================================
// A DirectML buffer tensor. Accessible by the CPU and GPU.
class Tensor {
 public:
  static int32 ElementSizeBits(DML_TENSOR_DATA_TYPE dataType);
  static int32 ElementSize(DML_TENSOR_DATA_TYPE dataType);
  static DXGI_FORMAT ElementFormat(DML_TENSOR_DATA_TYPE dataType);

  // Returns the index of the dimension with the given order. Negative values
  // start from inside, for example if there are four dimensions then an order
  // of -1 really means order 3, the innermost dimension.
  static int32 DimIndex(const TensorDesc& desc, int32 order);

  Tensor(
    const TensorDesc& desc, Context* pContext, bool forceCreateD3DResc = false
  );
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ~Tensor();

  //// Get the cpu accesible pointer to tensor's d3d resource.
  // void* GetMappedTensor();

  const TensorDesc& Desc() const { return m_desc; }

  // The tensor's CPU-visible memory.
  void* Buffer() const { return m_pBuffer; }

  // The tensor's GPU-visible resource.
  CComPtr<ID3D12Resource> Resource() const { return m_resource; }
  void* MappedResource() const { return m_pMappedResource; }
  void SetResource(
    CComPtr<ID3D12Resource> resource, void* pMappedResource,
    uint64_t d3dRescOffset, Context* pContext = nullptr,
    bool ignoreExternalResource = false
  );
  void UpdateResourceData(uint32 data);

  void CreateResource(Context* pContext);

  // Just a helper function to make it easier to call the static version.
  int32 DimIndex(int32 order) const { return DimIndex(m_desc, order); }

  void UpdateTensorBuffer(void* pExternalData);

  bool IsUploaded() const { return m_isUploaded; }
  void SetUploaded(bool uploaded) { m_isUploaded = uploaded; }

  void SetConstFlag(bool isConst) { m_isBufferConst = isConst; }
  bool GetConstFlag() const { return m_isBufferConst; }
  uint64_t ResourceOffset() const { return m_d3dRescOffset; }

  bool IsResourceUpdateNeeded() { return m_updateResourceFlag; }

  // Repack int4/uint4 tensor buffer to remove padding
  void RepackInt4TensorBuffer(void* pExternalData);
  const bool IsRepackNeeded() { return m_desc.isLastDimOdd; }

  // Jit
  void SetJitConstFlag(bool isJitConst) { m_isJitConst = isJitConst; }
  bool IsJitConst() const { return m_isJitConst; }
  void ReleaseResource() {
    if (m_resource) {
      m_resource.Release();
      m_resource = nullptr;
    }
    m_pMappedResource = nullptr;
  }

  // Local Window Attention related
  void SetAttnOffsetAndSize(const uint64_t offset, const uint64_t size) {
    m_localWindowOffset = offset;
    m_useLocalWindowAttention = true;
    m_AttnWindowSizeInBytes = size;
  }
  uint64_t GetLocalWindowOffset() const { return m_localWindowOffset; }

  bool UseLocalWindowAttn() const { return m_useLocalWindowAttention; }
  // void SetAttentionWindowSize(uint64_t size) { m_AttnWindowSizeInBytes =
  // size; }
  uint64_t GetAttentionWindowSize() const { return m_AttnWindowSizeInBytes; }

 private:
  const TensorDesc m_desc;  // The tensor's descriptor.
  // Use a single persistently mapped system memory allocation.
  const bool m_systemMem;
  void* m_pBuffer;  // A read/write pointer to the tensor's CPU data buffer.
  // don't free/delete If m_pBuffer comes externally from application
  bool m_ispBufferExternal = true;
  bool m_isBufferConst;
  // A clone of the staging resource in the default heap.
  CComPtr<ID3D12Heap> m_heap;
  // A placed resource used to access our GPU heap. It must be manually uploaded
  // and downloaded to keep the CPU data in sync.
  CComPtr<ID3D12Resource> m_resource;
  // The CPU-visible pointer to the GPU resource.
  void* m_pMappedResource = nullptr;
  bool m_isUploaded;  // Check if the constants / weight tensor is uploaded
  uint64_t m_d3dRescOffset = 0;  // Offset of the resource in the heap.
  bool m_updateResourceFlag =
    true;  // Flag to check if the resource update is required

  // Jit
  bool m_isJitConst = false;

  // repack
  bool m_tensorRepacked = false;

  // local window attention
  bool m_useLocalWindowAttention = false;
  uint64_t m_localWindowOffset = 0;
  uint64_t m_AttnWindowSizeInBytes = 0;
};

}  // namespace ryzenai::onnx_utils
