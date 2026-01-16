// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "tensor.h"

#include <iomanip>
#include <type_traits>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Computes the size of a data type in bits.
int32 Tensor::ElementSizeBits(DML_TENSOR_DATA_TYPE dataType) {
  int32 size = 0;

  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      size = sizeof(uint32) * 8;
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      size = sizeof(uint16) * 8;
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
      size = sizeof(uint8) * 8;
      break;
    case DML_TENSOR_DATA_TYPE_INT4:
    case DML_TENSOR_DATA_TYPE_UINT4:
      size = 4;
      break;

    case DML_TENSOR_DATA_TYPE_INT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
      size = sizeof(uint64) * 8;
      break;

    default:
      throw std::runtime_error("Unknown tensor data type.");
      break;
  }

  return size;
}

// =====================================================================================================================
// Computes the size of a data type in bytes.
int32 Tensor::ElementSize(DML_TENSOR_DATA_TYPE dataType) {
  int32 size = 0;

  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      size = sizeof(uint32);
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      size = sizeof(uint16);
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
    case DML_TENSOR_DATA_TYPE_INT4:
    case DML_TENSOR_DATA_TYPE_UINT4:
      size = sizeof(uint8);
      break;

    case DML_TENSOR_DATA_TYPE_INT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
      size = sizeof(uint64);
      break;

    default:
      throw std::runtime_error("Unknown tensor data type.");
      break;
  }

  return size;
}

// =====================================================================================================================
// Finds a DXGI format compatible with the given data type.
DXGI_FORMAT Tensor::ElementFormat(DML_TENSOR_DATA_TYPE dataType) {
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      format = DXGI_FORMAT_R32_FLOAT;
      break;

    case DML_TENSOR_DATA_TYPE_UINT32:
      format = DXGI_FORMAT_R32_UINT;
      break;

    case DML_TENSOR_DATA_TYPE_INT32:
      format = DXGI_FORMAT_R32_SINT;
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
      format = DXGI_FORMAT_R16_FLOAT;
      break;

    case DML_TENSOR_DATA_TYPE_UINT16:
      format = DXGI_FORMAT_R16_UINT;
      break;

    case DML_TENSOR_DATA_TYPE_INT16:
      format = DXGI_FORMAT_R16_SINT;
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_UINT4:
      format = DXGI_FORMAT_R8_UINT;
      break;

    case DML_TENSOR_DATA_TYPE_INT8:
      format = DXGI_FORMAT_R8_SINT;
      break;

    case DML_TENSOR_DATA_TYPE_INT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
      throw std::runtime_error("64-bit integer format not directly supported.");
      break;

    default:
      throw std::runtime_error("Unknown tensor data type.");
      break;
  }

  return format;
}

// =====================================================================================================================
// It's somewhat common to query the size, stride, etc., of a dimension relative
// to its physical memory ordering instead of its logcal ordering. This function
// finds the logical dimension index of a given physical order. Negative values
// start from inside, for example if there are four dimensions then an order of
// -1 really means order 3, the innermost dimension.
int32 Tensor::DimIndex(const TensorDesc& desc, int32 order) {
  // Convert a negative value into a proper non-zero order.
  const int32 size = static_cast<int32>(desc.dims.size());

  if (order < 0) {
    order += size;
  }

  if ((order < 0) || (order >= size)) {
    throw std::runtime_error("Tensor::DimIndex: order is out of bounds!");
  }

  int32 index = -1;

  for (int32 idx = 0; idx < size; ++idx) {
    if (desc.dims[idx].order == order) {
      index = idx;
      break;
    }
  }

  // It should be impossible for us to not find a legal order value.
  assert(index >= 0);

  return index;
}

void Tensor::SetResource(
  CComPtr<ID3D12Resource> resource, void* pMappedResource,
  uint64_t d3dRescOffset, Context* pContext, bool ignoreExternalResource
) {
  if (ignoreExternalResource == false) {
    m_resource = resource;
    m_d3dRescOffset = d3dRescOffset;
    m_pMappedResource = pMappedResource;
  } else {
    CreateResource(pContext);
  }
}

// =====================================================================================================================
void Tensor::UpdateResourceData(uint32 data) {
  if (m_pMappedResource != nullptr) {
    uint32* rsrc = (uint32*)m_pMappedResource;
    *rsrc = data;
  } else {
    throw std::runtime_error("Mapped resource is null, cannot update data.");
  }
}

void Tensor::CreateResource(Context* pContext) {
  // Create a default heap resource to accelerate GPU access during model
  // execution. We create it as a placed resources so that we can do things like
  // disable the heap's zero-init phase.

  CComPtr<ID3D12Device3> d3d12Device = pContext->D3d12Device();

  // Make sure we are not overridding the mapped resource if it is already set.
  if (m_pMappedResource) {
    throw std::runtime_error("Shared memory Resource already mapped");
  }

  if (!m_resource) {
    // If the tensor is not constant, we need to allocate a heap for it.
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    resourceDesc.Width = static_cast<UINT64>(m_desc.totalBytes);
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;

    const D3D12_RESOURCE_ALLOCATION_INFO allocInfo =
      d3d12Device->GetResourceAllocationInfo(0, 1, &resourceDesc);

    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = allocInfo.SizeInBytes;
    heapDesc.Alignment = allocInfo.Alignment;
    heapDesc.Flags =
      D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;

    if (m_systemMem) {
      // Force the device to allocate system memory.
      heapDesc.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
      heapDesc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      heapDesc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    } else {
      // Use the default heap (probably CPU invisible GPU local memory).
      heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;

      if (pContext->GetInfo().noPaging == false) {
        // Optimize the runtime's residency paging overhead as well.
        heapDesc.Flags |= D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT;
      }
    }

    if (FAILED(d3d12Device->CreateHeap(&heapDesc, IID_PPV_ARGS(&m_heap)))) {
      throw std::runtime_error("Failed to create a tensor's default heap.");
    }

    if (FAILED(d3d12Device->CreatePlacedResource(
          m_heap, 0, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
          IID_PPV_ARGS(&m_resource)
        ))) {
      throw std::runtime_error("Failed to create a tensor's placed resource.");
    }
  }

  if (m_systemMem) {
    // Just map the buffer resource to get our CPU mapping.
    if (FAILED(m_resource->Map(0, nullptr, &m_pMappedResource))) {
      throw std::runtime_error(
        "Failed to map the system memory placed resource."
      );
    }
    ZeroMemory(m_pMappedResource, m_desc.totalBytes);
  } else {
    if (pContext->GetInfo().noPaging == false) {
      // Kick off an async make resident so that we can keep doing work while
      // it's being paged in.
      pContext->MakeResident(m_heap);
    }

    // if (false == m_ispBufferExternal)  // if we are using external memory
    //                                    // which has tensor data.
    //{
    //   // Allocate a full-sized CPU-side buffer for uploading initial data,
    //   // downloading results, and validation.
    //   m_pBuffer = malloc(static_cast<size_t>(m_desc.totalBytes));

    //  if (m_pBuffer == nullptr) {
    //    throw std::runtime_error(
    //      "Failed to create a tensor's CPU-side buffer.");
    //  }
    //}
  }
}

// =====================================================================================================================
Tensor::Tensor(
  const TensorDesc& desc, Context* pContext, bool forceCreateD3DResc
)
  : m_desc(desc),
    m_systemMem(pContext->GetInfo().systemMem),
    m_pBuffer(nullptr),
    m_isUploaded(false),
    m_isBufferConst(false),
    m_ispBufferExternal(pContext->GetInfo().useExternalMemory) {
  CComPtr<ID3D12Device3> d3d12Device = pContext->D3d12Device();

  if (d3d12Device == nullptr) {
    throw std::runtime_error("We must have a D3D12 device to create tensors.");
  }

  if (forceCreateD3DResc || m_desc.createResource) {
    // Don't update the resource later when creating locally
    m_updateResourceFlag = !m_desc.createResource;
    CreateResource(pContext);
  }
}

// =====================================================================================================================
void Tensor::RepackInt4TensorBuffer(void* pExternalData) {
  // Return if tensor is already repacked
  if (m_tensorRepacked) {
    return;
  }

  // check if the data type is int4/ uint4
  if (!(m_desc.dataType == DML_TENSOR_DATA_TYPE_INT4 ||
        m_desc.dataType == DML_TENSOR_DATA_TYPE_UINT4)) {
    return;
  }

  // each byte contains 2 nibbles, 0/low and 1/high
  // low nibble : bits 0-3, high nibble : bits 4-7
  uint8_t* srcDataPtr = static_cast<uint8_t*>(pExternalData);
  size_t totaldims = m_desc.dims.size();
  int64 rows = m_desc.dims[totaldims - 2].size;
  int64 cols = m_desc.dims[totaldims - 1].size;

  for (size_t r = 0; r < rows; ++r) {
    // src contains padded nibble
    const size_t srcRowBaseNib = r * (cols + 1);
    const size_t dstRowBaseNib = r * cols;

    for (size_t c = 0; c < cols; ++c) {
      // get the nibble from src
      const size_t srcNib = srcRowBaseNib + c;
      // get the byte index and low/high nibble
      const size_t srcDataIdx = srcNib >> 1;
      // low nibble if 0, high nibble if 1
      const bool srcNibLow = (srcNib & 1) == 0;

      const uint8_t srcByte = srcDataPtr[srcDataIdx];

      // extract the nibble
      uint8_t nibble = srcNibLow ? (srcByte & 0x0F) : (srcByte >> 4);

      const size_t dstNib = dstRowBaseNib + c;

      const size_t destDataIdx = dstNib >> 1;
      const bool dstNibLow = (dstNib & 1) == 0;
      uint8_t* dstDataPtr = static_cast<uint8_t*>(m_pMappedResource);
      uint8_t& dstByte = dstDataPtr[destDataIdx];

      // keep only the 4 bits
      nibble &= 0x0F;

      // set the nibble in dest
      if (dstNibLow) {
        dstByte = (uint8_t)((dstByte & 0xF0) | nibble);
      } else {
        dstByte = (uint8_t)((dstByte & 0x0F) | (nibble << 4));
      }
    }
  }

  // Since the resource is now local, it
  // shouldn't be updated with the resource
  // created for jit. Hence, it is marked as
  // non jit resource and offset is set to 0
  m_tensorRepacked = true;
  m_isJitConst = false;
  m_d3dRescOffset = 0;
}

// =====================================================================================================================
Tensor::~Tensor() {
  // if (m_systemMem) {
  //   m_resource->Unmap(0, nullptr);
  // } else if (false == m_ispBufferExternal) {
  //   // only free it, if its not externally passed pointer to tensor data
  //   free(m_pBuffer);
  // }
}

void Tensor::UpdateTensorBuffer(void* pExternalData) {
  m_pBuffer = pExternalData;
  m_ispBufferExternal = true;
}

}  // namespace ryzenai::onnx_utils
