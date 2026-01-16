// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#include "operator.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "context.h"

namespace ryzenai::onnx_utils {

// This array defines the official mapping between DML_TENSOR_DATA_TYPE and the
// data type parameter strings.
static const std::array<hstring, 14> DataTypeStrings = {
  L"Unknown",  // DML_TENSOR_DATA_TYPE_UNKNOWN = 0,
  L"Float",    // DML_TENSOR_DATA_TYPE_FLOAT32 = 1,
  L"Float16",  // DML_TENSOR_DATA_TYPE_FLOAT16 = 2,
  L"Uint32",   // DML_TENSOR_DATA_TYPE_UINT32  = 3,
  L"Uint16",   // DML_TENSOR_DATA_TYPE_UINT16  = 4,
  L"Uint8",    // DML_TENSOR_DATA_TYPE_UINT8   = 5,
  L"Int32",    // DML_TENSOR_DATA_TYPE_INT32   = 6,
  L"Int16",    // DML_TENSOR_DATA_TYPE_INT16   = 7,
  L"Int8",     // DML_TENSOR_DATA_TYPE_INT8    = 8,
  L"Float64",  // DML_TENSOR_DATA_TYPE_FLOAT64 = 9,
  L"Uint64",   // DML_TENSOR_DATA_TYPE_UINT64  = 10,
  L"Int64",    // DML_TENSOR_DATA_TYPE_INT64   = 11,
  L"Uint4",    // DML_TENSOR_DATA_TYPE_UINT4   = 12,
  L"Int4",     // DML_TENSOR_DATA_TYPE_INT4    = 13
};

static const size_t NumDataTypeStrings = DataTypeStrings.size();

// =====================================================================================================================
// Helper function which translates DATA_TYPE enum values into parameter
// strings.
hstring DmlOperator::DataTypeToString(DML_TENSOR_DATA_TYPE dataType) {
  if (dataType > NumDataTypeStrings) {
    std::wstringstream ss;
    ss << L"Unknown data type: " << dataType;
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  return DataTypeStrings[dataType];
}

// =====================================================================================================================
// Helper function which matches parameter strings to DATA_TYPE enum values.
DML_TENSOR_DATA_TYPE DmlOperator::StringToDataType(hstring dataTypeStr) {
  DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

  for (int32 idx = 0; idx < NumDataTypeStrings; ++idx) {
    if (dataTypeStr == DataTypeStrings[idx]) {
      dataType = static_cast<DML_TENSOR_DATA_TYPE>(idx);
      break;
    }
  }

  if (dataType == DML_TENSOR_DATA_TYPE_UNKNOWN) {
    std::wstringstream ss;
    ss << L"Unknown data type: " << dataTypeStr.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  return dataType;
}

// This array defines the official mapping between ActivationFunc and the
// activation function parameter strings.
static const hstring ActivationFuncStrings[] = {
  L"None",         // None        = 0,
  L"Elu",          // Elu         = 1,
  L"HardSigmoid",  // HardSigmoid = 2,
  L"LeakyRelu",    // LeakyRelu   = 3,
  L"Linear",       // Linear      = 4,
  L"PSoftPlus",    // PSoftPlus   = 5,
  L"Relu",         // Relu        = 6,
  L"Selu",         // Selu        = 7,
  L"ScaledTanh",   // ScaledTanh  = 8,
  L"Sigmoid",      // Sigmoid     = 9,
  L"SoftPlus",     // SoftPlus    = 10,
  L"SoftSign",     // SoftSign    = 11,
  L"Tanh",         // Tanh        = 12,
  L"TRelu",        // TRelu       = 13
};

static_assert(
  sizeof(ActivationFuncStrings) / sizeof(ActivationFuncStrings[0]) ==
    static_cast<int32>(ActivationFunc::Count),
  "ActivationFuncStrings is out of date."
);

DmlOperator::DmlOperator()
  : m_descCount(0), m_temporaryResSize(0), m_persistentResSize(0) {}

DmlOperator::~DmlOperator() {
  m_operator.Release();
  m_bindingTable.Release();
  m_temporaryRes.Release();
  m_persistentRes.Release();
  m_descHeap.Release();
}

void DmlOperator::RebindInputAndOutput() {
  BindInputs(m_bindingTable, m_inputAndConstTensorVector);
  BindOutputs(m_bindingTable, m_outputTensorVector);
}

void DmlOperator::RebindOutputs() {
  BindOutputs(m_bindingTable, m_outputTensorVector);
}

// =====================================================================================================================
// Binds a compiled operator and its input and output tensors. This will create
// and initialize GPU state.
void DmlOperator::InitializeAndBindOperator(Context* pCtx) {
  assert(pCtx != nullptr);

  // First, make sure any async make resident calls are done.
  // pCtx->WaitForFence(pCtx->GetPagingFence(), pCtx->GetPagingValue());

  // Initialize the operator and create the necessary helper resources.
  CComPtr<IDMLOperatorInitializer> opInitializer;
  IDMLCompiledOperator* const pOperator = m_operator;

  if (FAILED(pCtx->DmlDevice()->CreateOperatorInitializer(
        1, &pOperator, IID_PPV_ARGS(&opInitializer)
      ))) {
    throw std::runtime_error(
      "Failed to create a DirectML operator initializer."
    );
  }

  const DML_BINDING_PROPERTIES initializeProps =
    opInitializer->GetBindingProperties();
  const DML_BINDING_PROPERTIES executeProps =
    m_operator->GetBindingProperties();

  // The descriptor set and temporary resource is used for initialization and
  // execution. The persistent resource is only needed for execution.
  const UINT neededDescriptors = Max(
    initializeProps.RequiredDescriptorCount,
    executeProps.RequiredDescriptorCount
  );
  const UINT64 neededTempSize = Max(
    initializeProps.TemporaryResourceSize, executeProps.TemporaryResourceSize
  );

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.SampleDesc.Count = 1;

  if (neededTempSize > m_temporaryResSize) {
    // Recreate the temporary resource.
    m_temporaryRes.Release();
    m_temporaryResSize = neededTempSize;
    resourceDesc.Width = neededTempSize;

    if (FAILED(pCtx->D3d12Device()->CreateCommittedResource(
          &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
          D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_temporaryRes)
        ))) {
      throw std::runtime_error(
        "Failed to create the DirectML temporary buffer."
      );
    }
  }

  if (executeProps.PersistentResourceSize > m_persistentResSize) {
    // Recreate the persistent resource.
    m_persistentRes.Release();
    m_persistentResSize = executeProps.PersistentResourceSize;
    resourceDesc.Width = executeProps.PersistentResourceSize;

    if (FAILED(pCtx->D3d12Device()->CreateCommittedResource(
          &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
          D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_persistentRes)
        ))) {
      throw std::runtime_error(
        "Failed to create the DirectML persistent buffer."
      );
    }
  }

  if (neededDescriptors > m_descCount) {
    // Recreate the descriptor heap.
    m_descHeap.Release();
    m_descCount = neededDescriptors;

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = neededDescriptors;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    if (FAILED(pCtx->D3d12Device()->CreateDescriptorHeap(
          &desc, IID_PPV_ARGS(&m_descHeap)
        ))) {
      throw std::runtime_error("Failed to create a descriptor heap.");
    }
  }

  DML_BINDING_TABLE_DESC tableDesc = {};
  tableDesc.Dispatchable = opInitializer;
  tableDesc.CPUDescriptorHandle =
    m_descHeap->GetCPUDescriptorHandleForHeapStart();
  tableDesc.GPUDescriptorHandle =
    m_descHeap->GetGPUDescriptorHandleForHeapStart();
  tableDesc.SizeInDescriptors = m_descCount;

  if (m_bindingTable == nullptr) {
    // Create the binding table the first time we bind an operator.
    if (FAILED(pCtx->DmlDevice()->CreateBindingTable(
          &tableDesc, IID_PPV_ARGS(&m_bindingTable)
        ))) {
      throw std::runtime_error("Failed to create a DirectML binding table.");
    }
  } else {
    if (FAILED(m_bindingTable->Reset(&tableDesc))) {
      throw std::runtime_error("Failed to reset a DirectML binding table.");
    }
  }

  // Set up the binding table for initialization, which computes the persistent
  // buffer contents.
  if (initializeProps.TemporaryResourceSize > 0) {
    DML_BINDING_DESC desc = {};
    DML_BUFFER_BINDING binding = {};
    desc.Type = DML_BINDING_TYPE_BUFFER;
    desc.Desc = &binding;
    binding.Buffer = m_temporaryRes;
    binding.SizeInBytes = initializeProps.TemporaryResourceSize;

    m_bindingTable->BindTemporaryResource(&desc);
  }

  if (executeProps.PersistentResourceSize > 0) {
    DML_BINDING_DESC desc = {};
    DML_BUFFER_BINDING binding = {};
    desc.Type = DML_BINDING_TYPE_BUFFER;
    desc.Desc = &binding;
    binding.Buffer = m_persistentRes;
    binding.SizeInBytes = executeProps.PersistentResourceSize;

    m_bindingTable->BindOutputs(1, &desc);
  }

  // Build, submit, and wait on initialization.
  if (FAILED(pCtx->GetExecCmdAlloc()->Reset())) {
    throw std::runtime_error(
      "Failed to reset the execution command allocator."
    );
  }

  if (FAILED(pCtx->GetExecCmdList()->Reset(pCtx->GetExecCmdAlloc(), nullptr))) {
    throw std::runtime_error(
      "Failed to reset the execution command list for initialization."
    );
  }

  ID3D12DescriptorHeap* const pDescHeap = m_descHeap;
  pCtx->GetExecCmdList()->SetDescriptorHeaps(1, &pDescHeap);
  pCtx->GetDmlRecorder()->RecordDispatch(
    pCtx->GetExecCmdList(), opInitializer, m_bindingTable
  );

  if (FAILED(pCtx->GetExecCmdList()->Close())) {
    throw std::runtime_error(
      "Failed to record the execution command list for initialization."
    );
  }

  ID3D12CommandList* pInitCmdLists[] = {pCtx->GetExecCmdList()};
  pCtx->GetEvalQueue()->ExecuteCommandLists(
    sizeof(pInitCmdLists) / sizeof(pInitCmdLists[0]), pInitCmdLists
  );

  pCtx->SignalBinaryFence(pCtx->GetEvalQueue(), pCtx->GetEvalFence());
  pCtx->WaitForFence(pCtx->GetEvalFence(), 1);

  // Now reset the binding table and get it ready for execution.
  tableDesc.Dispatchable = m_operator;

  if (FAILED(m_bindingTable->Reset(&tableDesc))) {
    throw std::runtime_error(
      "Failed to reset a DirectML binding table for execution."
    );
  }

  if (executeProps.TemporaryResourceSize > 0) {
    DML_BINDING_DESC desc = {};
    DML_BUFFER_BINDING binding = {};
    desc.Type = DML_BINDING_TYPE_BUFFER;
    desc.Desc = &binding;
    binding.Buffer = m_temporaryRes;
    binding.SizeInBytes = executeProps.TemporaryResourceSize;

    m_bindingTable->BindTemporaryResource(&desc);
  }

  if (executeProps.PersistentResourceSize > 0) {
    DML_BINDING_DESC desc = {};
    DML_BUFFER_BINDING binding = {};
    desc.Type = DML_BINDING_TYPE_BUFFER;
    desc.Desc = &binding;
    binding.Buffer = m_persistentRes;
    binding.SizeInBytes = executeProps.PersistentResourceSize;

    m_bindingTable->BindPersistentResource(&desc);
  }

  BindInputs(m_bindingTable, m_inputAndConstTensorVector);
  BindOutputs(m_bindingTable, m_outputTensorVector);

  if (pCtx->m_info.systemMem == false) {
    // Reallocate the staging resources up to some max size. We will do multiple
    // copies for really big tensors.
    UINT64 maxTensorSize = 0;

    for (size_t idx = 0; idx < m_inputAndConstTensorVector.size(); ++idx) {
      if (m_inputAndConstTensorVector[idx] != nullptr) {
        maxTensorSize = Max(
          maxTensorSize, static_cast<UINT64>(
                           m_inputAndConstTensorVector[idx]->Desc().totalBytes
                         )
        );
      }
    }

    for (size_t idx = 0; idx < m_outputTensorVector.size(); ++idx) {
      if (m_outputTensorVector[idx] != nullptr) {
        maxTensorSize = Max(
          maxTensorSize,
          static_cast<UINT64>(m_outputTensorVector[idx]->Desc().totalBytes)
        );
      }
    }

    const UINT64 resSize = Min(maxTensorSize, MaxCopyResSize);

    if (pCtx->m_copyResourceSize < resSize) {
      pCtx->m_copyResourceSize = resSize;
      resourceDesc.Width = resSize;

      D3D12_HEAP_PROPERTIES stagingHeapProps = {};
      stagingHeapProps.Type = D3D12_HEAP_TYPE_CUSTOM;
      stagingHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      stagingHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;

      for (int32 idx = 0; idx < NumCopyState; ++idx) {
        if (pCtx->m_copyState[idx].pBuffer != nullptr) {
          pCtx->m_copyState[idx].resource->Unmap(0, nullptr);
          pCtx->m_copyState[idx].pBuffer = nullptr;
        }

        pCtx->m_copyState[idx].resource.Release();

        if (FAILED(pCtx->D3d12Device()->CreateCommittedResource(
              &stagingHeapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
              D3D12_RESOURCE_STATE_COMMON, nullptr,
              IID_PPV_ARGS(&pCtx->m_copyState[idx].resource)
            ))) {
          throw std::runtime_error(
            "Failed to create the copy staging resource."
          );
        }

        if (FAILED(pCtx->m_copyState[idx].resource->Map(
              0, nullptr, &pCtx->m_copyState[idx].pBuffer
            ))) {
          throw std::runtime_error("Failed to map the copy staging resource.");
        }
      }
    }
  }
}
// =====================================================================================================================
// Helper function which translates DATA_TYPE enum values into parameter
// strings.
hstring DmlOperator::ActivationFuncToString(ActivationFunc func) {
  if (func >= ActivationFunc::Count) {
    std::wstringstream ss;
    ss << L"Unknown data type: " << static_cast<int32>(func);
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  return DataTypeStrings[static_cast<int32>(func)];
}

// =====================================================================================================================
// Helper function which matches parameter strings to ActivationFunc enum
// values.
ActivationFunc DmlOperator::StringToActivationFunc(hstring activationFuncStr) {
  ActivationFunc func = ActivationFunc::Count;

  for (int32 idx = 0; idx < static_cast<int32>(ActivationFunc::Count); ++idx) {
    if (activationFuncStr == ActivationFuncStrings[idx]) {
      func = static_cast<ActivationFunc>(idx);
      break;
    }
  }

  if (func == ActivationFunc::Count) {
    std::wstringstream ss;
    ss << L"Unknown activation function: " << activationFuncStr.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  return func;
}

// =====================================================================================================================
// Helper function which computes the stride of each dimension in a tensor.
std::vector<int64> DmlOperator::ComputeStrides(
  const std::vector<int64>& shape
) {
  std::vector<int64> strides(shape.size(), -1);
  auto nonBroadcastDimsIter = shape.rbegin();
  uint32_t elementCount = 1;

  for (int descDimIndex = shape.size() - 1; descDimIndex >= 0; --descDimIndex) {
    if (nonBroadcastDimsIter == shape.rend() || (*nonBroadcastDimsIter == 1)) {
      strides[descDimIndex] = 0;
    } else {
      strides[descDimIndex] = elementCount;
      elementCount *= (*nonBroadcastDimsIter);
    }

    if (nonBroadcastDimsIter != shape.rend()) {
      ++nonBroadcastDimsIter;
    }
  }

  return strides;
}

// =====================================================================================================================
void DmlOperator::MarkRepackIfOddDim(
  std::shared_ptr<TensorDesc>& pDesc, bool isGpuJit
) {
  auto& tensorDesc = pDesc;
  // check if last dim is odd
  if ((tensorDesc->dims.back().size % 2 != 0)) {
    tensorDesc->createResource = true;
    tensorDesc->isLastDimOdd = true;
  }
}

// =====================================================================================================================
// Helper function which creates a new TensorDesc based on its parameters.
std::shared_ptr<TensorDesc> DmlOperator::CreateTensorDesc(
  hstring name,
  hstring order,  // The dimension ordering (e.g., NCHW).
  DML_TENSOR_DATA_TYPE dataType,
  const std::vector<int64>& shape,  // The size of each dimension of the tensor.
  const std::vector<int64>& strides
)  // Each dimension's stride in elements or
   // -1 for an automatic stride.
{
  if (shape.size() != strides.size()) {
    std::wstringstream ss;
    ss << L"Tensor: " << name.c_str()
       << L". Tensor shapes and strides must have the same length.";
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // The order string tells us how our logical dimensions are ordered in the
  // physical tensor. For example, an NWHC order means that the logical C
  // dimension is physically the innermost dimension (smallest stride). The
  // identity order strings are any subset of NCDHW or 01234.
  //
  // We want to convert our order string into an array of integers which maps
  // physical dimensions (index) to logical dimensions (value), sorted from
  // outermost to innermost. For example, NWHC would become [0, 2, 3, 1]. This
  // array will be useful in validating strides and sizes because you can plug
  // its values into our other dimension arrays to look up the size or stride of
  // the given physical dimension.
  //
  // The values [0, size() - 1] must show up once each to get a proper mapping.
  // If the order string has more characters than we have dimensions we ignore
  // the characters on the left. If the order string has fewer characters than
  // we have dimensions we pad left with a linear ordering. Thus, a 3D tensor
  // with NCHWD ordering becomes [1, 2, 0], and a 5D tensor with HWD becomes [0,
  // 1, 3, 4, 2].
  std::vector<size_t> orderInts(shape.size());
  std::wstring orderStr(order);
  size_t padSize = 0;

  if (orderStr.length() > orderInts.size()) {
    // Chop off any extra characters on the left.
    orderStr = orderStr.substr(orderStr.length() - orderInts.size());
  } else if (orderInts.size() > orderStr.length()) {
    // Fill in the linear ordering padding on the left.
    padSize = orderInts.size() - orderStr.length();

    for (size_t idx = 0; idx < padSize; ++idx) {
      orderInts[idx] = idx;
    }
  }

  // Search the string for valid order characters from the higher dimensions on
  // down. This is where we compact our indices if the user has skipped a
  // dimension for their convenience (e.g., NHW becomes [0, 1, 2]). We will end
  // up assigning each character a unique number in the range [0, len-1] unless
  // the string is invalid.
  const std::wstring orderChars[5] = {L"4Nn", L"3Cc", L"2Dd", L"1Hh", L"0Ww"};
  size_t nextIndex = padSize;

  for (const std::wstring& chars : orderChars) {
    for (wchar_t chr : chars) {
      const size_t strIdx = orderStr.find(chr);

      if (strIdx != std::wstring::npos) {
        orderInts[padSize + strIdx] = nextIndex++;
        break;
      }
    }
  }

  // Make sure we have a valid dimension ordering. The values [0, size() - 1]
  // must show up once each.
  for (size_t val = 0; val < orderInts.size(); ++val) {
    if (std::find(orderInts.begin(), orderInts.end(), val) == orderInts.end()) {
      std::wstringstream ss;
      ss << L"Tensor: " << name.c_str() << L". Invalid order: " << orderStr;
      ss << L". Missing dimension index " << val << ": ";

      for (size_t idx = 0; idx < orderInts.size(); ++idx) {
        ss << orderInts[idx] << ", ";
      }

      throw std::runtime_error(winrt::to_string(ss.str()));
    }
  }

  // Now we can actually build our TensorDesc.
  std::shared_ptr<TensorDesc> pDesc = std::make_shared<TensorDesc>();

  pDesc->name = name;
  pDesc->dataType = dataType;
  pDesc->dims.resize(shape.size());

  for (size_t idx = 0; idx < pDesc->dims.size(); ++idx) {
    pDesc->dims[idx].size = shape[idx];
    pDesc->dims[idx].stride = strides[idx];

    // We invert the usual pattern here because we want each TensorDim to know
    // its sorted index.
    pDesc->dims[orderInts[idx]].order = static_cast<int32>(idx);
  }

  // Now we can walk the tensor's dimensions from innermost to outermost,
  // replacing "-1" with packed strides.
  int64 packedStride = 1;

  for (int32 idx = static_cast<int32>(orderInts.size()) - 1; idx >= 0; --idx) {
    TensorDim& dim = pDesc->dims[orderInts[idx]];

    if (dim.stride < 0) {
      dim.stride = packedStride;
      packedStride *= dim.size;
    } else if (dim.stride > 0) {
      if (dim.stride < packedStride) {
        std::wstringstream ss;
        ss << L"Tensor: " << name.c_str()
           << L". Explicit strides must follow tensor ordering. Stride ";
        ss << dim.stride << " at index " << orderInts[idx]
           << " is too small; it must be at least ";
        ss << packedStride;
        throw std::runtime_error(winrt::to_string(ss.str()));
      }

      packedStride = dim.stride * dim.size;
    }
  }

  int32 elemSize = Tensor::ElementSize(dataType);

  // We must compute the tensor size. The easiest way to calculate the tensor
  // size while accounting for padding and broadcasting is to compute the index
  // of the last element in the tensor plus one:
  //     elements = dot(sizes - 1, strides) + 1
  int64 elements = 1;

  for (size_t idx = 0; idx < pDesc->dims.size(); ++idx) {
    elements += (pDesc->dims[idx].size - 1) * pDesc->dims[idx].stride;
  }

  int unalignedTotalBytes = elements * elemSize;

  // this code is needed, since two int4 takes one int8.
  // IF WE EXPECT DATA AS: ONE INT4 in ONE INT8 size, then we need to skip this
  // if block
  if (dataType == DML_TENSOR_DATA_TYPE_INT4 ||
      dataType == DML_TENSOR_DATA_TYPE_UINT4) {
    unalignedTotalBytes *= 0.5;
  }

  // Note that D3D12 requires that our size is DWORD aligned.
  pDesc->totalBytes = Pow2Align(unalignedTotalBytes, sizeof(uint32));

  return pDesc;
}

// =====================================================================================================================
// Helper function which converts a TensorDesc into a DirectML tensor
// descriptor.
void DmlOperator::ConvertTensorDesc(
  const TensorDesc& tensorDesc, DmlTensorDesc* pDmlDesc
) {
  // Convert the size and stride arrays. DirectML requires us to left-pad up to
  // 4 dimensions.
  const size_t numDimensions = Max<size_t>(4, tensorDesc.dims.size());

  if (numDimensions > 5) {
    throw std::runtime_error("DirectML only supports 4D and 5D tensors.");
  }

  pDmlDesc->sizes.assign(numDimensions, 0);
  pDmlDesc->strides.assign(numDimensions, 0);

  // Copy the source values from the innermost dimension outwards.
  int32 dmlIdx = static_cast<int32>(numDimensions) - 1;
  UINT maxStride = 0;
  UINT padStride = 0;

  for (int32 idx = static_cast<int32>(tensorDesc.dims.size()) - 1; idx >= 0;
       --idx) {
    assert(tensorDesc.dims[idx].size <= UINT_MAX);
    assert(tensorDesc.dims[idx].stride <= UINT_MAX);

    pDmlDesc->sizes[dmlIdx] = static_cast<UINT>(tensorDesc.dims[idx].size);
    pDmlDesc->strides[dmlIdx] = static_cast<UINT>(tensorDesc.dims[idx].stride);

    // Find the largest stride and the size of the dimension it contains. That
    // is our padding stride.
    if (pDmlDesc->strides[dmlIdx] > maxStride) {
      maxStride = pDmlDesc->strides[dmlIdx];
      padStride = pDmlDesc->strides[dmlIdx] * pDmlDesc->sizes[dmlIdx];
    }

    dmlIdx--;
  }

  // Fill out any padding dimensions.
  for (; dmlIdx >= 0; --dmlIdx) {
    pDmlDesc->sizes[dmlIdx] = 1;
    pDmlDesc->strides[dmlIdx] = padStride;
  }

  // Fill out the structs.
  pDmlDesc->desc.Type = DML_TENSOR_TYPE_BUFFER;
  pDmlDesc->desc.Desc = &pDmlDesc->buffer;
  pDmlDesc->buffer.DataType = tensorDesc.dataType;
  pDmlDesc->buffer.Flags = DML_TENSOR_FLAG_NONE;
  pDmlDesc->buffer.DimensionCount = static_cast<UINT>(numDimensions);
  pDmlDesc->buffer.Sizes = pDmlDesc->sizes.data();
  pDmlDesc->buffer.Strides = pDmlDesc->strides.data();

  pDmlDesc->buffer.TotalTensorSizeInBytes =
    static_cast<UINT64>(tensorDesc.totalBytes);
  pDmlDesc->buffer.GuaranteedBaseOffsetAlignment = 0;
}

// =====================================================================================================================
// Helper function which converts an ActivationFuncInfo into a DirectML
// descriptor.
void DmlOperator::ConvertActivationFuncInfo(
  const ActivationFuncInfo& info, DmlActivationFuncDesc* pDmlDesc
) {
  // Note that we set InputTensor and OutputTensor to null because we're fusing
  // this function to some other operator.
  switch (info.func) {
    case ActivationFunc::Elu:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_ELU;
      pDmlDesc->desc.Desc = &pDmlDesc->elu;
      pDmlDesc->elu.InputTensor = nullptr;
      pDmlDesc->elu.OutputTensor = nullptr;
      pDmlDesc->elu.Alpha = info.param1;
      break;

    case ActivationFunc::HardSigmoid:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_HARD_SIGMOID;
      pDmlDesc->desc.Desc = &pDmlDesc->hardSigmoid;
      pDmlDesc->hardSigmoid.InputTensor = nullptr;
      pDmlDesc->hardSigmoid.OutputTensor = nullptr;
      pDmlDesc->hardSigmoid.Alpha = info.param1;
      pDmlDesc->hardSigmoid.Beta = info.param2;
      break;

    case ActivationFunc::LeakyRelu:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_LEAKY_RELU;
      pDmlDesc->desc.Desc = &pDmlDesc->leakyRelu;
      pDmlDesc->leakyRelu.InputTensor = nullptr;
      pDmlDesc->leakyRelu.OutputTensor = nullptr;
      pDmlDesc->leakyRelu.Alpha = info.param1;
      break;

    case ActivationFunc::Linear:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_LINEAR;
      pDmlDesc->desc.Desc = &pDmlDesc->linear;
      pDmlDesc->linear.InputTensor = nullptr;
      pDmlDesc->linear.OutputTensor = nullptr;
      pDmlDesc->linear.Alpha = info.param1;
      pDmlDesc->linear.Beta = info.param2;
      break;

    case ActivationFunc::PSoftPlus:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS;
      pDmlDesc->desc.Desc = &pDmlDesc->paramSoftPlus;
      pDmlDesc->paramSoftPlus.InputTensor = nullptr;
      pDmlDesc->paramSoftPlus.OutputTensor = nullptr;
      pDmlDesc->paramSoftPlus.Alpha = info.param1;
      pDmlDesc->paramSoftPlus.Beta = info.param2;
      break;

    case ActivationFunc::Relu:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_RELU;
      pDmlDesc->desc.Desc = &pDmlDesc->relu;
      pDmlDesc->relu.InputTensor = nullptr;
      pDmlDesc->relu.OutputTensor = nullptr;
      break;

    case ActivationFunc::Selu:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_SCALED_ELU;
      pDmlDesc->desc.Desc = &pDmlDesc->selu;
      pDmlDesc->selu.InputTensor = nullptr;
      pDmlDesc->selu.OutputTensor = nullptr;
      pDmlDesc->selu.Alpha = info.param1;
      pDmlDesc->selu.Gamma = info.param2;
      break;

    case ActivationFunc::ScaledTanh:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_SCALED_TANH;
      pDmlDesc->desc.Desc = &pDmlDesc->scaledTanh;
      pDmlDesc->scaledTanh.InputTensor = nullptr;
      pDmlDesc->scaledTanh.OutputTensor = nullptr;
      pDmlDesc->scaledTanh.Alpha = info.param1;
      pDmlDesc->scaledTanh.Beta = info.param2;
      break;

    case ActivationFunc::Sigmoid:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
      pDmlDesc->desc.Desc = &pDmlDesc->sigmoid;
      pDmlDesc->sigmoid.InputTensor = nullptr;
      pDmlDesc->sigmoid.OutputTensor = nullptr;
      break;

    case ActivationFunc::SoftPlus:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_SOFTPLUS;
      pDmlDesc->desc.Desc = &pDmlDesc->softPlus;
      pDmlDesc->softPlus.InputTensor = nullptr;
      pDmlDesc->softPlus.OutputTensor = nullptr;
      pDmlDesc->softPlus.Steepness = info.param1;
      break;

    case ActivationFunc::SoftSign:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_SOFTSIGN;
      pDmlDesc->desc.Desc = &pDmlDesc->softSign;
      pDmlDesc->softSign.InputTensor = nullptr;
      pDmlDesc->softSign.OutputTensor = nullptr;
      break;

    case ActivationFunc::Tanh:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_TANH;
      pDmlDesc->desc.Desc = &pDmlDesc->tanh;
      pDmlDesc->tanh.InputTensor = nullptr;
      pDmlDesc->tanh.OutputTensor = nullptr;
      break;

    case ActivationFunc::TRelu:
      pDmlDesc->desc.Type = DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU;
      pDmlDesc->desc.Desc = &pDmlDesc->trelu;
      pDmlDesc->trelu.InputTensor = nullptr;
      pDmlDesc->trelu.OutputTensor = nullptr;
      pDmlDesc->trelu.Alpha = info.param1;
      break;

    default:
      throw std::runtime_error("Unknown activation function.");
      break;
  }
}

// =====================================================================================================================
// We use this to populate every binding in our binding tables.
void DmlOperator::BuildBufferBindDesc(
  const std::shared_ptr<Tensor>& pTensor, DML_BUFFER_BINDING* pBinding,
  DML_BINDING_DESC* pDesc
) {
  if (pTensor != nullptr) {
    pBinding->Buffer =
      pTensor->IsJitConst() ? m_jitResource : pTensor->Resource();
    pBinding->Offset =
      pTensor->ResourceOffset() + pTensor->GetLocalWindowOffset();

    uint64_t sizeInBytes = 0;
    if (pTensor->UseLocalWindowAttn()) {
      // For local window attention, only bind the window size
      sizeInBytes = pTensor->GetAttentionWindowSize();
    } else {
      sizeInBytes = pTensor->Desc().totalBytes;
    }

    pBinding->SizeInBytes = sizeInBytes;
    pDesc->Type = DML_BINDING_TYPE_BUFFER;
    pDesc->Desc = pBinding;
  } else {
    pDesc->Type = DML_BINDING_TYPE_NONE;
    pDesc->Desc = nullptr;
  }
}

// =====================================================================================================================
// This generic version assumes the operator creates its input tensors in the
// same order that DirectML expects.
void DmlOperator::BindInputs(
  CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
) {
  if (inputs.size() > 0) {
    std::vector<DML_BUFFER_BINDING> bindings(inputs.size());
    std::vector<DML_BINDING_DESC> descs(inputs.size());

    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      BuildBufferBindDesc(inputs[idx], &bindings[idx], &descs[idx]);
    }

    bindingTable->BindInputs(static_cast<UINT>(descs.size()), descs.data());
  }
}

// =====================================================================================================================
// This generic version assumes the operator creates its output tensors in the
// same order that DirectML expects.
void DmlOperator::BindOutputs(
  CComPtr<IDMLBindingTable> bindingTable, const TensorVector& outputs
) {
  if (outputs.size() > 0) {
    std::vector<DML_BUFFER_BINDING> bindings(outputs.size());
    std::vector<DML_BINDING_DESC> descs(outputs.size());

    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      BuildBufferBindDesc(outputs[idx], &bindings[idx], &descs[idx]);
    }

    bindingTable->BindOutputs(static_cast<UINT>(descs.size()), descs.data());
  }
}
// =====================================================================================================================
// Initialize the input/output tensors
void DmlOperator::InitializeTensors(
  Context* pContext, bool forceCreateInputD3DResc, bool forceCreateOutputD3DResc
) {
  try {
    m_inputAndConstTensorVector.clear();
    for (const std::shared_ptr<TensorDesc>& pDesc : m_inputTensorDescVec) {
      if (pDesc != nullptr && !pDesc->isInputFromPrevNode) {
        m_inputAndConstTensorVector.emplace_back(
          std::make_unique<Tensor>(
            *pDesc, pContext, !pDesc->isConst && forceCreateInputD3DResc
          )
        );
      } else {
        m_inputAndConstTensorVector.emplace_back(nullptr);
      }
    }

    m_outputTensorVector.clear();
    for (const std::shared_ptr<TensorDesc>& pDesc : m_outputTensorDescVec) {
      if (pDesc != nullptr) {
        m_outputTensorVector.emplace_back(
          std::make_unique<Tensor>(*pDesc, pContext, forceCreateOutputD3DResc)
        );
      } else {
        m_outputTensorVector.emplace_back(nullptr);
      }
    }
  } catch (const std::exception& exception) {
    std::cout << "Exception During Operator Initialiazing: " << exception.what()
              << std::endl;
  }
}

// =====================================================================================================================
void DmlOperator::CompileGraph(
  const Context& pContext,
  const std::vector<CComPtr<IDMLOperator>>& dmlOperators,
  const std::vector<DML_INPUT_GRAPH_EDGE_DESC>& inputEdges,
  const std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC>& intermediateEdges,
  const std::vector<DML_OUTPUT_GRAPH_EDGE_DESC>& outputEdges
) {
  std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );
  std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(
    static_cast<uint32_t>(dmlOperators.size())
  );

  DML_GRAPH_DESC graphDesc = {};
  std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(
    static_cast<uint32_t>(inputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(
    static_cast<uint32_t>(outputEdges.size())
  );
  std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(
    static_cast<uint32_t>(intermediateEdges.size())
  );

  // build the graph description from the above edges
  graphDesc.InputCount = static_cast<uint32_t>(inputEdges.size());
  graphDesc.OutputCount = static_cast<uint32_t>(outputEdges.size());
  graphDesc.NodeCount = static_cast<uint32_t>(dmlOperators.size());

  for (size_t i = 0; i < graphDesc.NodeCount; ++i) {
    dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{dmlOperators[i]};
    dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{
      DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]
    };
  }

  graphDesc.Nodes = dmlGraphNodes.data();

  // set the input edges
  graphDesc.InputEdgeCount = static_cast<uint32_t>(inputEdges.size());
  for (size_t i = 0; i < graphDesc.InputEdgeCount; ++i) {
    dmlInputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &inputEdges[i]};
  }
  graphDesc.InputEdges = dmlInputEdges.data();

  // set the output edges
  graphDesc.OutputEdgeCount = graphDesc.OutputCount;
  for (size_t i = 0; i < graphDesc.OutputEdgeCount; ++i) {
    dmlOutputEdges[i] =
      DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &outputEdges[i]};
  }
  graphDesc.OutputEdges = dmlOutputEdges.data();

  // set the intermediate edges
  graphDesc.IntermediateEdgeCount =
    static_cast<uint32_t>(intermediateEdges.size());
  for (size_t i = 0; i < graphDesc.IntermediateEdgeCount; ++i) {
    dmlIntermediateEdges[i] = DML_GRAPH_EDGE_DESC{
      DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediateEdges[i]
    };
  }
  graphDesc.IntermediateEdges = dmlIntermediateEdges.data();

  HRESULT status = S_OK;
  CComPtr<IDMLDevice1> dmlDevice1;
  status = pContext.DmlDevice()->QueryInterface(IID_PPV_ARGS(&dmlDevice1));
  if (FAILED(status)) {
    throw std::runtime_error("Failed to acquire dml graph interface.");
  }

  DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_NONE;
  executionFlags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;

  status = dmlDevice1->CompileGraph(
    &graphDesc, executionFlags, IID_PPV_ARGS(&m_operator)
  );
  if (FAILED(status)) {
    throw std::runtime_error("Failed to compile a DirectML operator.");
  }
}

}  // namespace ryzenai::onnx_utils
