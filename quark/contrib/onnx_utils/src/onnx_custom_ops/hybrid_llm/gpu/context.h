// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <atlbase.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <vector>

#include "DirectML.h"
#include "execution_provider.hpp"
#include "tensor.h"

namespace ryzenai::onnx_utils {

class DmlOperator;

// The number of staging resources used to upload and download memory and the
// max size of those resources.
constexpr int32 NumCopyState = 2;
constexpr UINT64 MaxCopyResSize = 64 * 1024 * 1024;

// Everything needed to create a Context.
struct ContextInfo {
  int32 adapter;   // The index of the first adapter to consider.
  bool debug;      // If the D3D12 debug layer should be enabled.
  bool systemMem;  // If all tensors only use persistently mapped system memory
                   // for CPU or NPU to use (no copies!).
  bool noPaging;   // Make the tensor always resident.
  bool useExternalMemory;  // If we get tensor data externally. This and
                           // systemMem should be opposite
};

// We use multiple copies of this state to build and execute both steps of a
// tensor copy in parallel.
struct TensorCopyState {
  CComPtr<ID3D12GraphicsCommandList> cmdList;  // Copies one chunk of a tensor.
  CComPtr<ID3D12Fence> fence;                  // Waits for cmdList to complete.
  CComPtr<ID3D12Resource> resource;  // A staging resource for GPU copies.
  void* pBuffer;   // If non-null, a mapped pointer to the resource.
  bool needsWait;  // If this could could still be active on the GPU.

  // This state is only needed by downloads.
  void* pMemcpyDst;   // Where that memcpy should write to (tensor CPU memory).
  size_t memcpySize;  // How many bytes to memcpy.
};

// =====================================================================================================================
// This class wraps all DirectML and D3D12 state necessary to execute and
// profile operators.
class Context : public ExecutionProviderExtensions {
 public:
  explicit Context(
    const ContextInfo& info, CComPtr<ID3D12Device3> d3d12Device = nullptr
  );
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context();

  // Creates a DML operator object on our DML device.
  CComPtr<IDMLCompiledOperator> CompileOperator(
    const DML_OPERATOR_DESC* pDesc, bool disableMetacmds, bool halfPrecision
  ) const;

  // Enqueues an async make resident call. The context will wait for all paging
  // requests in BindOperator.
  void MakeResident(CComPtr<ID3D12Heap> heap);

  // Uploads tensors to the GPU. Must be called on inputs and outputs between
  // BindOperator and Evaluate.
  void UploadInputOrConstData(
    const TensorVector& tensors, bool useCustomAllocator = false
  );

  void UploadConstData(
    const TensorVector& tensors, bool useCustomAllocator = false
  );
  void UploadInputData(
    const TensorVector& tensors, bool useCustomAllocator = false
  );

  // Downloads tensors from the GPU. Must be called on outputs after Evaluate if
  // validation is on.
  void DownloadData(
    const TensorVector& tensors, bool useCustomAllocator = false
  );

  // Executes the bound operator. Outputs will be written to the bound output
  // tensors.
  void ExecuteOperator(
    const std::vector<std::shared_ptr<DmlOperator>>& pTargetOp
  );

  const std::vector<float>& GpuTimes() const { return m_gpuTimes; }

  const ContextInfo& GetInfo() const { return m_info; }

  hstring AdapterName() const { return m_adapterName; }

  CComPtr<ID3D12Device3> D3d12Device() const { return m_d3d12Device; }
  CComPtr<IDMLDevice> DmlDevice() const { return m_dmlDevice; }

  CComPtr<ID3D12Fence> GetPagingFence() const {
    return m_pagingFence;
  }  // Used to track async GPU paging requests.
  UINT64 GetPagingValue() const { return m_pagingValue; }

  CComPtr<ID3D12CommandAllocator> GetExecCmdAlloc() {
    return m_execCmdAlloc;
  }  // Used by the execution command list.
  CComPtr<ID3D12GraphicsCommandList> GetExecCmdList() { return m_execCmdList; }
  CComPtr<IDMLCommandRecorder> GetDmlRecorder() { return m_dmlRecorder; }
  CComPtr<ID3D12Fence> GetEvalFence() {
    return m_evalFence;
  }  // Signaled when all execution is done
  CComPtr<ID3D12CommandQueue> GetEvalQueue() {
    return m_evalQueue;
  }  // We use this queue to execute operators.

  void SignalBinaryFence(
    CComPtr<ID3D12CommandQueue> queue, CComPtr<ID3D12Fence> fence
  );
  void WaitForFence(CComPtr<ID3D12Fence> fence, UINT64 value);

  const ContextInfo m_info;  // The context's create-time description.
  UINT64
  m_copyResourceSize;  // The size of the copy staging resources in bytes.
  TensorCopyState m_copyState[NumCopyState];

  // Gpu Jit
  void SetIsGpuJitEnabled(bool flag) { m_isGpuJit = flag; }
  const bool IsGpuJitEnabled() const { return m_isGpuJit; }

 private:
  void CreateDxObjects();
  void CreateDxObjects(CComPtr<ID3D12Device3> d3d12Device);

  // General state
  hstring m_adapterName;            // Our DXGI adapter's name.
  CComPtr<IDMLDevice> m_dmlDevice;  // Creates all DirectML state.
  CComPtr<IDMLCommandRecorder>
    m_dmlRecorder;  // Records DirectML commands into command lists.

  // General GPU state
  CComPtr<IDXGIFactory4>
    m_dxgiFactory;  // The DXGI factory that created our adapter.
  CComPtr<IDXGIAdapter4>
    m_dxgiAdapter;  // The DXGI adapter that created our D3D12 device.
  CComPtr<ID3D12Device3>
    m_d3d12Device;  // The device that owns all of our D3D12 objects.
  CComPtr<ID3D12CommandQueue>
    m_presentQueue;  // Used by swap chains if our eval queue isn't direct.
  CComPtr<ID3D12Fence>
    m_pagingFence;       // Used to track async GPU paging requests.
  UINT64 m_pagingValue;  // The most recent paging fence value.
  HANDLE m_doneEvent;    // Used to block the CPU until a fence is signaled.

  // Stated required to upload or download tensors to the GPU.
  CComPtr<ID3D12CommandQueue>
    m_copyQueue;  // We use this queue to upload and download tensors.
  CComPtr<ID3D12CommandAllocator>
    m_copyCmdAlloc;  // Used by the upload/download command lists.

  // State required to evalute operators. Some of this must be
  // created/initialized during BindOperator.
  const D3D12_COMMAND_LIST_TYPE
    m_cmdListType;  // Which command queue/list type we use for execution.
  // CComPtr<IDMLCompiledOperator> m_operator;  // The bound DirectML operator.
  CComPtr<ID3D12CommandQueue>
    m_evalQueue;  // We use this queue to execute operators.
  CComPtr<ID3D12CommandAllocator>
    m_execCmdAlloc;  // Used by the execution command list.
  CComPtr<ID3D12GraphicsCommandList>
    m_execCmdList;  // Records commands to time and evaluate operators.
  CComPtr<ID3D12Fence> m_evalFence;  // Signaled when all execution is done.

  // State required to get GPU times.
  std::vector<float> m_gpuTimes;  // GPU times in execution order, in seconds.
  CComPtr<ID3D12QueryHeap>
    m_timerQueryHeap;  // Used to measure the GPU execution time.
  CComPtr<ID3D12Resource>
    m_timerBuffer;        // Contains the GPU times for CPU read-back.
  UINT64 m_gpuTimerFreq;  // The GPU timer frequency.

  // GPU Jit
  bool m_isGpuJit = false;
};

}  // namespace ryzenai::onnx_utils
