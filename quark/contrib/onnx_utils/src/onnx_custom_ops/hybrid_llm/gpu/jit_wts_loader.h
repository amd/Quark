// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Don't define min/max macros in the Windows headers.
#define NOMINMAX

#include <d3d12.h>
#include <ryzenai/ryzen_mm.h>
#include <stdint.h>
#include <windows.h>

#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "gpu_utils.h"

namespace ryzenai::onnx_utils {

namespace proto {
class Header;
}

struct HeapSizeDesc {
  uint64_t sizeInBytes = 0;
  LARGE_INTEGER offset = {0};
  uint32_t containedLayers = 0;
  RyzenMM::BufferRef buffer;
  std::unordered_set<std::string> operators;
  int64_t remainingSize = 0;
  bool isDynamicHeap = false;

  auto GetD3DResource() const {
    CComPtr<ID3D12Resource> res;
    res.Attach(RyzenMM::Platform::DX::GetUnderlyingD3D12Resource(buffer));
    return res;
  }
};

struct ChunkRead {
  OVERLAPPED overlap = {};
  HANDLE hEvent = nullptr;
};

class JitWtsLoader {
 public:
  enum HeapType { STATIC_HEAP = 0, DYNAMIC_HEAP = 1, HEAP_TYPE_COUNT = 2 };

  // static constexpr uint32_t kMaxLayersInDynamicHeap = 5;
  static constexpr uint64_t kParallelReadChunkSize =
    256ULL * 1024ULL * 1024ULL;  // 256 MB

  JitWtsLoader(
    const std::unordered_map<std::string, std::string>& session_configs,
    std::unique_ptr<proto::Header> protoHeader
  );
  ~JitWtsLoader();

  void DynamicLoadWeights();
  void RequestLoadWeights(uint32_t, bool wait = false);
  bool IsWeightsReady(uint32_t heapIdx);

  void ReleaseWeights();

  // PRoto
  bool InitializeJitTensorInfo(
    const std::string& node_name, int tensor_index, OnnxTensorInfo& tensorInfo,
    uint64_t tensorDataTypeSize
  );

  void PrefillWts(uint32_t heapIdx, uint64_t resourceSize);

  bool IsJitWtsForGpuEnabled() const { return m_isJitWtsForGpuEnabled; }

  bool IsDyamicHeap(uint32_t heapIdx) {
    return m_heapSizeDesc[heapIdx].isDynamicHeap;
  }

  void CreateResourceForJit(uint64_t resourceSize, uint32_t heapIdx);

  HeapSizeDesc& GetHeapSizeDesc(uint32_t idx) { return m_heapSizeDesc[idx]; }

  uint32_t GetHeapId(const std::string& node_name);

  void SetFirstAndLastNode(const std::string& node) {
    if (m_firstNode.empty()) {
      m_firstNode = node;
    } else {
      m_lastNode = node;
    }
  }

  std::string GetFirstNode() { return m_firstNode; }

  std::string GetLastNode() { return m_lastNode; }

 private:
  // Proto header
  std::unique_ptr<proto::Header> m_protoHeader;
  std::string m_wtsFile;

  bool m_isJitWtsForGpuEnabled = false;
  bool m_useDynamicJit = false;

  std::array<HeapSizeDesc, HEAP_TYPE_COUNT> m_heapSizeDesc;

  std::mutex stateMutex;

  // File management
  HANDLE m_fileHandle = nullptr;

  // FUTURES
  std::future<void> m_jit_futures[HEAP_TYPE_COUNT];

  std::string m_firstNode;
  std::string m_lastNode;
};

}  // namespace ryzenai::onnx_utils
