// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "jit_wts_loader.h"

#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>

#include "../external_data.pb.h"

namespace ryzenai::onnx_utils {

JitWtsLoader::JitWtsLoader(
  const std::unordered_map<std::string, std::string>& session_configs,
  std::unique_ptr<proto::Header> protoHeader
) {
  m_protoHeader = std::move(protoHeader);

  const auto& external_data_file = session_configs.at("external_data_file");

  // Max levels of GPU JIT (excluding 0). Each level will increase the JIT wts
  // for better memory but performance is reduced. Level 5 will put all GPU wts
  // in JIT
  static constexpr uint32_t kMaxLevels = 5;
  const auto& level_str = session_configs.at("hybrid_opt_gpu_jit");
  const auto level = level_str.empty() ? 0 : std::stoi(level_str);
  uint32_t gpuJitLevel = std::min<uint32_t>(std::max(level, 0), kMaxLevels);
  m_useDynamicJit = gpuJitLevel > 0;

  m_wtsFile = (std::filesystem::path(external_data_file).parent_path() /
               m_protoHeader->external_data().filename())
                .string();

  m_fileHandle = CreateFileA(
    m_wtsFile.c_str(), GENERIC_READ | GENERIC_WRITE,
    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
    FILE_FLAG_OVERLAPPED, NULL
  );
  if (m_fileHandle == INVALID_HANDLE_VALUE) {
    throw std::invalid_argument("Cannot open file " + m_wtsFile);
  }

  uint32_t layersInDynamicHeapPerJitLevel =
    (m_protoHeader->layers().size() / kMaxLevels);
  uint32_t layersInDynamicHeap = layersInDynamicHeapPerJitLevel * gpuJitLevel;
  if (layersInDynamicHeap > m_protoHeader->layers().size() ||
      (gpuJitLevel >= kMaxLevels)) {
    layersInDynamicHeap = m_protoHeader->layers().size();
  }
  uint32_t layersInStaticHeap =
    m_protoHeader->layers().size() - layersInDynamicHeap;

  if (layersInStaticHeap > 0) {
    HeapSizeDesc staticHeapSizeDesc;
    staticHeapSizeDesc.offset.QuadPart = m_protoHeader->layers().at(0).offset();
    for (uint32_t j = 0; j < layersInStaticHeap; j++) {
      uint32_t currentLayer = j;
      auto layersize = m_protoHeader->layers().at(currentLayer).size();

      for (uint32_t k = 0;
           k < m_protoHeader->layers().at(currentLayer).operators().size();
           k++) {
        staticHeapSizeDesc.operators.insert(
          m_protoHeader->layers().at(currentLayer).operators().at(k)
        );
      }
      staticHeapSizeDesc.sizeInBytes += layersize;
    }
    staticHeapSizeDesc.containedLayers = layersInStaticHeap;
    staticHeapSizeDesc.remainingSize = staticHeapSizeDesc.sizeInBytes;
    m_heapSizeDesc[STATIC_HEAP] = staticHeapSizeDesc;
    RequestLoadWeights(STATIC_HEAP);
  }

  if (m_useDynamicJit) {
    HeapSizeDesc dynamicHeapSizeDesc;
    dynamicHeapSizeDesc.offset.QuadPart =
      m_protoHeader->layers().at(layersInStaticHeap).offset();
    for (uint32_t j = layersInStaticHeap; j < m_protoHeader->layers().size();
         j++) {
      uint32_t currentLayer = j;
      auto layersize = m_protoHeader->layers().at(currentLayer).size();

      for (uint32_t k = 0;
           k < m_protoHeader->layers().at(currentLayer).operators().size();
           k++) {
        dynamicHeapSizeDesc.operators.insert(
          m_protoHeader->layers().at(currentLayer).operators().at(k)
        );
      }
      dynamicHeapSizeDesc.sizeInBytes += layersize;
    }
    dynamicHeapSizeDesc.containedLayers = layersInStaticHeap;
    dynamicHeapSizeDesc.remainingSize = dynamicHeapSizeDesc.sizeInBytes;
    dynamicHeapSizeDesc.isDynamicHeap = true;
    m_heapSizeDesc[DYNAMIC_HEAP] = dynamicHeapSizeDesc;
  }
}

uint32_t JitWtsLoader::GetHeapId(const std::string& node_name) {
  uint32_t heapId = 0;
  for (uint32_t i = 0; i < m_heapSizeDesc.size(); i++) {
    if (m_heapSizeDesc[i].operators.find(node_name) !=
        m_heapSizeDesc[i].operators.end()) {
      heapId = i;
      break;
    }
  }
  return heapId;
}

bool JitWtsLoader::InitializeJitTensorInfo(
  const std::string& node_name, int tensor_index, OnnxTensorInfo& tensorInfo,
  uint64_t tensorDataTypeSize
) {
  const auto& tensor =
    m_protoHeader->operators().at(node_name).data().at(tensor_index);
  const auto& opType = m_protoHeader->operators().at(node_name).op_type();

  tensorInfo.heapId = GetHeapId(node_name);

  uint64_t heapOffset = m_heapSizeDesc[tensorInfo.heapId].offset.QuadPart;
  uint64_t heapSize = m_heapSizeDesc[tensorInfo.heapId].sizeInBytes;
  int64_t& remainingSize = m_heapSizeDesc[tensorInfo.heapId].remainingSize;

  if (tensor.size() == 0) {
    return false;
  }

  tensorInfo.elementCount = tensor.size() / tensorDataTypeSize;
  // tensor offset relative to the data in heap
  tensorInfo.offsetInBytes = tensor.offset() - heapOffset;

  if (auto buffer = m_heapSizeDesc[tensorInfo.heapId].buffer)
    tensorInfo.UseRMMBuffer(std::move(buffer));
  else {
    tensorInfo.d3dResource = nullptr;
    tensorInfo.pCpuMappedD3DResc = nullptr;
  }

  tensorInfo.isDynamicTensor = m_heapSizeDesc[tensorInfo.heapId].isDynamicHeap;
  tensorInfo.isConstForJit = true;

  for (uint32_t i = 0; i < tensor.shape().size(); i++) {
    tensorInfo.shape.push_back(tensor.shape()[i]);
  }
  auto dataType = tensor.data_type();

  return true;
}

void JitWtsLoader::PrefillWts(uint32_t heapIdx, uint64_t resourceSize) {
  uint64_t totalChunks =
    (resourceSize + kParallelReadChunkSize - 1) / kParallelReadChunkSize;
  std::vector<ChunkRead> chunkReads;
  std::vector<HANDLE> eventHandles;

  chunkReads.clear();
  eventHandles.clear();

  eventHandles.resize(totalChunks);
  chunkReads.resize(totalChunks);
  for (uint64_t i = 0; i < totalChunks; ++i) {
    auto& cr = chunkReads[i];

    cr.hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (cr.hEvent == nullptr) {
      throw std::runtime_error("Failed to create event");
    }

    DWORD read = 0;
    uint64_t chunkOffset = i * kParallelReadChunkSize;
    uint64_t remainingSize = resourceSize - chunkOffset;
    uint64_t chunkSize = std::min(kParallelReadChunkSize, remainingSize);

    LARGE_INTEGER li = m_heapSizeDesc[heapIdx].offset;
    li.QuadPart += chunkOffset;

    ZeroMemory(&cr.overlap, sizeof(OVERLAPPED));
    cr.overlap.Offset = li.LowPart;
    cr.overlap.OffsetHigh = li.HighPart;
    cr.overlap.hEvent = cr.hEvent;
    bool ret = ReadFile(
      m_fileHandle,
      static_cast<byte*>(m_heapSizeDesc[heapIdx].buffer.Data()) + chunkOffset,
      chunkSize, &read, &cr.overlap
    );

    if (!ret && GetLastError() != ERROR_IO_PENDING) {
      CloseHandle(cr.hEvent);
      throw std::runtime_error("Failed to read file");
    }

    eventHandles[i] = cr.hEvent;
  }

  WaitForMultipleObjects(
    eventHandles.size(), eventHandles.data(), TRUE, INFINITE
  );

  for (auto handle : eventHandles) {
    CloseHandle(handle);
  }
}

JitWtsLoader::~JitWtsLoader() { CloseHandle(m_fileHandle); }

void JitWtsLoader::CreateResourceForJit(
  uint64_t resourceSize, uint32_t heapIdx
) {
  auto buffer = RyzenMM::GPUAllocator<'JIT0'>().AllocateBuffer(resourceSize);

  std::lock_guard<std::mutex> lock(stateMutex);
  m_heapSizeDesc[heapIdx].buffer = std::move(buffer);
}

void JitWtsLoader::DynamicLoadWeights() {
  if (!m_useDynamicJit) {
    return;
  }
  RequestLoadWeights(DYNAMIC_HEAP);
}

void JitWtsLoader::RequestLoadWeights(uint32_t heapIdx, bool wait) {
  uint64_t resourceSize;

  {
    std::lock_guard<std::mutex> lock(stateMutex);
    // set a flag to check if the heap is released
    if (m_heapSizeDesc[heapIdx].buffer) {
      // return;
      throw std::runtime_error("Heap is not released");
    }
    resourceSize = m_heapSizeDesc[heapIdx].sizeInBytes;
  }

  CreateResourceForJit(resourceSize, heapIdx);
  m_jit_futures[heapIdx] = std::async(
    std::launch::async, &JitWtsLoader::PrefillWts, this, heapIdx, resourceSize
  );

  if (wait) {
    m_jit_futures[heapIdx].wait();
  }
}

bool JitWtsLoader::IsWeightsReady(uint32_t heapIdx) {
  if (m_jit_futures[heapIdx].valid()) {
    m_jit_futures[heapIdx].wait();
    return true;
  }
  return false;
}

void JitWtsLoader::ReleaseWeights() {
  if (!m_useDynamicJit) {
    return;
  }

  std::lock_guard<std::mutex> lock(stateMutex);
  m_heapSizeDesc[DYNAMIC_HEAP].buffer = {};
}

}  // namespace ryzenai::onnx_utils
