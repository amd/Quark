// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications: Copyright (c) 2024 Advanced Micro Devices, Inc.

#include "ryzenai/onnx_utils/custom_allocator.hpp"

#include <assert.h>
#include <ryzenai/ryzen_mm.h>

#include "onnxruntime_cxx_api.h"

namespace ryzenai::onnx_utils {

// NOTE: these tags could be anything and will be presented in RyzenMM memory
// usage reports once implemented. So, it's better to keep them meaningful and
// unique.
constexpr auto MemoryTag1 = 'COP1';
constexpr auto MemoryTag2 = 'COP2';

struct CustomAllocator::Data {
  OrtMemoryInfo* memoryInfo = nullptr;
};

CustomAllocator::CustomAllocator(bool enable)
  : data_{std::make_unique<Data>()} {
  assert(enable);

  Ort::ThrowOnError(
    Ort::GetApi().CreateCpuMemoryInfo(
      OrtDeviceAllocator, OrtMemTypeDefault, &data_->memoryInfo
    )
  );

  OrtAllocator::version = ORT_API_VERSION;

  OrtAllocator::Alloc = [](OrtAllocator*, size_t size) -> void* {
    return RyzenMM::GPUAllocator<MemoryTag1>()
      .AllocateBuffer(size)
      .CreateUnmanagedBuffer();
  };

  OrtAllocator::Free = [](OrtAllocator*, void* p) {
    RyzenMM::FreeUnmanagedBuffer(p);
  };

  OrtAllocator::Info =
    [](const OrtAllocator* allocator) -> const OrtMemoryInfo* {
    return static_cast<const CustomAllocator*>(allocator)->data_->memoryInfo;
  };

  OrtAllocator::Reserve = [](OrtAllocator*, size_t size) -> void* {
    return RyzenMM::GPUAllocator<MemoryTag2>()
      .AllocateBuffer(size)
      .CreateUnmanagedBuffer();
  };
}

CustomAllocator::~CustomAllocator() noexcept {
  Ort::GetApi().ReleaseMemoryInfo(data_->memoryInfo);
  RyzenMM::Cleanup();
}

}  // namespace ryzenai::onnx_utils
