// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#ifndef BF16_TO_FP16_SHADER_H
#define BF16_TO_FP16_SHADER_H

#pragma once
#include "shader.h"

namespace ryzenai::onnx_utils {

class BF16ToFP16Shader : public BaseShader {
 private:
  CComPtr<ID3D12Resource> m_inputOutputBuffer;
  void* m_mappedInputOutputData = nullptr;
  // SRV and UAV descriptor heap for the input/output buffers
  CComPtr<ID3D12DescriptorHeap> m_heap;
  size_t m_elementCount = 0;  // keeps track of element count

  bool m_isCustomAllocatorUsed =
    false;  // Flag to indicate if custom allocator is used

  // Create the root signature for the compute shader
  void CreateRootSignature() override;

  // Create the pipeline state for the BF16 to FP16 shader
  void CreatePipelineState(const wchar_t* shaderFile) override;

  // Dispatch the compute shader
  void Dispatch(UINT numGroupsX, UINT numGroupsY, UINT numGroupsZ) override;

 public:
  // BF16ToFP16Shader() {}
  BF16ToFP16Shader(
    std::shared_ptr<Context> ctx, bool isCustomAllocatorUsed = false
  );
  ~BF16ToFP16Shader() {}

  // Create and upload the buffers for input/output
  void CreateBuffers(size_t elementCount);
  void FreeBuffers();

  // Public function to encapsulate the entire conversion process
  void ConvertBF16ToFP16(
    ID3D12Resource* resourcePtr, void* inputOutputBuffer, size_t elementCount
  );
};

}  // namespace ryzenai::onnx_utils

#endif  // BF16_TO_FP16_SHADER_H
