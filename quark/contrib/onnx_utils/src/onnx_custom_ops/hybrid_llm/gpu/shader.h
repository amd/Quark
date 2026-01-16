// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// shader.h
#pragma once
#include <atlbase.h>
#include <d3d12.h>
#include <d3dcompiler.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "context.h"
// #include "opInterface.h"

namespace ryzenai::onnx_utils {

class BaseShader {
 protected:
  CComPtr<ID3D12RootSignature> rootSignature_ =
    nullptr;  // Root signature for the shader
  CComPtr<ID3D12PipelineState> pipelineState_ =
    nullptr;  // Pipeline state object

  CComPtr<ID3D12DescriptorHeap>
    descriptorHeap_;        // Descriptor heap for SRV and UAV
  UINT descriptorSize = 0;  // Size of a descriptor in the descriptor heap

  // DML_Ops::DMLOps* dml_instance_;           // Pointer to the DML instance
  std::shared_ptr<Context> m_ctx;  // Context object

  // Create descriptor heap (for SRV and UAV)
  void CreateDescriptorHeap(UINT numDescriptors) {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = numDescriptors;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    HRESULT hr = m_ctx->D3d12Device()->CreateDescriptorHeap(
      &heapDesc, IID_PPV_ARGS(&descriptorHeap_)
    );
    if (FAILED(hr)) {
      throw std::runtime_error("Failed to create descriptor heap");
    }

    descriptorSize = m_ctx->D3d12Device()->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
    );
  }

  // Compilation of shaders
  virtual CComPtr<ID3DBlob> CompileShaderFromFile(
    const std::wstring& shaderFile, const std::string& entryPoint,
    const std::string& target
  ) {
    // Compile the shader
    CComPtr<ID3DBlob> shaderBlob = nullptr;
    CComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3DCompileFromFile(
      shaderFile.c_str(), nullptr, nullptr, entryPoint.c_str(), target.c_str(),
      0, 0, &shaderBlob, &errorBlob
    );

    if (FAILED(hr)) {
      if (errorBlob) {
        std::cerr << (char*)errorBlob->GetBufferPointer() << std::endl;
      }
      throw std::runtime_error("Failed to compile shader");
    }
    return shaderBlob;
  }

  // Compilation of shaders
  virtual CComPtr<ID3DBlob> CompileShaderFromString(
    const std::string shaderSource, const std::string& entryPoint,
    const std::string& target
  ) {
    // Compile the shader
    CComPtr<ID3DBlob> shaderBlob = nullptr;
    CComPtr<ID3DBlob> errorBlob = nullptr;

    HRESULT hr = D3DCompile(
      shaderSource.c_str(), shaderSource.size(), nullptr, nullptr, nullptr,
      entryPoint.c_str(), target.c_str(), 0, 0, &shaderBlob, &errorBlob
    );

    if (FAILED(hr)) {
      if (errorBlob) {
        std::cerr << (char*)errorBlob->GetBufferPointer() << std::endl;
      }
      throw std::runtime_error("Failed to compile shader");
    }
    return shaderBlob;
  }

 public:
  BaseShader() {
    /*     dml_instance_ = &DML_Ops::DMLOps::getInstance();
         m_ctx = const_cast<Context*>(dml_instance_->getContext());*/
  }

  BaseShader(std::shared_ptr<Context> ctx) : m_ctx(ctx) {}
  virtual ~BaseShader() = default;

  // Virtual functions to be overridden by derived class
  virtual void CreateRootSignature() = 0;
  virtual void CreatePipelineState(const wchar_t* shaderFile) = 0;
  virtual void Dispatch(UINT numGroupsX, UINT numGroupsY, UINT numGroupsZ) = 0;
};

}  // namespace ryzenai::onnx_utils
