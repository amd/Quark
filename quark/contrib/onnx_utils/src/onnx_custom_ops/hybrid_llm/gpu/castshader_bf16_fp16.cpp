// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "castshader_bf16_fp16.h"

#include <sstream>

namespace ryzenai::onnx_utils {

std::string bf16ToFp16Shader = R"(
    // Define a Structured Buffer for input bfloat16 data and Read Write StructuredBuffer for output float16 data
    RWStructuredBuffer<uint> inputOutputBuffer : register(u0); // Input / Output float16 data

    uint ConvertBf16ToFp16(uint bfloat16_value)
    {
	    // Extract the sign, exponent, and mantissa from the bfloat16 (IEEE 754 format)
        uint sign     = (bfloat16_value >> 15) & 0x1;  // 1-bit sign
        uint exponent = (bfloat16_value >> 7) & 0xFF;  // 8-bit exponent
        uint mantissa = (bfloat16_value >> 0) & 0x7F;  // 7-bit mantissa
        // Convert bfloat16 exponent (8 bits) to float16 exponent (5 bits)
        // Adjust bias from bfloat16 (127) to float16 (15)
        int new_exponent = exponent - 127 + 15;
        // Handle exponent underflow, overflow, and normal cases
        if (new_exponent <= 0)
        {
            // Exponent underflow, denormalize result
            new_exponent = 0;
            mantissa = 0;
        }
        else if (new_exponent >= 31)
        {
            // Exponent overflow, set to infinity
            new_exponent = 31;
            mantissa = 0;
        }
        // Construct the float16 result (1 sign bit, 5 exponent bits, 10 mantissa bits)
        uint float16_value = ((sign << 15) | (new_exponent << 10) | (mantissa << 3));  // Shift mantissa to 10 bits
        return float16_value;
    }

    // Thread group dimensions
    [numthreads(256, 1, 1)]
    void BFloat16ToFloat16CS(uint3 id : SV_DispatchThreadID)
    {
        // Calculate the index for the input based on threadGroupX and threadGroupY
        //65535 * 256 is maximum number of threads we can handle in X dimension
        uint index = id.x + (id.y * 65535 * 256);

        // Fetch the bfloat16 value from the input buffer using the current thread ID
        uint packed_bf16 = inputOutputBuffer[index];
	    uint bfloat16_low = packed_bf16 & 0xFFFF;  // Extract 1st bf16
        uint bfloat16_high = packed_bf16 >> 16; // extract 2nd bf16
        uint float16_low = ConvertBf16ToFp16(bfloat16_low);
        uint float16_high = ConvertBf16ToFp16(bfloat16_high);

	    uint packed_fp16 = (float16_high << 16) | float16_low;
        // Write the result to the output buffer
        inputOutputBuffer[index] = packed_fp16;
    }
)";

BF16ToFP16Shader::BF16ToFP16Shader(
  std::shared_ptr<Context> ctx, bool isCustomAllocatorUsed
)
  : BaseShader(ctx) {
  CreateRootSignature();
  CreatePipelineState(nullptr /*L"cast_bf16_to_fp16.hlsl"*/);

  // Create descriptor heap for the SRV and UAV
  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.NumDescriptors = 1;  // SRV and UAV
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  HRESULT hr = m_ctx->D3d12Device()->CreateDescriptorHeap(
    &heapDesc, IID_PPV_ARGS(&m_heap)
  );
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create SRV UAV heap");
  }

  m_isCustomAllocatorUsed = isCustomAllocatorUsed;
}

// Create the root signature for the compute shader
void BF16ToFP16Shader::CreateRootSignature() {
  D3D12_DESCRIPTOR_RANGE tensorRanges = {};

  tensorRanges.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  tensorRanges.NumDescriptors = 1;
  tensorRanges.BaseShaderRegister = 0;

  D3D12_ROOT_PARAMETER rootParameters = {};

  rootParameters.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rootParameters.DescriptorTable.NumDescriptorRanges = 1;
  rootParameters.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  rootParameters.DescriptorTable.pDescriptorRanges = &tensorRanges;

  D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
  rootSignatureDesc.NumParameters = 1;
  rootSignatureDesc.pParameters = &rootParameters;
  rootSignatureDesc.Flags =
    (D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS |
     D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
     D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
     D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
     D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS);

  CComPtr<ID3DBlob> serializedRootSignature;
  CComPtr<ID3DBlob> error;
  if (FAILED(D3D12SerializeRootSignature(
        &rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSignature, &error
      ))) {
    std::stringstream ss;
    ss << "Failed to serialize our root signature: "
       << static_cast<const char*>(error->GetBufferPointer());
    throw std::runtime_error(ss.str());
  }

  if (FAILED(m_ctx->D3d12Device()->CreateRootSignature(
        0, serializedRootSignature->GetBufferPointer(),
        serializedRootSignature->GetBufferSize(), IID_PPV_ARGS(&rootSignature_)
      ))) {
    throw std::runtime_error("Failed to create our root signature.");
  }
}

// Create the pipeline state for the BF16 to FP16 shader
void BF16ToFP16Shader::CreatePipelineState(const wchar_t* shaderFile) {
  CComPtr<ID3DBlob> computeShader;
  if (!shaderFile) {
    computeShader = CompileShaderFromString(
      bf16ToFp16Shader, "BFloat16ToFloat16CS", "cs_5_0"
    );
  } else {
    computeShader =
      CompileShaderFromFile(shaderFile, "BFloat16ToFloat16CS", "cs_5_0");
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
  psoDesc.pRootSignature = rootSignature_.p;
  psoDesc.CS = {
    computeShader->GetBufferPointer(), computeShader->GetBufferSize()
  };

  HRESULT hr = m_ctx->D3d12Device()->CreateComputePipelineState(
    &psoDesc, IID_PPV_ARGS(&pipelineState_)
  );
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create pipeline state");
  }
}

// Dispatch the compute shader
void BF16ToFP16Shader::Dispatch(
  UINT numGroupsX, UINT numGroupsY, UINT numGroupsZ
) {
  m_ctx->GetExecCmdAlloc()->Reset();
  HRESULT hr =
    m_ctx->GetExecCmdList()->Reset(m_ctx->GetExecCmdAlloc(), nullptr);
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to reset the execution command list.");
  }

  m_ctx->GetExecCmdList()->SetPipelineState(pipelineState_.p);
  m_ctx->GetExecCmdList()->SetComputeRootSignature(rootSignature_.p);

  ID3D12DescriptorHeap* descriptorHeaps[] = {m_heap.p};
  m_ctx->GetExecCmdList()->SetDescriptorHeaps(
    _countof(descriptorHeaps), descriptorHeaps
  );

  D3D12_GPU_DESCRIPTOR_HANDLE uavHandle =
    m_heap->GetGPUDescriptorHandleForHeapStart();
  m_ctx->GetExecCmdList()->SetComputeRootDescriptorTable(
    0, uavHandle
  );  // input / Output UAV

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

  m_ctx->GetExecCmdList()->Dispatch(numGroupsX, numGroupsY, numGroupsZ);

  m_ctx->GetExecCmdList()->ResourceBarrier(1, &barrier);
  if (FAILED(m_ctx->GetExecCmdList()->Close())) {
    throw std::runtime_error("Failed to record the execution command list.");
  }

  ID3D12CommandList* pCmdLists[] = {m_ctx->GetExecCmdList()};
  m_ctx->GetEvalQueue()->ExecuteCommandLists(
    sizeof(pCmdLists) / sizeof(pCmdLists[0]), pCmdLists
  );

  m_ctx->SignalBinaryFence(m_ctx->GetEvalQueue(), m_ctx->GetEvalFence());
  m_ctx->WaitForFence(m_ctx->GetEvalFence(), 1);
}

void BF16ToFP16Shader::FreeBuffers() {
  if (m_elementCount) {
    m_inputOutputBuffer.Release();
    m_elementCount = 0;
  }
}

// Create and upload the buffers for input/output
void BF16ToFP16Shader::CreateBuffers(size_t elementCount) {
  if (m_isCustomAllocatorUsed) {
    return;
  }

  if (m_elementCount >= elementCount) {
    return;
  }

  m_elementCount = elementCount;
  UINT64 bufferSize = elementCount * sizeof(uint16_t);

  // Input / output buffer (bfloat16 data)
  D3D12_HEAP_PROPERTIES heapProps = {};

  heapProps.Type = D3D12_HEAP_TYPE_CUSTOM;
  heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
  heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Width = bufferSize;
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.SampleDesc.Count = 1;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  // Create output buffer (for FP16 result)
  // heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
  heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  CComPtr<ID3D12Resource> pNewOutputputResource;
  m_ctx->D3d12Device()->CreateCommittedResource(
    &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
    IID_PPV_ARGS(&pNewOutputputResource)
  );

  m_inputOutputBuffer = std::move(pNewOutputputResource);

  // Map the CPU handle for input / output d3dresource
  m_inputOutputBuffer->Map(
    0, nullptr, reinterpret_cast<void**>(&m_mappedInputOutputData)
  );

  // Create UAV for the input / output buffer
  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uavDesc.Format = DXGI_FORMAT_R32_UINT;
  uavDesc.Buffer.FirstElement = 0;
  // 2 bf16 values packed into 1 uint32
  uavDesc.Buffer.NumElements = static_cast<uint32_t>(elementCount / 2);
  uavDesc.Buffer.StructureByteStride = 0U;
  uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

  // Create the actual UAV
  D3D12_CPU_DESCRIPTOR_HANDLE uavHandle =
    m_heap->GetCPUDescriptorHandleForHeapStart();
  m_ctx->D3d12Device()->CreateUnorderedAccessView(
    m_inputOutputBuffer.p, nullptr, &uavDesc, uavHandle
  );
}

// Function to encapsulate the entire conversion process
// inputBF16Data is Id3d12Resource* in case of custom allocator
void BF16ToFP16Shader::ConvertBF16ToFP16(
  ID3D12Resource* resourcePtr, void* inputOutputBuffer, size_t elementCount
) {
  // max value 65535 (uint16_t) as set by d3d12 per group dimension
  // AMD hardware can go up to uint32_t max value, but to keep it compatible
  // across hardware, we will keep it uint16_t
  uint32_t numGroupsX = 1;
  uint32_t numGroupsY = 1;
  uint32_t numGroupsZ = 1;

  // Max theoretical threads per dispatch = 65535*65535*65535*256 where 256 is
  // threads per group Since this number is way too large, we will only use
  // numGroupsX and numGroupsY and keep numGroupsZ as 1. This will handle the
  // largest output by the model

  const uint32_t THREADS_PER_GROUP = 256;  // generally 256 is used
  // Maximum threads per dimension
  const uint32_t MAX_THREADS_PER_DIMENSION = 65535;

  // elementcount/2 since the data is packed (2 bf16 values in 1 uint32)
  uint64_t totalGroups =
    ((elementCount / 2) + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

  if (totalGroups <= MAX_THREADS_PER_DIMENSION) {
    numGroupsX = static_cast<uint32_t>(totalGroups);
  } else {
    numGroupsX = MAX_THREADS_PER_DIMENSION;
    numGroupsY = static_cast<uint32_t>(
      (totalGroups + MAX_THREADS_PER_DIMENSION - 1) / MAX_THREADS_PER_DIMENSION
    );
  }

  // Copy the input data to the input buffer

  if (resourcePtr) {
    // Create UAV for the output buffer
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_R32_UINT;
    uavDesc.Buffer.FirstElement = 0;
    // 2 bf16 values packed into 1 uint32
    uavDesc.Buffer.NumElements = static_cast<uint32_t>(elementCount / 2);
    uavDesc.Buffer.StructureByteStride = 0;
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    // Create the UAV
    D3D12_CPU_DESCRIPTOR_HANDLE uavHandle =
      m_heap->GetCPUDescriptorHandleForHeapStart();
    m_ctx->D3d12Device()->CreateUnorderedAccessView(
      resourcePtr, nullptr, &uavDesc, uavHandle
    );

    Dispatch(numGroupsX, numGroupsY, numGroupsZ);
  } else {
    memcpy(
      m_mappedInputOutputData, inputOutputBuffer,
      elementCount * sizeof(uint16_t)
    );

    // Dispatch the compute shader
    Dispatch(numGroupsX, numGroupsY, numGroupsZ);

    memcpy(
      inputOutputBuffer, m_mappedInputOutputData,
      elementCount * sizeof(uint16_t)
    );
  }
}

}  // namespace ryzenai::onnx_utils
