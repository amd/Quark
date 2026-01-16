// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <sstream>

#include "castshader_bf16_fp16.h"
#include "context.h"
#include "gpu_utils.h"
#include "gpuprofile.h"
#include "jit_wts_loader.h"
#include "operator.h"
#include "operators/SkipSimplifiedLayerNorm.h"
#include "operators/gather.h"
#include "operators/gemm.h"
#include "operators/gqo.h"
#include "operators/matMulNBits.h"
#include "operators/norm.h"
#include "operators/reduce.h"
#include "operators/rotaryEmbedding.h"
#include "operators/ssmlp.h"
#include "operators/ssmlp_gelu.h"
#include "operators/subtract.h"
#include "tensor.h"

namespace ryzenai::onnx_utils {

// Prints the logs
template <typename... Args>
void printMessage(Args... args) {
#ifdef ENABLE_PRINT
  std::ostringstream oss;
  (oss << ... << args);  // Fold expression (C++17) to handle multiple arguments
  std::cout << oss.str() << std::endl;
#endif
}

namespace DML_Ops {

/* This class is the interface class to be called from the Custom OP. It will
 * provide calling functions for all the supported operators*/
class DMLOps {
 public:
  ~DMLOps();
  void ComputeMatMulGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&
  );

  void ComputeMatMulNBitsGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&, bool rebindIO
  );

  void ComputeRotaryEmbeddingGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&
  );

  void ComputeSlrn(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&, bool rebindIO
  );

  void ComputeSkipSimplifiedLayerNormGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&
  );

  void ComputeSSMLPGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
    std::vector<OnnxTensorInfo>& pBufOutput, bool rebindIO
  );

  void ComputeGatherGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
    std::vector<OnnxTensorInfo>& pBufOutput
  );

  void Initialize(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void DMLOps::ReinitializeAndBindIO(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void InitializeMatMulNBits(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void InitializeRotaryEmbedding(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void InitializeSlrn(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void InitializeSkipSimplifiedLayerNorm(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void InitializeSSMLP(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void CreateMatMulOperator(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void CreateMatMulNBitsOperator(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&, int64_t block_size = 32
  );

  void CreateRotaryEmbeddingOperator(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&, bool interleaved = false
  );

  void CreateSlrnOp(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&, const AttributeForSlrn& attr
  );

  void CreateSkipSimplifiedLayerNormOp(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&, const float& epsilon
  );

  void CreateGatherOperator(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void CreateSSMLPOp(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&, const AttributeForSSMLP& attr
  );

  void CreateReduceOp(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const std::vector<OnnxTensorInfo>&
  );

  void InitializeReduce(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void ComputeReduceGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&
  );

  void CreateSubOp(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void InitializeSub(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void ComputeSubGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
    std::vector<OnnxTensorInfo>& pBufOutput
  );

  void CreateGQOOp(
    const std::string& opName, const std::vector<OnnxTensorInfo>&,
    const AttributeForGQO& attr
  );

  void GQOOpDynamicInitialization(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput,
    const bool isOutputShapeChanged = false
  );

  void InitializeGQO(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& pTensorDescInput,
    const std::vector<OnnxTensorInfo>& pTensorDescOutput
  );

  void ComputeGQOGPU(
    const std::string& opName, std::vector<OnnxTensorInfo>&,
    std::vector<OnnxTensorInfo>&, bool rebindIO
  );

  void UpdateResourceHandle(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& onnxTensorInput,
    const std::vector<OnnxTensorInfo>& onnxTensorOutput
  );

  void SetConstants(
    std::shared_ptr<DmlOperator>& pOperator,
    const std::vector<OnnxTensorInfo>& onnxTensorInput,
    const std::vector<OnnxTensorInfo>& onnxTensorOutput
  );

  void SetTensorResource(
    std::shared_ptr<Tensor>& tensor, const OnnxTensorInfo& onnxTensorInfo,
    uint64_t offset
  );

  void SetTensorConstFlags(
    std::shared_ptr<Tensor>& tensor, const OnnxTensorInfo& onnxTensorInfo
  );

  /*void GetMappedInputOutputTensors(std::string nodeName,
                                   std::vector<void*>& mappedInputTensors,
                                   std::vector<void*>& mappedOutputTensors);*/
  static std::shared_ptr<DMLOps> getInstance(
    const std::unordered_map<std::string, std::string>& session_config,
    std::optional<CComPtr<ID3D12Device3>> d3d12Device = std::nullopt
  );

  const Context* getContext() const;
  BF16ToFP16Shader* getCastShader() { return bf16_ptr.get(); }
  const DmlOperator* getDMLOperator(const std::string& opName) const {
    return m_OpLists.at(opName).get();
  }

  const int64 getResourceSizePerInput(
    const std::string& opName, const int64 index
  ) const {
    return getDMLOperator(opName)
      ->GetInputTensorDescVector()
      .at(index)
      ->totalBytes;
  }
  const int64 getResourceSizePerOutput(
    const std::string& opName, const int64 index
  ) const {
    return getDMLOperator(opName)
      ->GetOutputTensorDescVector()
      .at(index)
      ->totalBytes;
  }

  void CheckForOpChaining(
    const std::string& opInputName, const std::string& opOutputName,
    const std::string& opName
  );

  inline void RecordOrExecuteOperator(
    const std::shared_ptr<DmlOperator>& pOperator, const std::string& opName
  );

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  const std::map<int, Duration>& GetPerfData(const std::string& opName) const;
#endif

  DMLOps(const DMLOps&) = delete;
  DMLOps& operator=(const DMLOps&) = delete;

  /*static auto& instance() {
 static DMLOps dmpOp_;
 return dmpOp_;
 }*/
  const bool& isGPURescMappedForCPU() const;

  const void SetSessionConfigs(
    const std::unordered_map<std::string, std::string>& session_configs
  ) {
    m_session_configs = session_configs;
  }

  const std::unordered_map<std::string, std::string> GetSessionConfigs() {
    return m_session_configs;
  }

  bool IsJitWtsLoaderEnabled() {
    if (m_JitWtsLoader) {
      return true;
    }
    return false;
  }

  void SetFirstAndLastNodeForJit(const std::string& node) {
    if (m_JitWtsLoader) {
      m_JitWtsLoader->SetFirstAndLastNode(node);
    }
  }

  bool InitializeJitTensorInfo(
    const std::string& node_name, int tensor_index, OnnxTensorInfo& tensorInfo,
    uint64_t tensorDataTypeSize
  ) {
    if (m_JitWtsLoader) {
      return m_JitWtsLoader->InitializeJitTensorInfo(
        node_name, tensor_index, tensorInfo, tensorDataTypeSize
      );
    }

    return false;
  }

  std::string GetFirstNodeForJit() {
    if (m_JitWtsLoader) {
      return m_JitWtsLoader->GetFirstNode();
    }
    return std::string();
  }

  std::string GetLastNodeForJit() {
    if (m_JitWtsLoader) {
      return m_JitWtsLoader->GetLastNode();
    }
    return std::string();
  }

  void ReleaseWeightsForJit() {
    if (!m_isDynamicJitWtsLoaded) {
      return;
    }

    if (m_JitWtsLoader) {
      m_JitWtsLoader->ReleaseWeights();
      m_isDynamicJitWtsLoaded = false;
    }
  }

  void DynamicLoadWeightsForJit() {
    if (m_isDynamicJitWtsLoaded) {
      return;
    }

    if (m_JitWtsLoader) {
      m_JitWtsLoader->DynamicLoadWeights();
      m_isDynamicJitWtsLoaded = true;
    }
  }

  void SetDynamicJitWtsLoaded(bool isJitWtsLoaded) {
    m_isDynamicJitWtsLoaded = isJitWtsLoaded;
  }

  bool IsDynamicJitWtsLoaded() { return m_isDynamicJitWtsLoaded; }

  void UpdateBindings(
    const std::string& opName,
    const std::vector<OnnxTensorInfo>& onnxTensorInput,
    const std::vector<OnnxTensorInfo>& onnxTensorOutput, bool rebindIO,
    bool updateTensors = true
  );

  void ReleaseJitWts();
  void UpdateJitParamsForOp(const std::string& opName);

 private:
  explicit DMLOps(const bool& mapGPURescForCPU);
  DMLOps(
    const bool& mapGPURescForCPU, CComPtr<ID3D12Device3> d3d12Device,
    const std::optional<std::unordered_map<std::string, std::string>>
      session_config = std::nullopt
  );
  std::shared_ptr<Context> m_Context;
  // BF16ToFP16Shader* bf16_ptr;
  std::unique_ptr<BF16ToFP16Shader> bf16_ptr;
  bool m_isGPURescMappedForCPU;
  bool m_IsCustomAllocatorUsed = false;
  std::unordered_map<std::string, std::shared_ptr<DmlOperator>>
    m_OpLists;  // List of all the operator objects

  std::unordered_map<std::string, std::string> m_session_configs;

  // Chaining operators
  // List of all ops that are recorded for chaining
  std::vector<std::shared_ptr<DmlOperator>> m_recorded_op;
  // Name of the output to match with the input of the next op
  std::string m_op_output_name;
  std::string m_submit_op_name;   // Name of the last op that was recorded
  bool m_enable_chaining = true;  // Flag to enable chaining of ops

  // Jit Weights Loader
  std::unique_ptr<JitWtsLoader> m_JitWtsLoader = nullptr;
  bool m_isDynamicJitWtsLoaded = false;
};
}  // namespace DML_Ops

}  // namespace ryzenai::onnx_utils
