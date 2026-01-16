// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <atlbase.h>

#include <vector>

#include "DirectML.h"
#include "gpuprofile.h"
#include "tensor.h"
#include "types.h"

namespace ryzenai::onnx_utils {

class Context;

// An enum which lists all supported activation functions.
enum class ActivationFunc : int32 {
  None = 0,     // There is no activation function.
  Elu,          // f(x) = p1 * (e^x - 1) for x < 0, f(x) = x otherwise
  HardSigmoid,  // f(x) = max(0, min(1, p1 * x + p2))
  LeakyRelu,    // f(x) = p1 * x for x < 0, f(x) = x otherwise
  Linear,       // f(x) = p1 * x + p2
  PSoftPlus,    // f(x) = p1 * ln(e^(p2 * x) + 1)
  Relu,         // f(x) = max(0, x)
  Selu,        // f(x) = p2 * (p1 * e^x - p1) for x < 0, f(x) = p2 * x otherwise
  ScaledTanh,  // f(x) = p1 * tanh(p2 * x)
  Sigmoid,     // f(x) = 1 / (1 + e^-x)
  SoftPlus,    // f(x) = ln(e^x + 1)
  SoftSign,    // f(x) = x / (1 + |x|)
  Tanh,        // f(x) = tanh(x)
  TRelu,       // f(x) = x for x > p1, f(x) = 0 otherwise
  Count
};

// Wraps an ActivationFunc with its constant parameters.
struct ActivationFuncInfo {
  ActivationFunc func;
  float param1;
  float param2;
};

// A helper struct which encapsulates all state necessary to describe a DirectML
// tensor.
struct DmlTensorDesc {
  DML_TENSOR_DESC desc;           // We need the generic base struct.
  DML_BUFFER_TENSOR_DESC buffer;  // The concrete buffer tensor.
  std::vector<UINT> sizes;        // Storage for the size array.
  std::vector<UINT> strides;      // And storage for the stride array.
};

// A helper struct which encapsulates all state necessary to describe a DirectML
// activation function.
struct DmlActivationFuncDesc {
  DML_OPERATOR_DESC desc;  // We need the generic base struct.

  // Every single specific function struct in a union.
  union {
    DML_ACTIVATION_ELU_OPERATOR_DESC elu;
    DML_ACTIVATION_HARDMAX_OPERATOR_DESC hardMax;
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC hardSigmoid;
    DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity;
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leakyRelu;
    DML_ACTIVATION_LINEAR_OPERATOR_DESC linear;
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC logSoftMax;
    DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC paramRelu;
    DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC paramSoftPlus;
    DML_ACTIVATION_RELU_OPERATOR_DESC relu;
    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC selu;
    DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC scaledTanh;
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid;
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softMax;
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC softPlus;
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC softSign;
    DML_ACTIVATION_TANH_OPERATOR_DESC tanh;
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC trelu;
  };
};

// struct OpConstants
//{
//     Tensor weights;
//     Tensor scale;
// };

// =====================================================================================================================
// This class abstracts a DirectML operator and its tensor descriptions.
class DmlOperator {
 public:
  DmlOperator();
  virtual ~DmlOperator();

  virtual void InitializeAndBindOperator(Context* pCtx);
  virtual void RebindInputAndOutput();
  virtual void RebindOutputs();
  // virtual void UploadConstData(Context* pCtx, const OpConstants &consts);
  const TensorDescVector& GetInputTensorDescVector() const {
    return m_inputTensorDescVec;
  }
  const TensorDescVector& GetOutputTensorDescVector() const {
    return m_outputTensorDescVec;
  }

  const TensorVector& GetInputTensorVector() const {
    return m_inputAndConstTensorVector;
  }
  void SetInputTensorVector(uint32 id, const std::shared_ptr<Tensor>& tensor) {
    m_inputAndConstTensorVector[id] = tensor;
  }
  const TensorVector& GetOutputTensorVector() const {
    return m_outputTensorVector;
  }

  virtual void SetConstantForInputTensorDesc(uint32 idx, bool isConstant) {
    m_inputTensorDescVec[idx]->isConst = isConstant;
  }

  CComPtr<IDMLCompiledOperator> GetOperator() const { return m_operator; }

  // The DirectML tensor binding order may not match our tensor order. These
  // functions handle that issue.
  virtual void BindInputs(
    CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
  );
  virtual void BindOutputs(
    CComPtr<IDMLBindingTable> bindingTable, const TensorVector& outputs
  );

  // Initialize the input/output tensors
  virtual void InitializeTensors(
    Context* pContext, bool forceCreateInputD3DResc = false,
    bool forceCreateOutputD3DResc = false
  );

  //// Initialize the input/output tensors
  // virtual std::vector<void*>& GetMappedInputTensors();
  // virtual std::vector<void*>& GetMappedOutputTensors();

  // workaround : update node mappings in of gqo operator
  virtual void UpdateNodeMappings() {}
  virtual void DynamicInitialization(
    [[maybe_unused]] const Context& pContext, [[maybe_unused]] void* params,
    [[maybe_unused]] bool rebuildOp = false
  ) {}

  virtual bool IsFirstRun() { return m_firstRun; }
  void FirstRunDone() { m_firstRun = false; }

  void CompileGraph(
    const Context& pContext,
    const std::vector<CComPtr<IDMLOperator>>& dmlOperators,
    const std::vector<DML_INPUT_GRAPH_EDGE_DESC>& inputEdges,
    const std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC>& intermediateEdges,
    const std::vector<DML_OUTPUT_GRAPH_EDGE_DESC>& outputEdges
  );

  // QKV offset vector
  virtual std::vector<int64_t> GetQKVOffsetsVector() { return {0}; }

#ifdef DEBUG
  static DML_TENSOR_DATA_TYPE StringToDataType(hstring dataTypeStr);
#endif

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  std::map<int, Duration> m_measurements;
#endif

  CComPtr<ID3D12DescriptorHeap>
    m_descHeap;  // Holds descriptors for IDMLDispatchable objects.
  CComPtr<IDMLBindingTable>
    m_bindingTable;  // DML wrapper over the descriptor heap.
  std::vector<std::shared_ptr<DmlOperator>> GetFusedOperatorList() {
    return m_fusedOps;
  }

  void UpdateResourceForJit(
    CComPtr<ID3D12Resource> resource, void* mappedResc
  ) {
    m_jitResource = resource;
    m_cpuMappedJitResource = mappedResc;
  }

  void ReleaseResourceForJit() {
    if (m_jitResource) {
      m_jitResource.Release();
      m_jitResource = nullptr;
      m_cpuMappedJitResource = nullptr;
    }
  }

  void* GetMappedJitResource() { return m_cpuMappedJitResource; }

  void SetHeapIdForJit(uint32_t heapId) { m_heapId = heapId; }
  void SetJitEnabled(bool enabled) { m_isJitEnabled = enabled; }
  bool IsJitEnabled() const { return m_isJitEnabled; }
  void SetDynamicOp(bool isDynamicOp) { m_isJitDynamicOp = isDynamicOp; }

  void MarkRepackIfOddDim(std::shared_ptr<TensorDesc>& pDesc, bool isGpuJit);

  // Local Window Attention related
  virtual const int32 GetLocalWindowSize() const { return 0; }

 protected:
  // Helper functions used by child classes.
  static hstring DataTypeToString(DML_TENSOR_DATA_TYPE dataType);

#ifndef DEBUG
  static DML_TENSOR_DATA_TYPE StringToDataType(hstring dataTypeStr);
#endif

  static hstring ActivationFuncToString(ActivationFunc func);
  static ActivationFunc StringToActivationFunc(hstring activationFuncStr);

  static std::vector<int64> ComputeStrides(const std::vector<int64>& shape);
  static std::shared_ptr<TensorDesc> CreateTensorDesc(
    hstring name, hstring order, DML_TENSOR_DATA_TYPE dataType,
    const std::vector<int64>& shape, const std::vector<int64>& strides
  );

  static void ConvertTensorDesc(
    const TensorDesc& tensorDesc, DmlTensorDesc* pDmlDesc
  );
  static void ConvertActivationFuncInfo(
    const ActivationFuncInfo& info, DmlActivationFuncDesc* pDmlDesc
  );
  void BuildBufferBindDesc(
    const std::shared_ptr<Tensor>& pTensor, DML_BUFFER_BINDING* pBinding,
    DML_BINDING_DESC* pDesc
  );

  CComPtr<IDMLCompiledOperator> m_operator;
  TensorDescVector m_inputTensorDescVec;
  TensorDescVector m_outputTensorDescVec;

  // These would hold D3D resources for inputs and outputs
  TensorVector m_inputAndConstTensorVector;
  TensorVector m_outputTensorVector;

  // If Jit is enabled, use heapId to get the d3d resource
  bool m_isJitEnabled = false;
  bool m_isJitDynamicOp = false;
  uint32_t m_heapId = 0;
  CComPtr<ID3D12Resource> m_jitResource;
  void* m_cpuMappedJitResource = nullptr;

  // Constants for the operator
  // OpConstants opConsts;

  std::vector<void*> m_mappedInputTensors;
  std::vector<void*> m_mappedOutputTensors;

  // Members from Context class. They should be per operator.
  // CComPtr<ID3D12DescriptorHeap>
  //   m_descHeap;  // Holds descriptors for IDMLDispatchable objects.
  // CComPtr<IDMLBindingTable>
  //   m_bindingTable;  // DML wrapper over the descriptor heap.
  CComPtr<ID3D12Resource> m_temporaryRes;   // The DML temporary resource.
  CComPtr<ID3D12Resource> m_persistentRes;  // The DML persistent resource.
  UINT m_descCount;  // The number of descriptors in the descriptor heap.
  UINT64 m_temporaryResSize;   // The size of the temporary resource in bytes.
  UINT64 m_persistentResSize;  // The size of the persistent resource in bytes.

  // List of ops to be fused with this op
  std::vector<std::shared_ptr<DmlOperator>> m_fusedOps;
  bool m_firstRun = true;
};

}  // namespace ryzenai::onnx_utils
