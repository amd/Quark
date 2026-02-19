// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "opInterface.h"

#include "../external_data.pb.h"
// #include "../operators/opUtils.h"

namespace ryzenai::onnx_utils {

// This macro is required to turn off or on the GQO graph
#define GQO_GRAPH

namespace DML_Ops {
std::shared_ptr<DMLOps> DMLOps::getInstance(
  const std::unordered_map<std::string, std::string>& session_config,
  std::optional<CComPtr<ID3D12Device3>> d3d12Device
) {
  static std::unordered_map<std::string, std::weak_ptr<DMLOps>>
    weak_per_session;
  static std::mutex weak_per_session_mutex;

  std::unique_lock<std::mutex> lock(weak_per_session_mutex);

  auto& weak = weak_per_session[session_config.at("session_id")];
  auto strong = weak.lock();

  if (!strong) {
    strong.reset(new DMLOps(true, d3d12Device.value(), session_config));
    weak = strong;
  }

  return strong;
}

//=====================================================================================================
DMLOps::DMLOps(const bool& mapGPURescForCPU)
  : DMLOps(mapGPURescForCPU, nullptr) {};

DMLOps::DMLOps(
  const bool& mapGPURescForCPU, CComPtr<ID3D12Device3> d3d12Device,
  const std::optional<std::unordered_map<std::string, std::string>>
    session_config
)
  : m_isGPURescMappedForCPU(mapGPURescForCPU) {
  ContextInfo m_ContextInfo = {
    0,  // The index of the first adapter to consider.
#ifdef DEBUG
    true,  // If the D3D12 debug layer should be enabled.
#else
    false,
#endif

    m_isGPURescMappedForCPU,  // If all tensors only use persistently mapped
                              // system memory for CPU or NPU to use (no
                              // copies!).
    false,                    // Make the tensor always resident.
    !m_isGPURescMappedForCPU  // If we get tensor data externally. This and
                              // systemMem should be opposite
  };

  // m_ContextInfo.useCustomAllocator = (nullptr != d3d12Device) ? true : false;
  m_IsCustomAllocatorUsed = (nullptr != d3d12Device);

  m_enable_chaining &= m_IsCustomAllocatorUsed;

  m_Context = std::make_shared<Context>(m_ContextInfo, d3d12Device);
  bf16_ptr =
    std::make_unique<BF16ToFP16Shader>(m_Context, m_IsCustomAllocatorUsed);

  // Jit weights loader
  const auto& external_data_file =
    session_config.value().at("external_data_file");

  // Used for 2 purposes
  // 1. load external header file if needed
  // 2. get parent path to find where JIT weights are
  if (external_data_file.empty()) {
    throw std::invalid_argument(
      "external_data_file not specified in the session options"
    );
  }

  const auto& external_data_blob_size_str =
    session_config.value().at("external_data_blob_size");

  std::unique_ptr<proto::Header> protoHeader =
    std::make_unique<proto::Header>();
  size_t external_data_blob_size = 0;

  if (!external_data_blob_size_str.empty()) {
    external_data_blob_size = std::stoull(external_data_blob_size_str);
  }

  if (external_data_blob_size) {
    const void* external_data_blob =
      (const void*)std::stoull(session_config.value().at("external_data_blob"));

    if (!protoHeader->ParseFromArray(
          external_data_blob, external_data_blob_size
        )) {
      throw std::invalid_argument("Cannot read in memory header");
    }
  } else {
    if (std::fstream header_stream(
          external_data_file, std::ios::in | std::ios::binary
        );
        !protoHeader->ParseFromIstream(&header_stream)) {
      throw std::invalid_argument(
        "Cannot read header from " + external_data_file
      );
    }
  }

  if (protoHeader->external_data().gpu()) {
    m_JitWtsLoader = std::make_unique<JitWtsLoader>(
      session_config.value(), std::move(protoHeader)
    );
  }

  m_Context->SetIsGpuJitEnabled(IsJitWtsLoaderEnabled());

  // Print the adapter name to let the user know which GPU was selected.
  printMessage(
    "DirectML is executing on: ", winrt::to_string(m_Context->AdapterName())
  );
}

//=====================================================================================================
DMLOps::~DMLOps() {
  bf16_ptr.reset();
  m_Context.reset();
  m_JitWtsLoader.reset();
}

//=====================================================================================================
const bool& DMLOps::isGPURescMappedForCPU() const {
  return m_isGPURescMappedForCPU;
}

//=====================================================================================================
const Context* DMLOps::getContext() const { return m_Context.get(); }

//=====================================================================================================
void DMLOps::SetTensorResource(
  std::shared_ptr<Tensor>& tensor, const OnnxTensorInfo& onnxTensorInfo,
  uint64_t offset
) {
  SetTensorConstFlags(tensor, onnxTensorInfo);
  tensor->SetResource(
    onnxTensorInfo.d3dResource, onnxTensorInfo.pCpuMappedD3DResc, offset
  );
}

//=====================================================================================================
void DMLOps::SetTensorConstFlags(
  std::shared_ptr<Tensor>& tensor, const OnnxTensorInfo& onnxTensorInfo
) {
  tensor->SetJitConstFlag(onnxTensorInfo.isConstForJit);
  tensor->SetConstFlag(onnxTensorInfo.isExternalBufferConstant);
}

//=====================================================================================================
void DMLOps::SetConstants(
  std::shared_ptr<DmlOperator>& pOperator,
  const std::vector<OnnxTensorInfo>& onnxTensorInput,
  const std::vector<OnnxTensorInfo>& onnxTensorOutput
) {
  TensorVector inputs = pOperator->GetInputTensorVector();
  TensorVector outputs = pOperator->GetOutputTensorVector();

  for (size_t i = 0; i < inputs.size(); i++) {
    if (onnxTensorInput[i].isExternalBufferConstant) {
      SetTensorConstFlags(inputs[i], onnxTensorInput[i]);

      if (inputs[i]->IsRepackNeeded()) {
        auto buffer = onnxTensorInput[i].pCpuMappedD3DResc;
        if (inputs[i]->IsJitConst()) {
          buffer = pOperator->GetMappedJitResource();
        }
        inputs[i]->RepackInt4TensorBuffer(
          (char*)buffer + onnxTensorInput[i].offsetInBytes
        );
      }
    }
  }
}

//=====================================================================================================
void DMLOps::UpdateResourceHandle(
  const std::string& opName, const std::vector<OnnxTensorInfo>& onnxTensorInput,
  const std::vector<OnnxTensorInfo>& onnxTensorOutput
) {
  TensorVector inputs = m_OpLists[opName]->GetInputTensorVector();
  TensorVector outputs = m_OpLists[opName]->GetOutputTensorVector();

  // keep track of correct tensorinput/output
  uint32_t j = 0;

  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      if (inputs[i]->IsResourceUpdateNeeded()) {
        const auto& ti = onnxTensorInput[j];
        inputs[i]->SetResource(
          ti.d3dResource, ti.pCpuMappedD3DResc, ti.offsetInBytes
        );
        // onnx buffer data
        inputs[i]->UpdateTensorBuffer(ti.pExternalBuffer);
      }
      j++;
    }
  }

  j = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr && outputs[i]->IsResourceUpdateNeeded()) {
      const auto& to = onnxTensorOutput[j];
      outputs[i]->SetResource(
        to.d3dResource, to.pCpuMappedD3DResc, to.offsetInBytes
      );
      j++;
    }
  }
}

//=====================================================================================================
// This function will be used by Jit weight during prefill.
// The d3d resource can change between multiple prompts
void DMLOps::UpdateBindings(
  const std::string& opName, const std::vector<OnnxTensorInfo>& onnxTensorInput,
  const std::vector<OnnxTensorInfo>& onnxTensorOutput, bool rebindIO,
  bool updateTensors
) {
  bool releaseJitWts = false;
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];
  if (updateTensors) {
    UpdateResourceHandle(opName, onnxTensorInput, onnxTensorOutput);
  }

  if (IsJitWtsLoaderEnabled()) {
    auto heapId = m_JitWtsLoader->GetHeapId(opName);
    const auto& heapInfo = m_JitWtsLoader->GetHeapSizeDesc(heapId);

    if (heapInfo.isDynamicHeap) {
      m_JitWtsLoader->IsWeightsReady(heapId);
      pOperator->UpdateResourceForJit(
        heapInfo.GetD3DResource(), heapInfo.buffer.Data()
      );
      rebindIO = true;
      releaseJitWts = true;
    }
  }

  if (rebindIO) {
    pOperator->RebindInputAndOutput();
  }

  if (releaseJitWts) {
    pOperator->ReleaseResourceForJit();
  }
}

//=====================================================================================================
void DMLOps::ReleaseJitWts() {
  if (IsJitWtsLoaderEnabled()) {
    m_JitWtsLoader->ReleaseWeights();
  }
}

//=====================================================================================================
void DMLOps::UpdateJitParamsForOp(const std::string& opName) {
  if (IsJitWtsLoaderEnabled()) {
    const auto heapId = m_JitWtsLoader->GetHeapId(opName);
    m_OpLists[opName]->SetJitEnabled(true);
    m_OpLists[opName]->SetHeapIdForJit(heapId);
    m_OpLists[opName]->SetDynamicOp(m_JitWtsLoader->IsDyamicHeap(heapId));
    const auto& heapInfo = m_JitWtsLoader->GetHeapSizeDesc(heapId);
    m_OpLists[opName]->UpdateResourceForJit(
      heapInfo.GetD3DResource(), heapInfo.buffer.Data()
    );
  }
}

//=====================================================================================================
void DMLOps::CheckForOpChaining(
  const std::string& opInputName, const std::string& opOutputName,
  const std::string& opName
) {
  if (!m_enable_chaining) {
    return;
  }
  // record the last custom op which will be used as a condition
  // to submit/execute the operators
  m_submit_op_name = opName;
  if (m_op_output_name.empty()) {
    m_op_output_name = opOutputName;
  } else {
    if (m_op_output_name.compare(opInputName) != 0) {
      m_enable_chaining = false;
    }
    m_op_output_name = opOutputName;
  }
}

//=====================================================================================================
inline void DMLOps::RecordOrExecuteOperator(
  const std::shared_ptr<DmlOperator>& pOperator, const std::string& opName
) {
  if (m_enable_chaining) {
    m_recorded_op.push_back(pOperator);
    if (opName == m_submit_op_name) {
      m_Context->ExecuteOperator(m_recorded_op);
      m_recorded_op.clear();
    }
  } else {
    m_Context->ExecuteOperator({pOperator});
  }
}

//=====================================================================================================
void DMLOps::ReinitializeAndBindIO(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  // Todo: Any init code should go hear.
  UpdateResourceHandle(opName, pTensorDescInput, pTensorDescOutput);

  m_OpLists[opName]->RebindInputAndOutput();
}

//=====================================================================================================
void DMLOps::CreateGatherOperator(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  GatherParams params;

  params.batch = pTensorDescInput[0].shape[0];
  params.dataType = pTensorDescInput[0].dataType;
  params.inputShape = pTensorDescInput[1].shape;
  params.indicesShape = pTensorDescInput[0].shape;

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<GatherOperator>(*m_Context, 0, params);

  /*m_OpLists[opName]->InitializeTensors(m_Context);
  m_OpLists[opName]->InitializeAndBindOperator(m_Context);
  UpdateResourceHandleAndTensorFlags(opName, pTensorDescInput,
                                     pTensorDescOutput);*/
}

//=====================================================================================================
void DMLOps::CreateMatMulOperator(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  GemmParams params;
  params.m = pTensorDescInput[0].shape[1];
  params.n = pTensorDescInput[0].shape[2];
  params.k = pTensorDescInput[1].shape[1];
  params.batches = pTensorDescInput[0].shape[0];
  params.dataType = pTensorDescInput[0].dataType;

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<GemmOperator>(*m_Context, 0, params);

  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

//=====================================================================================================
void DMLOps::CreateMatMulNBitsOperator(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput, int64_t block_size
) {
  MatMulNBitsParams params;
  params.m = pTensorDescInput[0].shape[1];
  params.k = pTensorDescInput[0].shape[2];
  params.n = pTensorDescInput[1].shape[0];
  params.batches = pTensorDescInput[0].shape[0];

  params.dataType = pTensorDescInput[0].dataType;
  params.quantizedBlockB = block_size;
  params.hasZeroPoint = pTensorDescInput.size() > 3;
  params.hasC = pTensorDescInput.size() > 4;
  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<MatMulNBitsOperator>(*m_Context, params);
}

//=====================================================================================================
void DMLOps::InitializeMatMulNBits(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  if (!m_OpLists[opName]->IsFirstRun()) {
    return;
  }
  m_OpLists[opName]->FirstRunDone();
  UpdateJitParamsForOp(opName);

  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

//=====================================================================================================
void DMLOps::CreateSlrnOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutpu,
  const AttributeForSlrn& attr
) {
  NormParams params;
  auto inputSize = pTensorDescInput[0].shape.size();
  params.inputShape.resize(inputSize);
  for (int i = 0; i < inputSize; i++)
    params.inputShape[i] = pTensorDescInput[0].shape[i];

  params.scaleShape = pTensorDescInput[1].shape[0];
  // if size of this vector is more than 2 then bias is present from onnx
  if (pTensorDescInput.size() > 2) {
    params.hasBias = true;
  }

  params.dataType = pTensorDescInput[0].dataType;
  params.epsilon = attr.epsilon;
  params.isWtsFp32 =
    (winrt::to_string(pTensorDescInput[1].dataType) == "Float");

  // check if scale shape and input hiddensize are same
  if (!attr.shapeIn.empty() &&
      (params.scaleShape != pTensorDescInput[0].shape.back())) {
    params.shapeIn.resize(attr.shapeIn.size());
    params.shapeOut.resize(attr.shapeOut.size());
    for (int j = 0; j < attr.shapeIn.size(); j++) {
      params.shapeIn[j] = attr.shapeIn[j];
      params.shapeOut[j] = attr.shapeOut[j];
    }
  }

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<NormOperator>(*m_Context, params);
}

//=====================================================================================================
void DMLOps::InitializeSlrn(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

//=====================================================================================================
void DMLOps::ComputeSlrn(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput, bool rebindIO
) {
  printMessage("Start: Executing SimplifiedLayerNorm on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    // If block in the loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    if (false == m_IsCustomAllocatorUsed) {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != nullptr)
          inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
      }
      for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i] != nullptr)
          outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
      }
    }

    UpdateBindings(opName, pBufInput, pBufOutput, rebindIO);
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point upload_start = Clock::now();
#endif

    m_Context->UploadInputOrConstData(inputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point execute_start = Clock::now();
#endif

    RecordOrExecuteOperator(pOperator, opName);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_start = Clock::now();
#endif

    m_Context->DownloadData(outputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_end = Clock::now();
    const Duration upload_duration = execute_start - upload_start;
    const Duration execute_duration = download_start - execute_start;
    const Duration download_duration = download_end - download_start;

    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_UPLOAD_ID] +=
      upload_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_EXECUTE_ID] +=
      execute_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_DOWNLOAD_ID] +=
      download_duration;
#endif

  } catch (const std::exception& exception) {
    std::cout << "SimplifiedLayerNorm Exception: " << exception.what()
              << std::endl;
  }

  printMessage("End: Executing SimplifiedLayerNorm on GPU");
}

//=====================================================================================================
void DMLOps::CreateRotaryEmbeddingOperator(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput, bool interleaved
) {
  RotaryEmbeddingParams params;
  params.inputShape = pTensorDescInput[0].shape;
  params.cosShape = pTensorDescInput[2].shape;
  params.sinShape = pTensorDescInput[3].shape;

  params.interleaved = interleaved;
  params.dataType = pTensorDescInput[0].dataType;

  params.batchSize = params.inputShape[0];
  params.sequenceLength = params.inputShape[1];

  params.headSize = params.cosShape.back() * 2;
  params.numHeads = params.inputShape.back() / params.headSize;

  // Create a operator from each parameter struct.
  m_OpLists[opName] =
    std::make_shared<RotaryEmbeddingOperator>(*m_Context, params);
}

//=====================================================================================================
void DMLOps::Initialize(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  // Todo: Any init code should go here.
  m_OpLists[opName]->InitializeTensors(m_Context.get());
  SetConstants(m_OpLists[opName], pTensorDescInput, pTensorDescOutput);
  UpdateResourceHandle(opName, pTensorDescInput, pTensorDescOutput);

  m_OpLists[opName]->InitializeAndBindOperator(m_Context.get());

  // when custom allocator is used, then we dont need to upload constants.
  //  this function will return without doing anything then
  m_Context->UploadConstData(m_OpLists[opName]->GetInputTensorVector());
}

//=====================================================================================================
void DMLOps::InitializeRotaryEmbedding(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

// =====================================================================================================
void DMLOps::InitializeSSMLP(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  if (!m_OpLists[opName]->IsFirstRun()) {
    return;
  }
  m_OpLists[opName]->FirstRunDone();
  UpdateJitParamsForOp(opName);

  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

// =====================================================================================================
void DMLOps::CreateSSMLPOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput,
  const AttributeForSSMLP& attr
) {
  SSMLPGeluParams params;

  std::vector<int32> inputShape;
  inputShape.resize(pTensorDescInput[0].shape.size());
  for (int i = 0; i < pTensorDescInput[0].shape.size(); i++) {
    inputShape[i] = pTensorDescInput[0].shape[i];
  }

  if (attr.has_gelu == 1) {
    params.normTop.inputShape = inputShape;
    params.normTop.dataType = pTensorDescInput[1].dataType;
    params.normTop.epsilon = attr.epsilon;
    params.normTop.hasScale = true;
    params.normTop.hasBias = false;
    params.normTop.isWtsFp32 =
      (winrt::to_string(pTensorDescInput[2].dataType) == "Float");

    params.normBottom.inputShape = inputShape;
    params.normBottom.dataType = pTensorDescInput[1].dataType;
    params.normBottom.epsilon = attr.epsilon;
    params.normBottom.hasScale = true;
    params.normBottom.hasBias = false;
    params.normBottom.isWtsFp32 = params.normTop.isWtsFp32;
  }

  params.sslrnTop.inputShape = inputShape;
  params.sslrnTop.dataType = pTensorDescInput[0].dataType;
  params.sslrnTop.epsilon = attr.epsilon;
  params.sslrnTop.hasScale = true;
  params.sslrnTop.hasBias = false;
  params.sslrnTop.hasNonMVNBias = false;
  params.sslrnTop.isWtsFp32 =
    (winrt::to_string(pTensorDescInput[3].dataType) == "Float");

  params.sslrnBottom.inputShape =
    params.sslrnTop.inputShape;  // shapes for both the SSLRN should be same
  params.sslrnBottom.dataType = params.sslrnTop.dataType;
  params.sslrnBottom.epsilon = attr.epsilon;
  params.sslrnBottom.hasScale = true;
  params.sslrnBottom.hasBias = false;
  params.sslrnBottom.hasNonMVNBias = false;
  params.sslrnBottom.outputCount = pTensorDescOutput.size();
  params.sslrnBottom.isWtsFp32 = params.sslrnTop.isWtsFp32;

  params.matMulNBitsGate.hasC = false;
  params.matMulNBitsGate.hasZeroPoint =
    true;  // based on the llama3 fused graph
  params.matMulNBitsGate.m = 1;
  params.matMulNBitsGate.n = attr.gate_N;
  params.matMulNBitsGate.k = attr.gate_K;
  params.matMulNBitsGate.quantizedBlockB = attr.gateBlockSize;

  params.matMulNBitsUp.hasC = false;
  params.matMulNBitsUp.hasZeroPoint = true;  // based on the llama3 fused graph
  params.matMulNBitsUp.m = 1;
  params.matMulNBitsUp.n = attr.up_N;
  params.matMulNBitsUp.k = attr.up_K;
  params.matMulNBitsUp.quantizedBlockB = attr.upBlockSize;

  params.matMulNBitsDown.hasC = false;
  params.matMulNBitsDown.hasZeroPoint =
    true;  // based on the llama3 fused graph
  params.matMulNBitsDown.m = 1;
  params.matMulNBitsDown.n = attr.down_N;
  params.matMulNBitsDown.k = attr.down_K;
  params.matMulNBitsDown.quantizedBlockB = attr.downBlockSize;

  // Create a operator from each parameter struct.
  if (attr.has_gelu == 1) {
    m_OpLists[opName] =
      std::make_shared<SSMLPGeluOperator>(*m_Context, 0, params);
  } else {
    m_OpLists[opName] = std::make_shared<SSMLPOperator>(*m_Context, 0, params);
  }
}

// =====================================================================================================================
void DMLOps::ComputeMatMulNBitsGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput, bool rebindIO
) {
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];
  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    if (false == m_IsCustomAllocatorUsed) {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != nullptr)
          inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
      }
      for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i] != nullptr)
          outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
      }
    }

    UpdateBindings(opName, pBufInput, pBufOutput, rebindIO);

    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point upload_start = Clock::now();
#endif

    m_Context->UploadInputData(inputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point execute_start = Clock::now();
#endif

    RecordOrExecuteOperator(pOperator, opName);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_start = Clock::now();
#endif

    m_Context->DownloadData(outputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_end = Clock::now();
    const Duration upload_duration = execute_start - upload_start;
    const Duration execute_duration = download_start - execute_start;
    const Duration download_duration = download_end - download_start;

    m_OpLists[opName]->m_measurements[GPUEventID::MATMULNBITS_UPLOAD_ID] +=
      upload_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::MATMULNBITS_EXECUTE_ID] +=
      execute_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::MATMULNBITS_DOWNLOAD_ID] +=
      download_duration;
#endif
  } catch (const std::exception& exception) {
    std::cout << "Exception: " << exception.what() << std::endl;
  }
}

// =====================================================================================================
void DMLOps::ComputeMatMulGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing Matmul on GPU");

  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();
    ;
    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "Matmul Exception: " << exception.what() << std::endl;
  }

  printMessage("End: Executing Matmul on GPU");
}

//=====================================================================================================
void DMLOps::ComputeRotaryEmbeddingGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing Rotary Embedding on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();
    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "Exception: " << exception.what() << std::endl;
  }

  printMessage("End: Executing Rotary Embedding on GPU");
}

// =====================================================================================================
void DMLOps::ComputeGatherGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing Gather on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();
    ;
    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "Matmul Exception: " << exception.what() << std::endl;
  }

  printMessage("End: Executing Gather on GPU");
}

//=====================================================================================================
void DMLOps::CreateSkipSimplifiedLayerNormOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutpu, const float& epsilon
) {
  SkipSimplifiedLayerNormParams params;
  params.inputShape.resize(pTensorDescInput[0].shape.size());
  for (int i = 0; i < pTensorDescInput[0].shape.size(); i++)
    params.inputShape[i] = pTensorDescInput[0].shape[i];

  // if size of this vector is more than 1 then scale is present from onnx
  if (pTensorDescInput.size() > 2) {
    params.hasScale = true;
  }

  // if size of this vector is more than 1 then bias is present from onnx
  if (pTensorDescInput.size() > 3) {
    params.hasBias = true;
  }

  if (pTensorDescInput.size() > 4) {
    params.hasNonMVNBias = true;
  }

  params.dataType = pTensorDescInput[0].dataType;

  // Create a operator from each parameter struct.
  m_OpLists[opName] =
    std::make_shared<SkipSimplifiedLayerNormOperator>(*m_Context, 0, params);
}

//=====================================================================================================
void DMLOps::InitializeSkipSimplifiedLayerNorm(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

//=====================================================================================================
void DMLOps::ComputeSkipSimplifiedLayerNormGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing SimplifiedLayerNorm on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    // If block in the loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "SimplifiedLayerNorm Exception: " << exception.what()
              << std::endl;
  }

  printMessage("End: Executing SimplifiedLayerNorm on GPU");
}

//=====================================================================================================
void DMLOps::ComputeSSMLPGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput, bool rebindIO
) {
  printMessage("Start: Executing SSMLP on GPU");

  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    // If block in the loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    if (false == m_IsCustomAllocatorUsed) {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i] != nullptr)
          inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
      }
      for (size_t i = 0; i < outputs.size(); i++) {
        if (outputs[i] != nullptr)
          outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
      }
    }

    UpdateBindings(opName, pBufInput, pBufOutput, rebindIO);

    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point upload_start = Clock::now();
#endif

    m_Context->UploadInputOrConstData(inputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point execute_start = Clock::now();
#endif

    RecordOrExecuteOperator(pOperator, opName);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_start = Clock::now();
#endif

    m_Context->DownloadData(outputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
    const Clock::time_point download_end = Clock::now();
    const Duration upload_duration = execute_start - upload_start;
    const Duration execute_duration = download_start - execute_start;
    const Duration download_duration = download_end - download_start;

    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_UPLOAD_ID] +=
      upload_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_EXECUTE_ID] +=
      execute_duration;
    m_OpLists[opName]->m_measurements[GPUEventID::SS_MLP_DOWNLOAD_ID] +=
      download_duration;
#endif

  } catch (const std::exception& exception) {
    std::cout << "SSMLP Exception: " << exception.what() << std::endl;
  }

  printMessage("End: Executing SSMLP on GPU");
}

//=====================================================================================================
void DMLOps::CreateReduceOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutpu
) {
  ReduceParams params;
  params.reduceFunction = DML_REDUCE_FUNCTION_SUM;
  params.inputShape.resize(pTensorDescInput[0].shape.size());
  for (int i = 0; i < pTensorDescInput[0].shape.size(); i++) {
    params.inputShape[i] = pTensorDescInput[0].shape[i];
  }

  params.axes.resize(pTensorDescInput[1].shape.size());
  for (int i = 0; i < pTensorDescInput[1].shape.size(); i++) {
    params.axes[i] = pTensorDescInput[1].shape[i];
  }

  params.dataType = pTensorDescInput[0].dataType;

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<ReduceOperator>(*m_Context, 0, params);
}

//=====================================================================================================
void DMLOps::InitializeReduce(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

//=====================================================================================================
void DMLOps::ComputeReduceGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing Reduce on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    // If block in the loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "Reduce Exception: " << exception.what() << std::endl;
  }
  printMessage("End: Executing Reduce on GPU");
}

// =====================================================================================================================
void DMLOps::CreateSubOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  SubParams params;

  params.inputShapeA = pTensorDescInput[0].shape;
  params.inputShapeB = pTensorDescInput[1].shape;
  params.outputShape = params.inputShapeA;
  params.dataType = pTensorDescInput[0].dataType;

  m_OpLists[opName] = std::make_shared<SubOperator>(*m_Context, false, params);
}

// =====================================================================================================================
void DMLOps::InitializeSub(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  Initialize(opName, pTensorDescInput, pTensorDescOutput);
}

// =====================================================================================================================
void DMLOps::ComputeSubGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing Sub on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    // Update the inputs/outputs with external data
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator({pOperator});

    m_Context->DownloadData(outputs);

  } catch (const std::exception& exception) {
    std::cout << "Sub Exception: " << exception.what() << std::endl;
  }
  printMessage("End: Executing Sub on GPU");
}

//=====================================================================================================
void DMLOps::CreateGQOOp(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const AttributeForGQO& attr
) {
  GQOParams params;

  // shapes
  params.qkvPackedInputShape = pTensorDescInput[0].shape;

  params.rotEmbQuery.cosShape = pTensorDescInput[5].shape;
  params.rotEmbQuery.sinShape = pTensorDescInput[6].shape;
  params.rotEmbQuery.dataType = pTensorDescInput[0].dataType;
  params.rotEmbQuery.interleaved = static_cast<bool>(attr.rotary_interleaved);
  params.rotEmbQuery.numHeads = attr.num_heads;
  params.rotEmbQuery.rotaryEmbeddingDim = attr.rotary_embedding_dim;

  params.gqa.inputPastKeyShape = pTensorDescInput[1].shape;
  params.gqa.inputPastValueShape = pTensorDescInput[2].shape;
  params.gqa.inputSubCastShape = pTensorDescInput[3].shape;
  params.gqa.inputGatherCastShape = pTensorDescInput[4].shape[0];
  params.gqa.dataType = pTensorDescInput[0].dataType;
  params.gqa.scale = attr.scale;
  params.gqa.do_rotary =
    attr.do_rotary;  // not supported by DML hence RoPE is done outside the GQA
                     // This value is always true for GPU implementation
  params.gqa.kv_num_heads = attr.kv_num_heads;
  params.gqa.num_heads = attr.num_heads;
  params.gqa.local_window_size = attr.local_window_size;
  params.gqa.softcap = attr.softcap;

  params.matMulNBits.hasC = false;
  params.matMulNBits.hasZeroPoint = true;  // based on the llama3 fused graph
  params.matMulNBits.m = 1;
  params.matMulNBits.n = attr.o_proj_N;
  params.matMulNBits.k = attr.o_proj_K;
  params.matMulNBits.quantizedBlockB = attr.o_proj_block_size;

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<GQOOperator>(*m_Context, params);
}

// =====================================================================================================
void DMLOps::GQOOpDynamicInitialization(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput,
  const bool isOutputShapeChanged
) {
  if (!m_OpLists[opName]->IsFirstRun() && !isOutputShapeChanged) {
    return;
  }

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  const Clock::time_point start = Clock::now();
#endif

  // Update params
  GQOParams params;

  // shapes
  params.qkvPackedInputShape = pTensorDescInput[0].shape;

  params.rotEmbQuery.cosShape = pTensorDescInput[5].shape;
  params.rotEmbQuery.sinShape = pTensorDescInput[6].shape;
  params.rotEmbQuery.dataType = pTensorDescInput[0].dataType;

  params.gqa.inputPastKeyShape = pTensorDescInput[1].shape;
  params.gqa.inputPastValueShape = pTensorDescInput[2].shape;
  params.gqa.inputSubCastShape = pTensorDescInput[3].shape;
  params.gqa.inputGatherCastShape = pTensorDescInput[4].shape[0];
  params.gqa.dataType = pTensorDescInput[0].dataType;

  params.gqa.outputKeyShape = pTensorDescOutput[0].shape;
  params.gqa.outputValueShape = pTensorDescOutput[1].shape;
  params.gqa.outputQueryShape = pTensorDescOutput[2].shape;

  params.matMulNBits.hasC = false;
  params.matMulNBits.hasZeroPoint = true;  // based on the llama3 fused graph
  params.matMulNBits.m = 1;

#ifndef GQO_GRAPH
  std::vector<std::shared_ptr<DmlOperator>> fusedOpsList;
  CComPtr<ID3D12Resource> inputForMatmul;
  void* mappedPtr = nullptr;
  uint64_t d3dOffset = 0;
  if (isOutputShapeChanged) {
    fusedOpsList = m_OpLists[opName].get()->GetFusedOperatorList();
    inputForMatmul = fusedOpsList[2]->GetOutputTensorVector()[0]->Resource();
    mappedPtr = fusedOpsList[2]->GetOutputTensorVector()[0]->MappedResource();
    d3dOffset = fusedOpsList[2]->GetOutputTensorVector()[0]->ResourceOffset();
  }
#endif

  m_OpLists[opName]->DynamicInitialization(
    *m_Context, static_cast<void*>(&params), isOutputShapeChanged
  );

#ifndef GQO_GRAPH
  if (isOutputShapeChanged) {
    // op 2 in fused op is GQA
    std::shared_ptr<DmlOperator> pOpGQA =
      m_OpLists[opName].get()->GetFusedOperatorList()[2];
    // std::shared_ptr<DmlOperator> pOpMatmul =
    // m_OpLists[opName].get()->GetFusedOperatorList()[3];
    pOpGQA->InitializeTensors(m_Context, true, false);

    TensorVector inputs = pOpGQA->GetInputTensorVector();
    TensorVector outputs = pOpGQA->GetOutputTensorVector();

    outputs[0]->SetResource(
      inputForMatmul, mappedPtr, d3dOffset, m_Context, false
    );
    outputs[1]->SetResource(
      pTensorDescOutput[0].d3dResource, pTensorDescOutput[0].pCpuMappedD3DResc,
      pTensorDescOutput[0].offsetInBytes, m_Context, false
    );
    outputs[2]->SetResource(
      pTensorDescOutput[1].d3dResource, pTensorDescOutput[1].pCpuMappedD3DResc,
      pTensorDescOutput[1].offsetInBytes, m_Context, false
    );

    m_OpLists[opName]->UpdateNodeMappings();

    pOpGQA->InitializeAndBindOperator(m_Context);
    // pOpMatmul->InitializeAndBindOperator(m_Context);
    return;
  }
#endif

  UpdateJitParamsForOp(opName);
  InitializeGQO(opName, pTensorDescInput, pTensorDescOutput);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  const Clock::time_point end = Clock::now();
  const Duration duration = end - start;

  m_OpLists[opName]->m_measurements[GPUEventID::GQO_DYNAMIC_INIT_ID] +=
    duration;
#endif
}

#ifndef GQO_GRAPH
// =====================================================================================================
void DMLOps::InitializeGQO(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  OnnxTensorInfo tmpOnnxTensorInfo;
  std::vector<OnnxTensorInfo> gqaInputConstantsVec;
  gqaInputConstantsVec.resize(4, tmpOnnxTensorInfo);
  std::vector<OnnxTensorInfo> rotEmbInputConstantsVec = {
    tmpOnnxTensorInfo, tmpOnnxTensorInfo
  };  // added to match the input size of the op
  std::copy(
    pTensorDescInput.begin() + 5, pTensorDescInput.begin() + 7,
    std::back_inserter(rotEmbInputConstantsVec)
  );
  std::vector<OnnxTensorInfo> matmulInputConstantsVec = {
    tmpOnnxTensorInfo
  };  // added to match the input size of the op
  std::copy(
    pTensorDescInput.begin() + 7, pTensorDescInput.begin() + 10,
    std::back_inserter(matmulInputConstantsVec)
  );

  std::shared_ptr<DmlOperator> pGqoOperator = m_OpLists[opName];

  std::vector<std::shared_ptr<DmlOperator>> fusedOpsList =
    pGqoOperator.get()->GetFusedOperatorList();

  bool createD3DReasourceForOutputs = true;
  const bool createInputsInternally = !m_IsCustomAllocatorUsed;

  for (size_t i = 0; i < fusedOpsList.size(); i++) {
    std::shared_ptr<DmlOperator> pOperator = fusedOpsList[i];
    std::vector<OnnxTensorInfo> constVector = {};
    if (i == 0 || i == 1) {
      constVector = rotEmbInputConstantsVec;
    } else if (i == 3) {
      constVector = matmulInputConstantsVec;
      createD3DReasourceForOutputs = false;
    } else {
      constVector = gqaInputConstantsVec;
      createD3DReasourceForOutputs = false;
    }
    if (m_IsCustomAllocatorUsed) {
      for (int i = 0; i < constVector.size(); i++) {
        pOperator->SetConstantForInputTensorDesc(
          i, constVector[i].isExternalBufferConstant
        );
      }
    }
    // Initialize the input/output tensors for the fused operators
    pOperator->InitializeTensors(
      m_Context, createInputsInternally, createD3DReasourceForOutputs
    );

    // Upload Resource Handle and Tensor Flags for each operator
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();

    if (m_IsCustomAllocatorUsed) {
      switch (i) {
        case 0:  // Rope Query
          inputs[0]->SetResource(
            pTensorDescInput[0].d3dResource,
            pTensorDescInput[0].pCpuMappedD3DResc,
            pTensorDescInput[0].offsetInBytes, m_Context, false
          );
          inputs[1]->SetResource(
            pTensorDescInput[3].d3dResource,
            pTensorDescInput[3].pCpuMappedD3DResc,
            pTensorDescInput[3].offsetInBytes, m_Context, false
          );
          break;

        case 1:  // Rope Key
          inputs[0]->SetResource(
            pTensorDescInput[0].d3dResource,
            pTensorDescInput[0].pCpuMappedD3DResc,
            pTensorDescInput[0].offsetInBytes +
              pGqoOperator->GetQKVOffsetsVector()[0],
            m_Context, false
          );
          inputs[1]->SetResource(
            pTensorDescInput[3].d3dResource,
            pTensorDescInput[3].pCpuMappedD3DResc,
            pTensorDescInput[3].offsetInBytes, m_Context, false
          );
          break;

        case 2:  // GQA
          inputs[2]->SetResource(
            pTensorDescInput[0].d3dResource,
            pTensorDescInput[0].pCpuMappedD3DResc,
            pTensorDescInput[0].offsetInBytes +
              pGqoOperator->GetQKVOffsetsVector()[1],
            m_Context, false
          );
          inputs[3]->SetResource(
            pTensorDescInput[3].d3dResource,
            pTensorDescInput[3].pCpuMappedD3DResc,
            pTensorDescInput[3].offsetInBytes, m_Context, false
          );
          break;
      }
    }

    if (!createD3DReasourceForOutputs) {
      if (i == 2) {
        outputs[0]->SetResource(nullptr, nullptr, 0, m_Context, true);
        outputs[1]->SetResource(
          pTensorDescOutput[0].d3dResource,
          pTensorDescOutput[0].pCpuMappedD3DResc,
          pTensorDescOutput[0].offsetInBytes, m_Context, false
        );
        outputs[2]->SetResource(
          pTensorDescOutput[1].d3dResource,
          pTensorDescOutput[1].pCpuMappedD3DResc,
          pTensorDescOutput[1].offsetInBytes, m_Context, false
        );
      } else if (i == 3) {
        outputs[0]->SetResource(
          pTensorDescOutput[2].d3dResource,
          pTensorDescOutput[2].pCpuMappedD3DResc,
          pTensorDescOutput[2].offsetInBytes, m_Context, false
        );
      }
    }

    for (size_t j = 0; j < inputs.size(); j++) {
      if (inputs[j] != nullptr) {
        if (m_IsCustomAllocatorUsed &&
            constVector[j].isExternalBufferConstant) {
          inputs[j]->SetResource(
            constVector[j].d3dResource, constVector[j].pCpuMappedD3DResc,
            constVector[j].offsetInBytes, m_Context, false
          );
          inputs[j]->SetUploaded(constVector[j].isExternalBufferConstant);
        }
        inputs[j]->SetConstFlag(constVector[j].isExternalBufferConstant);
        inputs[j]->UpdateTensorBuffer(constVector[j].pExternalBuffer);
      }
    }
  }
  m_OpLists[opName]->UpdateNodeMappings();

  for (size_t i = 0; i < fusedOpsList.size(); i++) {
    // Initialize the input/output tensors for the fused operators
    fusedOpsList[i]->InitializeAndBindOperator(m_Context);
    // Uploads constants data to the GPU for each operator for each constant
    // Tensor vector
    m_Context->UploadConstData(
      fusedOpsList[i]->GetInputTensorVector(), m_IsCustomAllocatorUsed
    );
  }
}
#else  // GQO_GRAPH

//=====================================================================================================
void DMLOps::InitializeGQO(
  const std::string& opName,
  const std::vector<OnnxTensorInfo>& pTensorDescInput,
  const std::vector<OnnxTensorInfo>& pTensorDescOutput
) {
  bool createInputsInternally = m_IsCustomAllocatorUsed ? false : true;
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  pOperator->InitializeTensors(m_Context.get());

  TensorVector inputs = pOperator->GetInputTensorVector();
  TensorVector outputs = pOperator->GetOutputTensorVector();

  // For inputs 0, 4, 5, no need to set external buffer
  // copying of data for above resources happens in compute
  // Query
  SetTensorResource(
    inputs[0], pTensorDescInput[0], pTensorDescInput[0].offsetInBytes
  );

  // Position Id
  SetTensorResource(
    inputs[1], pTensorDescInput[3], pTensorDescInput[3].offsetInBytes
  );
  inputs[1]->UpdateTensorBuffer(pTensorDescInput[3].pExternalBuffer);

  // cos cache
  SetTensorResource(
    inputs[2], pTensorDescInput[5], pTensorDescInput[5].offsetInBytes
  );
  inputs[2]->UpdateTensorBuffer(pTensorDescInput[5].pExternalBuffer);

  // sin cache
  SetTensorResource(
    inputs[3], pTensorDescInput[6], pTensorDescInput[6].offsetInBytes
  );
  inputs[3]->UpdateTensorBuffer(pTensorDescInput[6].pExternalBuffer);

  // Key
  SetTensorResource(
    inputs[4], pTensorDescInput[0],
    pTensorDescInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[0]
  );

  // Value
  SetTensorResource(
    inputs[5], pTensorDescInput[0],
    pTensorDescInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[1]
  );

  // past sequence length
  uint32_t seqLen = *(uint32*)pTensorDescInput[3].pCpuMappedD3DResc;
  inputs[6]->UpdateResourceData(seqLen);

  // weights
  SetTensorResource(
    inputs[7], pTensorDescInput[7], pTensorDescInput[7].offsetInBytes
  );
  inputs[7]->UpdateTensorBuffer(pTensorDescInput[7].pExternalBuffer);

  // scale
  SetTensorResource(
    inputs[8], pTensorDescInput[8], pTensorDescInput[8].offsetInBytes
  );
  inputs[8]->UpdateTensorBuffer(pTensorDescInput[8].pExternalBuffer);

  // zero point
  SetTensorResource(
    inputs[9], pTensorDescInput[9], pTensorDescInput[9].offsetInBytes
  );
  inputs[9]->UpdateTensorBuffer(pTensorDescInput[9].pExternalBuffer);

  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      outputs[i]->SetResource(
        pTensorDescOutput[i].d3dResource,
        pTensorDescOutput[i].pCpuMappedD3DResc,
        pTensorDescOutput[i].offsetInBytes
      );
    }
  }

  pOperator->InitializeAndBindOperator(m_Context.get());
  m_Context->UploadConstData(inputs, m_IsCustomAllocatorUsed);
}

#endif  // GQO_GRAPH

#ifndef GQO_GRAPH
//=====================================================================================================
void DMLOps::ComputeGQOGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput
) {
  printMessage("Start: Executing GQO on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  // Offset for Rope Query
  void* pExtBufRopeQuery =
    reinterpret_cast<char*>(pBufInput[0].pExternalBuffer);
  // Offset for Rope Key
  void* pExtBufRopeKey = reinterpret_cast<char*>(pExtBufRopeQuery) +
                         pOperator->GetQKVOffsetsVector()[0];
  // Offset for Rope Value
  void* pExtBufRopeValue = reinterpret_cast<char*>(pExtBufRopeKey) +
                           pOperator->GetQKVOffsetsVector()[1];

  std::vector<std::shared_ptr<DmlOperator>> fusedOpsList =
    pOperator->GetFusedOperatorList();
  for (size_t i = 0; i < fusedOpsList.size(); i++) {
    try {
      TensorVector inputs = fusedOpsList[i]->GetInputTensorVector();
      TensorVector outputs = fusedOpsList[i]->GetOutputTensorVector();

      // Update the inputs/outputs with external data
      // If block in the loop should only run for inputs which are not constant.
      // Constant inputs should already been uploaded during the custom op
      // kernel's c'tor
#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
      const Clock::time_point upload_start = Clock::now();
#endif

      // For ropeQ:
      if (i == 0) {
        if (pBufInput[0].rebindD3DResc) {
          inputs[0]->SetResource(
            pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
            pBufInput[0].offsetInBytes
          );
          inputs[1]->SetResource(
            pBufInput[3].d3dResource, pBufInput[3].pCpuMappedD3DResc,
            pBufInput[3].offsetInBytes
          );
        }
        inputs[0]->UpdateTensorBuffer(pExtBufRopeQuery);
        inputs[1]->UpdateTensorBuffer(pBufInput[3].pExternalBuffer);
      }
      if (i == 1)  // ropeK
      {
        if (pBufInput[0].rebindD3DResc) {
          inputs[0]->SetResource(
            pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
            pBufInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[0]
          );
          inputs[1]->SetResource(
            pBufInput[3].d3dResource, pBufInput[3].pCpuMappedD3DResc,
            pBufInput[3].offsetInBytes
          );
        }
        inputs[0]->UpdateTensorBuffer(pExtBufRopeKey);
        inputs[1]->UpdateTensorBuffer(pBufInput[3].pExternalBuffer);
      }
      if (i == 2)  // GQA
      {
        if (pBufInput[0].rebindD3DResc) {
          inputs[2]->SetResource(
            pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
            pBufInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[1]
          );

          inputs[3]->SetResource(
            pBufInput[3].d3dResource, pBufInput[3].pCpuMappedD3DResc,
            pBufInput[3].offsetInBytes
          );
        }
        inputs[2]->UpdateTensorBuffer(pExtBufRopeValue);

        auto pastSeqLenBuffer = (int32*)pBufInput[3].pExternalBuffer;
        inputs[3]->UpdateTensorBuffer(
          (void*)pastSeqLenBuffer
        );  // input for past sequence length

        // Check if k cache resource is changed and update the resource
        // The resource is changed for every new prompt
        if (pBufOutput[0].rebindD3DResc) {
          // K cache
          outputs[1]->SetResource(
            pBufOutput[0].d3dResource, pBufOutput[0].pCpuMappedD3DResc,
            pBufOutput[0].offsetInBytes
          );

          // V cache
          outputs[2]->SetResource(
            pBufOutput[1].d3dResource, pBufOutput[1].pCpuMappedD3DResc,
            pBufOutput[1].offsetInBytes
          );

          // fusedOpsList[i]->RebindOutputs();
        }

        outputs[1]->UpdateTensorBuffer(pBufOutput[0].pExternalBuffer);
        outputs[2]->UpdateTensorBuffer(pBufOutput[1].pExternalBuffer);
      }
      if (i == 3) {  // matmul
        // Check if output resource is changed and update the resource
        // The resource is changed for every token
        if (pBufOutput[2].rebindD3DResc) {
          outputs[0]->SetResource(
            pBufOutput[2].d3dResource, pBufOutput[2].pCpuMappedD3DResc,
            pBufOutput[2].offsetInBytes
          );

          // fusedOpsList[i]->RebindOutputs();
        }
        outputs[0]->UpdateTensorBuffer(pBufOutput[2].pExternalBuffer);
      }

      fusedOpsList[i]->RebindInputAndOutput();

      // Upload the inputs and outputs to the GPU.
      m_Context->UploadInputData(inputs, m_IsCustomAllocatorUsed);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
      const Clock::time_point upload_end = Clock::now();
      const Duration upload_duration = upload_end - upload_start;

      m_OpLists[opName]->m_measurements[GPUEventID::GQO_UPLOAD_ID] +=
        upload_duration;
#endif
      // m_Context->DownloadData(outputs);

    } catch (const std::exception& exception) {
      std::cout << "GQO Exception: For internal node at index = " << i << " : "
                << exception.what() << std::endl;
    }
  }

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  const Clock::time_point execute_start = Clock::now();
#endif

  m_Context->ExecuteOperator(fusedOpsList);

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
  const Clock::time_point execute_end = Clock::now();
  const Duration execute_duration = execute_end - execute_start;

  m_OpLists[opName]->m_measurements[GPUEventID::GQO_EXECUTE_ID] +=
    execute_duration;

#endif

  printMessage("End: Executing GQO on GPU");
}
#else   // GQO_GRAPH
//=====================================================================================================
void DMLOps::ComputeGQOGPU(
  const std::string& opName, std::vector<OnnxTensorInfo>& pBufInput,
  std::vector<OnnxTensorInfo>& pBufOutput, bool rebindIO
) {
  printMessage("Start: Executing GQO on GPU");
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  TensorVector inputs = pOperator->GetInputTensorVector();
  TensorVector outputs = pOperator->GetOutputTensorVector();

  try {
    if (!m_IsCustomAllocatorUsed) {
      uint64_t totalBytesToCopy = inputs[0]->Desc().totalBytes +
                                  inputs[4]->Desc().totalBytes +
                                  inputs[5]->Desc().totalBytes;

      std::uint8_t* dst_base_ptr = (std::uint8_t*)inputs[0]->MappedResource();
      std::uint8_t* dst_ptr = dst_base_ptr + pBufInput[0].offsetInBytes;

      memcpy(dst_ptr, pBufInput[0].pExternalBuffer, totalBytesToCopy);
    }

    uint32_t seqLen = *(uint32_t*)pBufInput[3].pCpuMappedD3DResc;
    inputs[6]->UpdateResourceData(seqLen);

    if (rebindIO) {
      inputs[0]->SetResource(
        pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
        pBufInput[0].offsetInBytes
      );  // Query
      inputs[1]->SetResource(
        pBufInput[3].d3dResource, pBufInput[3].pCpuMappedD3DResc,
        pBufInput[3].offsetInBytes
      );  // Pos Id
      inputs[4]->SetResource(
        pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
        pBufInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[0]
      );  // Key
      inputs[5]->SetResource(
        pBufInput[0].d3dResource, pBufInput[0].pCpuMappedD3DResc,
        pBufInput[0].offsetInBytes + pOperator->GetQKVOffsetsVector()[1]
      );  // value
    }

    // Inputs 0, 4 and 5 use same memory with different offsets
    inputs[1]->UpdateTensorBuffer(
      pBufInput[3].pExternalBuffer
    );  // input for positionId
    // inputs 2 and 3 are cos_cache and sin_cache which have
    // already been uploaded during the custom op kernel's c'tor
    inputs[6]->UpdateTensorBuffer(
      pBufInput[3].pExternalBuffer
    );  // input for total sequence length
    // inputs 7, 8 and 9 are for matmul which have already been
    // uploaded during the custom op kernel's c'tor

    for (size_t i = 0; i < outputs.size(); i++) {
      if (rebindIO) {
        outputs[i]->SetResource(
          pBufOutput[i].d3dResource, pBufOutput[i].pCpuMappedD3DResc,
          pBufOutput[i].offsetInBytes
        );
      }
      outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }

    auto localWindow = pOperator->GetLocalWindowSize();

    // current sequence length after this token is generated
    int32_t currentSeqLen = seqLen + 1;
    bool useLocalWindow = (localWindow > -1) && (currentSeqLen > localWindow);
    if (useLocalWindow) {
      // Token start position for kv cache in the local window
      const uint32_t winStart =
        static_cast<uint32_t>(currentSeqLen) - localWindow;

      const auto& kOutCache = pOperator->GetOutputTensorVector().at(0);

      // offset in bytes for the attention computation
      uint64_t offsetInBytes =
        winStart * kOutCache->Desc().dims.back().size *
        kOutCache->ElementSize(kOutCache->Desc().dataType);

      // size in bytes for the attention computation
      uint64_t attnSizeBytes =
        localWindow * kOutCache->Desc().dims.back().size *
        kOutCache->ElementSize(kOutCache->Desc().dataType);

      // Set the offset and size of KV buffers for attention computation
      outputs[0]->SetAttnOffsetAndSize(offsetInBytes, attnSizeBytes);
      outputs[1]->SetAttnOffsetAndSize(offsetInBytes, attnSizeBytes);

      // Update the past sequence length input for next token
      inputs[6]->UpdateResourceData(localWindow - 1);
      rebindIO = true;
    }

    UpdateBindings(opName, pBufInput, pBufOutput, rebindIO, false);

    // Upload the inputs and outputs to the GPU.
    m_Context->UploadInputOrConstData(inputs, m_IsCustomAllocatorUsed);

    RecordOrExecuteOperator(pOperator, opName);
    printMessage("End: Executing GQO on GPU");
  } catch (const std::exception& exception) {
    std::cout << "GQO Exception: " << exception.what() << std::endl;
  }
}
#endif  // GQO_GRAPH

// =====================================================================================================
#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
const std::map<int, Duration>& DMLOps::GetPerfData(
  const std::string& opName
) const {
  return m_OpLists.at(opName)->m_measurements;
}
#endif

}  // namespace DML_Ops

}  // namespace ryzenai::onnx_utils
