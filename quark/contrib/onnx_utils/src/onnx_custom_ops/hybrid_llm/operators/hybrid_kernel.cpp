// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "hybrid_kernel.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#include "opUtils.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

namespace ryzenai::onnx_utils {

HybridKernel::HybridKernel(
  [[maybe_unused]] const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : kernel_info_(info) {
  node_name_ = kernel_info_.GetNodeName();

#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  with_custom_allocator_ = session_configs.count("custom_allocator") &&
                           session_configs.at("custom_allocator").size();
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY

  setTokenBackend(session_configs);
  setDisableNpuOps(session_configs);

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  // This minimal initialization must be kept at least for matmul that might
  // perform CAST operation on GPU
  try {
    CComPtr<ID3D12Device3> d3d_device;

    // since it's always built with RMM why don't we use main d3d device from
    // RMM uncoditionally
    d3d_device.Attach(RyzenMM::Platform::DX::GetCurrentD3D12Device3());

    dml_instance_ = DML_Ops::DMLOps::getInstance(session_configs, d3d_device);
  } catch (const std::exception& e) {
    std::cerr << "Exception caught in DML Instance constructor: " << e.what()
              << std::endl;
  }
#endif
}

void HybridKernel::setTokenBackend(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_token_backend");
  if (!value.empty()) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
    if (value == "npu") {
      token_backend_ = Backend::Npu;
      return;
    }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
    if (value == "gpu") {
      token_backend_ = Backend::Gpu;
      return;
    }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
    throw std::invalid_argument(
      "Unknown or unsupported token backend: " + value
    );
  }
}

const std::string& HybridKernel::nodeName() const { return node_name_; }

bool HybridKernel::withCustomAllocator() const {
  return with_custom_allocator_;
}

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

DML_Ops::DMLOps* HybridKernel::dmlInstance() const {
  return dml_instance_.get();
}

std::pair<std::vector<OnnxTensorInfo>&, std::vector<OnnxTensorInfo>&>
HybridKernel::gpuTensors() {
  return {tensor_inputs_, tensor_outputs_};
}

void HybridKernel::createGpu(
  const std::unordered_map<std::string, std::string>& session_configs,
  size_t gpu_input_num, int op_chaining_input_idx, int op_chaining_output_idx
) {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  try {
    dml_instance_->SetFirstAndLastNodeForJit(node_name_);

    uint32_t idx = 0;
    size_t found_inputs = 0;
    int i = -1;
    while (found_inputs < gpu_input_num) {
      i++;
      OnnxTensorInfo tensorInfo;
      tensorInfo.name = kernel_info_.GetInputName(i);
      if (tensorInfo.name.empty()) {
        continue;
      }

      found_inputs++;

      tensorInfo.index = i;
      auto type = kernel_info_.GetInputTypeInfo(i)
                    .GetTensorTypeAndShapeInfo()
                    .GetElementType();
      tensorInfo.dataType = ElementFormat(type);

      int isConstant = 0;
      Ort::ConstValue val = kernel_info_.GetTensorConstantInput(i, &isConstant);
      tensorInfo.isExternalBufferConstant = static_cast<bool>(isConstant);

      auto shape =
        kernel_info_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

      if (val) {
        // if custom allocator enabling session option was not set but RMM was
        // actually used to allocate memory for tensors, lets set the option
        with_custom_allocator_ |=
          RyzenMM::IsKnown(RyzenMM::UnmanagedBufferPtr(val.GetTensorRawData()));
      }

      if (isConstant) {
        if (dml_instance_->IsJitWtsLoaderEnabled() && (shape[0] == 0)) {
          bool success = dml_instance_->InitializeJitTensorInfo(
            node_name_, idx, tensorInfo, GetPackedTensorSize(type, 1)
          );
          if (!success) {
            continue;
          }
          idx++;
        } else {
          tensorInfo.shape = shape.empty() ? std::vector<int64_t>{1} : shape;
          if (tensorInfo.shape[0] == 0) continue;
          if (RyzenMM::IsKnown(
                RyzenMM::UnmanagedBufferPtr(val.GetTensorRawData())
              )) {
            tensorInfo.UseRMMAllocatedMemory(val.GetTensorRawData());
          } else {
            // Set the D3D resource and CPU mapped memory for the constant
            // tensor
            int64 elementCount =
              val.GetTensorTypeAndShapeInfo().GetElementCount();
            tensorInfo.UseRMMBuffer(gpuAllocator_.AllocateBuffer(
              GetPackedTensorSize(type, elementCount)
            ));
          }
          tensorInfo.pExternalBuffer =
            const_cast<void*>(val.GetTensorRawData());
        }
      } else {
        // for integer inputs, there's no shape data
        tensorInfo.shape = shape.empty() ? std::vector<int64_t>{1} : shape;
        if (tensorInfo.shape[0] == 0) continue;
      }

      // check for batch and sequence which are -1
      for (int x = 0; x < tensorInfo.shape.size(); x++) {
        tensorInfo.shape[x] =
          tensorInfo.shape[x] == -1 ? 1 : tensorInfo.shape[x];
      }

      /*if (tensorInfo.isExternalBufferConstant) {
          std::cout << node_name_ << " D3D resource : " <<
      tensorInfo.pCpuMappedD3DResc << "  data offset : " <<
      tensorInfo.offsetInBytes << std::endl;
      }*/
      tensor_inputs_.push_back(tensorInfo);
    }

    size_t outputCount = kernel_info_.GetOutputCount();
    for (size_t i = 0; i < outputCount; i++) {
      OnnxTensorInfo tensorInfo;
      std::string outputName = kernel_info_.GetOutputName(i);
      if (outputName.empty()) {
        printLog(
          " GPU Path: For ", node_name_, " Output[", i,
          "] : is invalid. Onnx includes this output in total num "
          "of outputs"
        );
        continue;
      }
      tensorInfo.index = i;
      auto shape = kernel_info_.GetOutputTypeInfo(i)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
      auto type = ElementFormat(kernel_info_.GetOutputTypeInfo(i)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetElementType());
      tensorInfo.name = outputName;
      tensorInfo.dataType = type;
      tensorInfo.shape = shape;
      // we should get this during the time Compute gets called.
      tensorInfo.pExternalBuffer = nullptr;
      tensorInfo.isExternalBufferConstant = false;
      // check for batch and sequence which are -1
      for (int x = 0; x < tensorInfo.shape.size(); x++) {
        tensorInfo.shape[x] =
          tensorInfo.shape[x] == -1 ? 1 : tensorInfo.shape[x];
      }
      tensor_outputs_.push_back(tensorInfo);
    }

    // Match previous op's output with current op's input and save current op's
    // output
    dml_instance_->CheckForOpChaining(
      tensor_inputs_[op_chaining_input_idx].name,
      tensor_outputs_[op_chaining_output_idx].name, node_name_
    );

  } catch (const std::exception& e) {
    std::cerr << "Exception caught in " + node_name_ + " constructor: "
              << e.what() << std::endl;
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
}

void HybridKernel::initializeGpu() {
  try {
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
    // Allocate D3D resource and save the cpu handle in a vector to be used by
    // NPU
    if (!with_custom_allocator_) {
      for (size_t i = 0; i < tensor_inputs_.size(); i++) {
        auto shape = kernel_info_.GetInputTypeInfo(i)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape();
        if (shape[0] == 0) continue;

        if (tensor_inputs_[i].isConstForJit) {
          continue;
        }

        int64 resourceSize =
          dml_instance_->getResourceSizePerInput(node_name_, i);
        tensor_inputs_[i].UseRMMBuffer(
          gpuAllocator_.AllocateBuffer(resourceSize)
        );
      }

      for (size_t i = 0; i < tensor_outputs_.size(); i++) {
        int64 resourceSize =
          dml_instance_->getResourceSizePerOutput(node_name_, i);
        tensor_outputs_[i].UseRMMBuffer(
          gpuAllocator_.AllocateBuffer(resourceSize)
        );
      }
    }
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY

  } catch (const std::exception& e) {
    std::cerr << "Exception caught in " + node_name_ + " constructor: "
              << e.what() << std::endl;
  }
}

bool HybridKernel::computeGpu(const Ort::KernelContext& ctx) {
  dml_instance_->DynamicLoadWeightsForJit();

  bool reBindD3DResc = false;
  for (size_t i = 0; i < tensor_inputs_.size(); i++) {
    auto index = tensor_inputs_[i].index;
    auto shape = ctx.GetInput(index).GetTensorTypeAndShapeInfo().GetShape();
    if (!shape.empty() && shape[0] == 0) continue;
#ifdef DEBUG
    auto type = ElementFormat(
      ctx.GetInput(index).GetTensorTypeAndShapeInfo().GetElementType()
    );

    // This info should be already present during the c'tor's call
    assert(tensor_inputs_[i].dataType == type);
    // This info should be already present during the c'tor's call
    assert(tensor_inputs_[i].shape == shape);
#endif

    Ort::ConstValue val = ctx.GetInput(index);
    auto newMappedD3dResc = const_cast<void*>(val.GetTensorRawData());

    if (false == tensor_inputs_[i].isExternalBufferConstant) {
      if (with_custom_allocator_) {
        if (tensor_inputs_[i].pCpuMappedD3DResc &&
            (tensor_inputs_[i].pCpuMappedD3DResc != newMappedD3dResc)) {
          reBindD3DResc = true;
        }
        tensor_inputs_[i].UseRMMAllocatedMemory(val.GetTensorRawData());
      } else {
        // this the case of input activations as Constant Tensor's pointer
        // should be already there during c'tor's call
        tensor_inputs_[i].pExternalBuffer =
          const_cast<void*>(val.GetTensorRawData());
      }
    }
  }
  for (size_t i = 0; i < tensor_outputs_.size(); i++) {
    Ort::UnownedValue val = ctx.GetOutput(i, tensor_outputs_[i].shape);
    if (val == nullptr) {
      // in the case of skipsimplifiedlayernorm onnx node, output count is 4
      // but only output at index 0 and 3 are present and at index 1 and 2
      // its null
      continue;
    }
    auto* val_data = val.GetTensorMutableRawData();
    if (with_custom_allocator_) {
      if (tensor_outputs_[i].pCpuMappedD3DResc &&
          (tensor_outputs_[i].pCpuMappedD3DResc != val_data)) {
        reBindD3DResc = true;
      }
      tensor_outputs_[i].UseRMMAllocatedMemory(val_data);
    } else {
      tensor_outputs_[i].pExternalBuffer = val_data;
    }
  }

  return reBindD3DResc;
}

#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

Backend HybridKernel::getBackend(const Ort::KernelContext& ctx) {
  auto input_0 = ctx.GetInput(0);  // activations - fp16/32
  auto dimensions = input_0.GetTensorTypeAndShapeInfo().GetShape();

  if (const auto is_token_phase = dimensions.at(1) == 1; is_token_phase) {
    return token_backend_;
  }
  return prefill_backend_;
}

void HybridKernel::releaseGpuJitWeights() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (node_name_ == dml_instance_->GetFirstNodeForJit()) {
    dml_instance_->ReleaseWeightsForJit();
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
}

void HybridKernel::setDisableNpuOps(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_disable_npu_ops");
  disable_npu_ops_ = value.empty() ? false : value == "1";
}
bool HybridKernel::disableNpuOps() const { return disable_npu_ops_; }

}  // namespace ryzenai::onnx_utils
