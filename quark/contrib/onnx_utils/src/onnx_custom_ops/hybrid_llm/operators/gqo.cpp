// Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "gqo.hpp"

// #include <shared_memory/d3d_xrt_allocator.h>

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "opUtils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
#include "../npu/gqo.hpp"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

GQOKernel::GQOKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  printLog("Constructing GQOKernel custom op...");

  Ort::ConstKernelInfo kernel_info{info};

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

  printLog(
    "GQOKernel custom op attributes:\n", "Input count: ", input_count, "\n",
    "Output count: ", output_count
  );

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    try {
      AttributeForGQO attrForGQO;
      attrForGQO.do_rotary = getAttribute<int64>("do_rotary");
      attrForGQO.kv_num_heads = getAttribute<int64>("kv_num_heads");
      attrForGQO.num_heads = getAttribute<int64>("num_heads");
      attrForGQO.o_proj_bits = getAttribute<int64>("o_proj_bits");
      attrForGQO.o_proj_block_size = getAttribute<int64>("o_proj_block_size");
      attrForGQO.o_proj_K = getAttribute<int64>("o_proj_K");
      attrForGQO.o_proj_N = getAttribute<int64>("o_proj_N");
      attrForGQO.rotary_interleaved = getAttribute<int64>("rotary_interleaved");
      attrForGQO.scale = getAttribute<float>("scale");

      attrForGQO.rotary_embedding_dim =
        getAttribute<int64_t>("rotary_embedding_dim", 0);
      attrForGQO.head_size = getAttribute<int64_t>("head_size", 0);
      attrForGQO.local_window_size =
        getAttribute<int64_t>("local_window_size", -1);
      attrForGQO.softcap = getAttribute<float>("softcap", 0.0f);

      // TODO: need update to support input schema with head_sink
      auto gpu_input_count = input_count > 10 ? 10 : input_count;

      createGpu(session_configs, gpu_input_count, 0, 2);

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      mapped_output_tensors_.resize(tensor_outputs.size());

      printLog(
        " GPU Path: Constructing ", nodeName(), " of type GQO Custom op\n "
      );

      dml_instance->CreateGQOOp(nodeName(), tensor_inputs, attrForGQO);
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in GQO constructor: " << e.what()
                << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  // std::cout << "NPU Path: GQOKernel custom op\n";
  if (!disableNpuOps()) {
    npu_instance_ =
      std::make_unique<AMDGQOKernel>(info, ort_api, session_configs);
  }
  // npu_instance_->initialize_shared_data(cpuInputMappedMemory,
  //                                       cpuOutputMappedMemory);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  printLog("Constructed ", nodeName(), " of type GQOKernel Custom op\n\n ");
}

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
bool GQOKernel::RebuildSharedKVCache(const Ort::KernelContext& ctx) {
  if (!shared_k_cache_) return true;

  size_t kvCacheSize =
    ctx.GetInput(3).GetTensorTypeAndShapeInfo().GetElementCount() *
    sizeof(Ort::Float16_t);
  if (mapped_output_tensors_[0].second >= kvCacheSize) {
    return false;
  } else {
    shared_k_cache_ = {};
    shared_v_cache_ = {};
    isKVShapeChanged_ = true;
  }
  return true;
}

void GQOKernel::CreateSharedKVCache(size_t resourceSize) {
  shared_k_cache_ = gpuAllocator_.AllocateBuffer(resourceSize);
  shared_v_cache_ = gpuAllocator_.AllocateBuffer(resourceSize);
}

std::pair<void*, void*> GQOKernel::GetCpuMappedMemSharedKVCache() const {
  return std::make_pair(shared_k_cache_.Data(), shared_v_cache_.Data());
}

std::pair<void*, void*> GQOKernel::GetD3DResourceSharedKVCache() const {
  auto k = RyzenMM::Platform::DX::GetUnderlyingD3D12Resource(shared_k_cache_);
  auto v = RyzenMM::Platform::DX::GetUnderlyingD3D12Resource(shared_v_cache_);

  k->Release();
  v->Release();

  return std::make_pair(k, v);
}
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

GQOKernel::~GQOKernel() {
#if defined(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU) && \
  defined(ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING)
  std::ostringstream os;

  for (const auto& [event_id, duration] :
       dml_instance->GetPerfData(nodeName())) {
    os << nodeName() << ",GPUGQO," << event_id << ","
       << MillisecondsFp{duration}.count() << "\n";
  }

  os << nodeName() << ",GPUGQO," << GPUEventID::GQO_DOWNLOAD_ID << ","
     << MillisecondsFp{download_duration_}.count() << "\n";

  std::cout << os.str() << std::flush;
#endif
}

void GQOKernel::Compute(OrtKernelContext* context) {
  // std::cout << "Executing " << nodeName()
  //           << " of type GQOKernel Custom op\n\n";

  Ort::KernelContext ctx(context);
  /*const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();*/

  // For now stub out cpu fallback as this custom op
  // assumes input is float16, and current NPU implementation
  // always does data conversion of float16 ->bfloat16
  // NOTE: eventually do fallback by setting to !gpu_path && !npu_path

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  auto& [tensor_inputs, tensor_outputs] = gpuTensors();

  if (!withCustomAllocator()) {
    if (RebuildSharedKVCache(ctx)) {
      auto past_k = ctx.GetInput(3);
      auto past_v = ctx.GetInput(4);
      auto elems_k = past_k.GetTensorTypeAndShapeInfo().GetElementCount();
      size_t kvCacheSize = elems_k * sizeof(Ort::Float16_t);
      CreateSharedKVCache(kvCacheSize);
      auto [k_cache, v_cache] = GetCpuMappedMemSharedKVCache();
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      npu_instance_->set_kv_cache(k_cache, v_cache);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

      mapped_output_tensors_[0] = std::make_pair(k_cache, kvCacheSize);
      mapped_output_tensors_[1] = std::make_pair(v_cache, kvCacheSize);
      // NOTE : Type should be generic, Allocating attention_score output
      auto [d3d_shared_k, d3d_shared_v] = GetD3DResourceSharedKVCache();
      tensor_outputs[0].d3dResource =
        static_cast<ID3D12Resource*>(d3d_shared_k);
      tensor_outputs[0].pCpuMappedD3DResc = k_cache;
      tensor_outputs[1].d3dResource =
        static_cast<ID3D12Resource*>(d3d_shared_v);
      tensor_outputs[1].pCpuMappedD3DResc = v_cache;

      if (!isKVShapeChanged_) {
        int64 resourceSize =
          tensor_outputs[2].shape[2] * sizeof(Ort::Float16_t);

        const auto data =
          tensor_outputs[2]
            .UseRMMBuffer(gpuAllocator_.AllocateBuffer(resourceSize))
            .Data();

        mapped_output_tensors_[2] = std::make_pair(data, resourceSize);
      }
    }
  }

#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      printLog("GPU Path for: ", nodeName(), " of type GQO Custom op");

      auto* dml_instance = dmlInstance();
      dml_instance->DynamicLoadWeightsForJit();

      tensor_inputs[1].shape =
        ctx.GetInput(3).GetTensorTypeAndShapeInfo().GetShape();  // past key
      tensor_inputs[2].shape =
        ctx.GetInput(4).GetTensorTypeAndShapeInfo().GetShape();  // past value

      // Equate output shape with past key vlaue to match the diemnsions
      // TO DO : check if there's a better way to equate the shape

      if (!isFirstRun_) {
        for (uint32_t i = 0; i < tensor_outputs[0].shape.size(); i++) {
          if (tensor_outputs[0].shape[i] != tensor_inputs[1].shape[i]) {
            isKVShapeChanged_ = true;
            break;
          }
        }
      }
      tensor_outputs[0].shape = tensor_inputs[1].shape;
      tensor_outputs[1].shape = tensor_inputs[2].shape;

      for (size_t i = 0; i < tensor_inputs.size(); i++) {
        if (!tensor_inputs[i].isExternalBufferConstant) {
          Ort::ConstValue val = ctx.GetInput(tensor_inputs[i].index);
          // this the case of input activations as Constant Tensor's pointer
          // should be already there during c'tor's call
          if (!withCustomAllocator()) {
            if (isFirstRun_) {
              // Set the D3D resource and CPU mapped memory for the activation
              // tensors In case of past key and past value, we need to skip as
              // DML doesn't use them
              if (i == 1 || i == 2) {
                continue;
              }

              int64 elementCount =
                val.GetTensorTypeAndShapeInfo().GetElementCount();
              tensor_outputs[i].UseRMMBuffer(
                gpuAllocator_.AllocateBuffer(GetPackedTensorSize(
                  val.GetTensorTypeAndShapeInfo().GetElementType(), elementCount
                ))
              );
            }
          }
        }
      }
      isFirstRun_ = false;

      bool rebindD3DResc = computeGpu(ctx);

      dml_instance->GQOOpDynamicInitialization(
        nodeName(), tensor_inputs, tensor_outputs, isKVShapeChanged_
      );
      // When KV cache shape changes, we need to reinitialize the op with new
      // size
      isKVShapeChanged_ = false;

      dml_instance->ComputeGQOGPU(
        nodeName(), tensor_inputs, tensor_outputs, rebindD3DResc
      );
#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
      const Clock::time_point copy_start = Clock::now();
#endif

      if (!withCustomAllocator()) {
        // copy only matmulnbit output to onnx. No need for kv cache to be
        // copied
        memcpy(
          tensor_outputs[2].pExternalBuffer, mapped_output_tensors_[2].first,
          mapped_output_tensors_[2].second
        );
      }

#ifdef ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING
      const Clock::time_point copy_end = Clock::now();
      download_duration_ += (copy_end - copy_start);
#endif
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      releaseGpuJitWeights();
      // std::cout << "NPU Path: Compute function\n";
      npu_instance_->Compute(context, withCustomAllocator());
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }

  // std::cout << "\nExecution done for: " << nodeName()
  //           << " of type GQOKernel Custom op\n\n";
}

}  // namespace ryzenai::onnx_utils
