// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "matmulnbits.hpp"

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
#if defined(ONNX_UTILS_ENABLE_CORELIB)
#include "corelib_kernel.hpp"
#else
#include "../npu/matmulnbits.hpp"
#endif
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

MatMulNBitsKernel::MatMulNBitsKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  // printLog("Constructing MatMulNBitsKernel custom op...");

  Ort::ConstKernelInfo kernel_info{info};

  // this is an optional attribute according to com.microsoft v1
  accuracy_level_ = getAttribute<int64_t>("accuracy_level", 0);
  bits_ = getAttribute<int64_t>("bits");
  block_size_ = getAttribute<int64_t>("block_size");
  k_ = getAttribute<int64_t>("K");
  n_ = getAttribute<int64_t>("N");

  const auto input_count = kernel_info.GetInputCount();
  const auto output_count = kernel_info.GetOutputCount();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  const std::string domain = "com.microsoft";
  constexpr auto version = 1;
  constexpr auto type_constraint_count = 4;

  std::array<const char*, type_constraint_count> type_constraint_names = {
    "T1", "T2", "T3", "T4"
  };
  std::array<ONNXTensorElementDataType, type_constraint_count>
    type_constraint_values = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    };

  constexpr size_t attr_count = 5;
  std::vector<Ort::OpAttr> attrs;
  // no default constructor for OpAttr
  attrs.reserve(attr_count);
  attrs.emplace_back(
    "accuracy_level", &accuracy_level_, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  attrs.emplace_back("bits", &bits_, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  attrs.emplace_back(
    "block_size", &block_size_, 1, OrtOpAttrType::ORT_OP_ATTR_INT
  );
  attrs.emplace_back("K", &k_, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
  attrs.emplace_back("N", &n_, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  cpu_op_ = Ort::Op::Create(
    info, "MatMulNBits", domain.c_str(), version, type_constraint_names.data(),
    type_constraint_values.data(), type_constraint_count, attrs.data(),
    attr_count, input_count, output_count
  );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    try {
      const size_t gpu_input_num = input_count == 6 ? 5 : input_count;
      createGpu(session_configs, gpu_input_num, 0, 0);

      printLog(
        " GPU Path: Constructing ", nodeName(),
        " of type MatMulNBitsKernel Custom op"
      );

      auto* dml_instance = dmlInstance();
      const auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      dml_instance->CreateMatMulNBitsOperator(
        nodeName(), tensor_inputs, tensor_outputs, block_size_
      );

      initializeGpu();
      if (!withCustomAllocator() && !dml_instance->IsJitWtsLoaderEnabled()) {
        // set D3D resources and upload consts / weights
        dml_instance->InitializeMatMulNBits(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception caught in MatMulNBitsKernel constructor: "
                << e.what() << std::endl;
    }
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  if (!disableNpuOps()) {
    printLog("NPU Path: MatMulKernel custom op");
#if defined(ONNX_UTILS_ENABLE_CORELIB)
    npu_instance_ = std::make_unique<CoreLibKernel>(ort_api, info, "MatMul");
#else
    npu_instance_ =
      std::make_unique<AMDMatMulNBitsKernel>(info, session_configs);
#endif
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU

  printLog(
    "Constructed ", nodeName(), " of type MatMulNBitsKernel Custom op\n "
  );
}
MatMulNBitsKernel::~MatMulNBitsKernel() {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  cpu_op_.release();
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU

#if defined(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU) && \
  defined(ONNX_UTILS_ENABLE_CUSTOM_OP_GPU_PROFILING)
  std::ostringstream os;

  for (const auto& [event_id, duration] :
       dml_instance->GetPerfData(nodeName())) {
    os << nodeName() << ",GPUMatmulnbits," << event_id << ","
       << MillisecondsFp{duration}.count() << "\n";
  }

  std::cout << os.str() << std::flush;
#endif
}

void MatMulNBitsKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  auto gpu_input_num = input_num == 6 ? 5 : input_num;

  auto input_0 = ctx.GetInput(0);  // activations - fp16/32
  auto dimensions = input_0.GetTensorTypeAndShapeInfo().GetShape();

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  const auto output_num = ctx.GetOutputCount();

  auto input_1 = ctx.GetInput(1);  // weights - int4
  auto scales = ctx.GetInput(2);   // scales - fp16/32
  Ort::ConstValue zero_points;
  if (gpu_input_num > 3) {
    zero_points = ctx.GetInput(3);  // zero points - uint8/int32/fp16/fp32
  }

  Ort::ConstValue bias;
  if (gpu_input_num > 4) {
    // bias to add to result, should match activation (fp16/32)
    bias = ctx.GetInput(4);
  }
  // this will be [batch_size, sequence_length, K (op attribute)]
  dimensions.at(2) = n_;

  // output dim should be [batch_size, sequence_length, N (op attribute)]
  auto output_0 = ctx.GetOutput(0, dimensions);
  std::vector<const OrtValue*> inputs = {input_0, input_1, scales};
  if (gpu_input_num > 3) {
    inputs.push_back(zero_points);
  }

  // g_idx input has to be set to nullptr. For more details refer documentation
  // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits
  inputs.push_back(nullptr);

  if (gpu_input_num > 4) {
    inputs.push_back(bias);
  }
  OrtValue* outputs[1] = {output_0};
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  // For now stub out cpu fallback as this custom op
  // assumes input is float16, and current NPU implementation
  // always does data conversion of float16 ->bfloat16
  // NOTE: eventually do fallback by setting to !gpu_path && !npu_path

  auto backend = getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      cpu_op_.Invoke(context, inputs.data(), input_num, outputs, output_num);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
      break;
    }
    case Backend::Gpu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();

      if (withCustomAllocator() || dml_instance->IsJitWtsLoaderEnabled()) {
        dml_instance->InitializeMatMulNBits(
          nodeName(), tensor_inputs, tensor_outputs
        );
      }
      dml_instance->ComputeMatMulNBitsGPU(
        nodeName(), tensor_inputs, tensor_outputs, reBindD3DResc
      );
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      releaseGpuJitWeights();
#if defined(ONNX_UTILS_ENABLE_CORELIB)
      npu_instance_->Compute(context);
#else
      npu_instance_->Compute(context, withCustomAllocator());
#endif
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
      break;
    }
  }
}

}  // namespace ryzenai::onnx_utils
