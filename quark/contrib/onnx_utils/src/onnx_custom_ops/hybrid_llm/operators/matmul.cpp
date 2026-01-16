// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "matmul.hpp"

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../operators/opUtils.h"
#include "../ort/matmul.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

MatMulKernel::MatMulKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  Ort::ConstKernelInfo kernel_info{info};

  // run matmul only on last row
  // optimization for inference when running lm head
  prune_en_ = getAttribute<int64_t>("prune", 0);

  ort_matmul_ = std::make_unique<OrtMatMul>();
  ort_matmul_->construct(kernel_info);

  ort_cast_fp16_to_fp32_ = std::make_unique<OrtCast<Ort::Float16_t, float>>();
  ort_cast_fp32_to_fp16_ = std::make_unique<OrtCast<float, Ort::Float16_t>>();

  ort_cast_fp16_to_fp32_->construct(kernel_info);
  ort_cast_fp32_to_fp16_->construct(kernel_info);

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if (usingGpu()) {
    auto dml_instance = DML_Ops::DMLOps::getInstance(session_configs);
    const auto& [tensor_inputs, tensor_outputs] = gpuTensors();

    size_t input_count = kernel_info.GetInputCount();
    createGpu(session_configs, input_count, 0, 0);

    dml_instance->CreateMatMulOperator(
      nodeName(), tensor_inputs, tensor_outputs
    );
  }

#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
}

MatMulKernel::~MatMulKernel() {}

void MatMulKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  const auto input_num = ctx.GetInputCount();
  const auto output_num = ctx.GetOutputCount();

  auto input_0 = ctx.GetInput(0);
  auto input_1 = ctx.GetInput(1);

  auto input_0_dim = input_0.GetTensorTypeAndShapeInfo().GetShape();
  auto input_1_dim = input_1.GetTensorTypeAndShapeInfo().GetShape();

  const bool run_prune_logits = prune_en_ && (input_0_dim.at(0) == 1);

  auto K = input_0_dim.back();
  auto M = input_0_dim.at(input_0_dim.size() - 2);

  const Ort::Float16_t* input_data_ptr =
    input_0.GetTensorData<Ort::Float16_t>();
  const float* wts_data_ptr = input_1.GetTensorData<float>();

  if (run_prune_logits) {
    input_0_dim.at(input_0_dim.size() - 2) = 1;
    // assume output is sized for pruned output
    // need to slice input tensor and pass last row to op
    input_data_ptr = &input_data_ptr[(M - 1) * K];
  }

  auto output_dim = input_0_dim;
  // replace K by N
  output_dim.back() = input_1_dim.back();

  auto input_num_elems = std::accumulate(
    input_0_dim.begin(), input_0_dim.end(), 1ULL, std::multiplies<>()
  );

  auto output_0 = ctx.GetOutput(0, output_dim);

  auto output_num_elems = std::accumulate(
    output_dim.begin(), output_dim.end(), 1ULL, std::multiplies<>()
  );

  Ort::Float16_t* out_data_ptr =
    output_0.GetTensorMutableData<Ort::Float16_t>();

  // need to implement/verfiy NPU/GPU path
  auto backend = Backend::Cpu;  // getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
      std::vector<float> input_tmp(input_num_elems);
      std::vector<float> output_tmp(output_num_elems);

      ort_cast_fp16_to_fp32_->execute(
        input_tmp.data(), const_cast<Ort::Float16_t*>(input_data_ptr),
        input_0_dim, context
      );
      ort_matmul_->execute(
        context, input_tmp.data(), input_0_dim,
        const_cast<float*>(wts_data_ptr), input_1_dim, output_tmp.data(),
        output_dim
      );
      ort_cast_fp32_to_fp16_->execute(
        out_data_ptr, output_tmp.data(), output_dim, context
      );
      break;
    }
    case Backend::Gpu: {
      throw std::runtime_error("need to verify");
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      std::cout << "GPU Path: Compute function\n";

      auto reBindD3DResc = computeGpu(ctx);

      auto* dml_instance = dmlInstance();
      auto& [tensor_inputs, tensor_outputs] = gpuTensors();
      dml_instance->ComputeMatMulGPU(nodeName(), tensor_inputs, tensor_outputs);
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
      break;
    }
    case Backend::Npu: {
      throw std::runtime_error("not implemented!");
      break;
    }
  }
}

}  // namespace ryzenai::onnx_utils
