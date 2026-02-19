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

  const auto K = input_0_dim.back();
  const auto M = input_0_dim.at(input_0_dim.size() - 2);

  const auto in_dtype = input_0.GetTensorTypeAndShapeInfo().GetElementType();
  const auto datum_size =
    (in_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? 2 : 4;

  if ((in_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
      (in_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
    throw std::runtime_error(
      "unsupported input dtype for matmul, expect float16 or float32"
    );
  }

  const bool input_cast = (in_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  const std::uint8_t* input_data_ptr = input_0.GetTensorData<std::uint8_t>();

  const float* wts_data_ptr = input_1.GetTensorData<float>();

  if (run_prune_logits) {
    input_0_dim.at(input_0_dim.size() - 2) = 1;
    // assume output is sized for pruned output
    // need to slice input tensor and pass last row to op
    input_data_ptr = &input_data_ptr[(M - 1) * K * datum_size];
  }

  auto output_dim = input_0_dim;
  // replace K by N
  output_dim.back() = input_1_dim.back();

  auto input_num_elems = std::accumulate(
    input_0_dim.begin(), input_0_dim.end(), 1ULL, std::multiplies<>()
  );

  auto output_0 = ctx.GetOutput(0, output_dim);

  const auto out_dtype = output_0.GetTensorTypeAndShapeInfo().GetElementType();

  bool output_cast = (out_dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto output_num_elems = std::accumulate(
    output_dim.begin(), output_dim.end(), 1ULL, std::multiplies<>()
  );

  std::uint8_t* out_data_ptr = output_0.GetTensorMutableData<std::uint8_t>();

  // need to implement/verify NPU/GPU path
  auto backend = Backend::Cpu;  // getBackend(ctx);

  switch (backend) {
    case Backend::Cpu: {
      std::vector<float> input_tmp;
      std::vector<float> output_tmp;

      float* matmul_in_ptr =
        (float*)(const_cast<std::uint8_t*>(input_data_ptr));
      float* matmul_output_ptr = (float*)out_data_ptr;

      if (input_cast) {
        input_tmp.resize(input_num_elems);
        matmul_in_ptr = input_tmp.data();

        ort_cast_fp16_to_fp32_->execute(
          matmul_in_ptr,
          (Ort::Float16_t*)(const_cast<std::uint8_t*>(input_data_ptr)),
          input_0_dim, context
        );
      }

      if (output_cast) {
        output_tmp.resize(output_num_elems);
        matmul_output_ptr = output_tmp.data();
      }

      ort_matmul_->execute(
        context, matmul_in_ptr, input_0_dim, const_cast<float*>(wts_data_ptr),
        input_1_dim, matmul_output_ptr, output_dim
      );

      if (output_cast) {
        ort_cast_fp32_to_fp16_->execute(
          (Ort::Float16_t*)out_data_ptr, matmul_output_ptr, output_dim, context
        );
      }

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
