//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_in.h"
#include "in/cpu/instance_norm.h"

#include <vector>
#include <cmath>

#include "core/framework/float16.h"

namespace vai_q {

static void PrepareForBF16(const Ort::ConstValue& data,
                           std::vector<float>& buf) {
    ONNXTensorElementDataType data_type = data.GetTensorTypeAndShapeInfo().GetElementType();
    if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* pdata = data.GetTensorData<float>();
        for (size_t k = 0; k < data.GetTensorTypeAndShapeInfo().GetElementCount(); k ++) {
            buf.push_back(float2bfloat_cpu(*pdata));  // Cast to bfloat16 directly
            pdata ++;
        }
    }
    /*
    else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        const onnxruntime::BFloat16* pdata = data.GetTensorData<onnxruntime::BFloat16>();
        for (size_t k = 0; k < data.GetTensorTypeAndShapeInfo().GetElementCount(); k ++) {
            buf.push_back(pdata->ToFloat());  // Get float value from BFloat16 structure 
            pdata ++;
        }
    }
    */
    else {
        ORT_CXX_API_THROW("CustomInstanceNormalization supports float or bfloat16 only.",
                          OrtErrorCode::ORT_INVALID_GRAPH);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

KernelCustomInstanceNormalization::KernelCustomInstanceNormalization(const OrtApi& api,
		                                           const OrtKernelInfo* info) : api_(api) {
  Ort::ConstKernelInfo const_info{info};
  info_ = const_info.Copy();

  auto status = api_.KernelInfoGetAttribute_float(info_, "epsilon", &epsilon_);
  if (status != nullptr) epsilon_ = 1e-05;
};

KernelCustomInstanceNormalization::~KernelCustomInstanceNormalization() {
};

void KernelCustomInstanceNormalization::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input = ctx.GetInput(0);

  ONNXTensorElementDataType input_type = input.GetTensorTypeAndShapeInfo().GetElementType();
  if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      ORT_CXX_API_THROW("CustomInstanceNormalization supports float only.",
                          OrtErrorCode::ORT_INVALID_GRAPH);
  }

  std::vector<int64_t> input_shape = input.GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() < 2) {
      ORT_CXX_API_THROW("CustomInstanceNormalization supports 2 dims at least.",
                          OrtErrorCode::ORT_INVALID_GRAPH);
  }

  // Fisrtly cast gamma (weight) and beta (bias) to bfloat16 and express them in float datatype
  auto weight = ctx.GetInput(1);
  gamma_.clear();
  PrepareForBF16(weight, gamma_);

  auto bias = ctx.GetInput(2);
  beta_.clear();
  PrepareForBF16(bias, beta_);

  // Calculate mean and variance of each channel in input tensor, note that it's NCHW order
  int64_t batch = input_shape[0];
  int64_t channel = input_shape[1];
  int64_t size = 1;  // this equals H*W (could extend to more dims)
  for (size_t index = 2; index < input_shape.size(); index ++)
      size = size * input_shape[index];

  means_.clear();
  variances_.clear();

  const float* pinput = input.GetTensorData<float>();
  calculate_mean_var(pinput, batch, channel, size, means_, variances_);

  // Execute instance normalization algorithm
  auto output = ctx.GetOutput(0, input_shape);

  float* poutput = output.GetTensorMutableData<float>();
  instance_normalization(pinput, batch, channel, size, means_, variances_, gamma_, beta_, epsilon_, poutput);
};

}  // namespace vai_q
