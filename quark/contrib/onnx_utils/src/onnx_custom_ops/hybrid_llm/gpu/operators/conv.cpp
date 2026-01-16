// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "operators/conv.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for ConvParams
static const ConvParams ConvDefaults = {
  1,                     // c
  1,                     // n
  1,                     // k
  1,                     // groups
  2,                     // dims
  false,                 // bias
  false,                 // backward
  true,                  // crossCorrelation
  {1, 1, 1},             // inputSize
  {1, 1, 1},             // filterSize
  {0, 0, 0},             // startPad
  {0, 0, 0},             // endPad
  {0, 0, 0},             // outPad
  {1, 1, 1},             // stride
  {1, 1, 1},             // dilation
  L"NCDHW",              // orderX
  L"NCDHW",              // orderW
  L"NCDHW",              // orderY
  {-1, -1, -1, -1, -1},  // stridesX
  {-1, -1, -1, -1, -1},  // stridesW
  {-1, -1, -1, -1, -1},  // stridesY
  -1,                    // strideB
  L"Float",              // dataType
  L"None",               // activationFunc
  0,                     // activationParam1
  0,                     // activationParam2
};

// =====================================================================================================================
// Some basic validation to run over ConvParams.
void ConvOperator::ValidateConvParams(const ConvParams& params) const {
  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::wstringstream ss;
    ss << "Conv only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // Verify that the ignored parameters have default/no-op values. This catches
  // users who fail to set dims = 3.
  for (int32 idx = params.dims; idx < 3; ++idx) {
    if (params.inputSize[idx] != 1) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", inputSize[" << idx
         << "] must have a default value, not ";
      ss << params.inputSize[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.filterSize[idx] != 1) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", filterSize[" << idx
         << "] must have a default value, not ";
      ss << params.filterSize[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.startPad[idx] != 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", startPad[" << idx
         << "] must have a default value, not ";
      ss << params.startPad[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.endPad[idx] != 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", endPad[" << idx
         << "] must have a default value, not ";
      ss << params.endPad[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.outPad[idx] != 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", outPad[" << idx
         << "] must have a default value, not ";
      ss << params.outPad[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.stride[idx] != 1) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", stride[" << idx
         << "] must have a default value, not ";
      ss << params.stride[idx];
      throw std::runtime_error(ss.str());
    }

    if (params.dilation[idx] != 1) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", dilation[" << idx
         << "] must have a default value, not ";
      ss << params.dilation[idx];
      throw std::runtime_error(ss.str());
    }

    // Tensors have two extra dimensions.
    const int32 idx2 = idx + 2;

    if (params.stridesX[idx2] > 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", stridesX[" << idx2
         << "] must have a default value, not ";
      ss << params.stridesX[idx2];
      throw std::runtime_error(ss.str());
    }

    if (params.stridesW[idx2] > 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", stridesW[" << idx2
         << "] must have a default value, not ";
      ss << params.stridesW[idx2];
      throw std::runtime_error(ss.str());
    }

    if (params.stridesY[idx2] > 0) {
      std::stringstream ss;
      ss << "With dims = " << params.dims << ", stridesY[" << idx2
         << "] must have a default value, not ";
      ss << params.stridesY[idx2];
      throw std::runtime_error(ss.str());
    }
  }
}

// =====================================================================================================================
ConvOperator::ConvOperator(
  const Context& context,
  bool disableMetacmds,  // If metacommands should be
                         // disabled for this operator.
  const ConvParams& params
)
  : m_dataType(StringToDataType(params.dataType)) {
  ValidateConvParams(params);

  // Convert the node parameters to generic convolution parameters.
  m_conv.c = params.c;
  m_conv.n = params.n;
  m_conv.k = params.k;
  m_conv.groups = params.groups;
  m_conv.dims = params.dims;
  m_conv.bias = params.bias;

  // We must always swap the filter CK for all backward passes. A forward cross
  // correlation filter looks just like a backward convolution filter so we only
  // need to reverse the filter if the polarities match. Note that our reference
  // function computes a cross correlation (just like most CNN libraries).
  m_conv.swapCK = params.backward;
  m_conv.reverseFilter = params.backward == params.crossCorrelation;

  for (int32 idx = 0; idx < params.dims; ++idx) {
    m_conv.inputSize[idx] = params.inputSize[idx];
    m_conv.filterSize[idx] = params.filterSize[idx];
    m_conv.outPad[idx] = params.outPad[idx];
    m_conv.filterStride[idx] = params.dilation[idx];

    const int32 dilatedSize =
      (params.filterSize[idx] - 1) * params.dilation[idx] + 1;

    if (params.backward == false) {
      m_conv.startPad[idx] = params.startPad[idx];
      m_conv.endPad[idx] = params.endPad[idx];
      m_conv.convStride[idx] = params.stride[idx];
      m_conv.inputStride[idx] = 1;

      const int32 numerator = params.inputSize[idx] + params.startPad[idx] +
                              params.endPad[idx] - dilatedSize;
      m_conv.outputSize[idx] = (numerator / params.stride[idx]) + 1;
    } else {
      // A backwards pass looks like a forward pass where the output tensor is
      // actually the input to some forward operation. The pad, stride, and
      // dilation parameters describe the properties of that reference forward
      // operation instead of acting directly on the real input tensor. This is
      // all very confusing and I recommend reading this article:
      // http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic

      m_conv.startPad[idx] = dilatedSize - params.startPad[idx] - 1;
      m_conv.endPad[idx] = dilatedSize - params.endPad[idx] - 1;
      m_conv.convStride[idx] = 1;
      m_conv.inputStride[idx] = params.stride[idx];
      m_conv.outputSize[idx] =
        (params.stride[idx] * (params.inputSize[idx] - 1) + params.outPad[idx] +
         dilatedSize - params.startPad[idx] - params.endPad[idx]);
    }
  }

  // Apply this activation function.
  m_activation.func = StringToActivationFunc(params.activationFunc);
  m_activation.param1 = params.activationParam1;
  m_activation.param2 = params.activationParam2;

  // Build the tensor descriptors. For DirectML's benefit, we must have one
  // pointer for every possible tensor and must use empty pointers as
  // placeholders for unused optional tensors.
  std::vector<int64> shapeX{params.n, params.c};
  std::vector<int64> shapeW;
  std::vector<int64> shapeY{params.n, params.k};
  std::vector<int64> shapeB{1, params.k};

  if (params.backward == false) {
    shapeW.push_back(params.k);
    shapeW.push_back(params.c / params.groups);
  } else {
    shapeW.push_back(params.c);
    shapeW.push_back(params.k / params.groups);
  }

  for (int32 idx = 0; idx < params.dims; ++idx) {
    shapeX.push_back(params.inputSize[idx]);
    shapeW.push_back(params.filterSize[idx]);
    shapeY.push_back(m_conv.outputSize[idx]);
    shapeB.push_back(1);
  }

  const std::vector<int64> stridesX(
    params.stridesX, params.stridesX + shapeX.size()
  );
  const std::vector<int64> stridesW(
    params.stridesW, params.stridesW + shapeW.size()
  );
  const std::vector<int64> stridesY(
    params.stridesY, params.stridesY + shapeY.size()
  );
  std::vector<int64> stridesB(shapeY.size(), -1);

  stridesB[1] = params.strideB;

  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"X", params.orderX, m_dataType, shapeX, stridesX)
  );
  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"W", params.orderW, m_dataType, shapeW, stridesW)
  );
  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"Y", params.orderY, m_dataType, shapeY, stridesY)
  );

  if (params.bias) {
    m_inputTensorDescVec.emplace_back(
      CreateTensorDesc(L"B", L"", m_dataType, shapeB, stridesB)
    );
  } else {
    m_inputTensorDescVec.emplace_back(std::shared_ptr<TensorDesc>());
  }

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlTensorX = {};
  DmlTensorDesc dmlTensorW = {};
  DmlTensorDesc dmlTensorB = {};
  DmlTensorDesc dmlTensorY = {};

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorX);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlTensorW);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorY);

  if (params.bias) {
    ConvertTensorDesc(*m_inputTensorDescVec[2], &dmlTensorB);
  }

  // Build the main DirectML operator descriptor.
  DML_CONVOLUTION_OPERATOR_DESC dmlDesc = {};
  dmlDesc.InputTensor = &dmlTensorX.desc;
  dmlDesc.FilterTensor = &dmlTensorW.desc;
  dmlDesc.BiasTensor = params.bias ? &dmlTensorB.desc : nullptr;
  dmlDesc.OutputTensor = &dmlTensorY.desc;
  dmlDesc.Mode = params.crossCorrelation
                   ? DML_CONVOLUTION_MODE_CROSS_CORRELATION
                   : DML_CONVOLUTION_MODE_CONVOLUTION;
  dmlDesc.Direction = params.backward ? DML_CONVOLUTION_DIRECTION_BACKWARD
                                      : DML_CONVOLUTION_DIRECTION_FORWARD;
  dmlDesc.DimensionCount = static_cast<UINT>(params.dims);
  dmlDesc.GroupCount = static_cast<UINT>(params.groups);

  // These arrays must have one value for each convolution volume dimension.
  constexpr UINT MaxDimCount = 3;
  UINT dmlStrides[MaxDimCount] = {};
  UINT dmlDilations[MaxDimCount] = {};
  UINT dmlStartPadding[MaxDimCount] = {};
  UINT dmlEndPadding[MaxDimCount] = {};
  UINT dmlOutputPadding[MaxDimCount] = {};

  for (int32 idx = 0; idx < params.dims; ++idx) {
    dmlStrides[idx] = static_cast<UINT>(params.stride[idx]);
    dmlDilations[idx] = static_cast<UINT>(params.dilation[idx]);
    dmlStartPadding[idx] = static_cast<UINT>(params.startPad[idx]);
    dmlEndPadding[idx] = static_cast<UINT>(params.endPad[idx]);
    dmlOutputPadding[idx] = static_cast<UINT>(params.outPad[idx]);
  }

  dmlDesc.Strides = dmlStrides;
  dmlDesc.Dilations = dmlDilations;
  dmlDesc.StartPadding = dmlStartPadding;
  dmlDesc.EndPadding = dmlEndPadding;
  dmlDesc.OutputPadding = dmlOutputPadding;

  // Add on the optional activation func.
  DmlActivationFuncDesc dmlFunc = {};

  if (m_activation.func != ActivationFunc::None) {
    ConvertActivationFuncInfo(m_activation, &dmlFunc);
    dmlDesc.FusedActivation = &dmlFunc.desc;
  }

  // Finally, compile the operator.
  DML_OPERATOR_DESC desc = {};
  desc.Type = DML_OPERATOR_CONVOLUTION;
  desc.Desc = &dmlDesc;

  m_operator = context.CompileOperator(
    &desc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
