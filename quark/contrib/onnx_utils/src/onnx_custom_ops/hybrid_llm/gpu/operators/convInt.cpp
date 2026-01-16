// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/convInt.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for ConvIntParams
static const ConvIntParams ConvIntDefaults = {
  1,                 // c
  1,                 // n
  1,                 // k
  1,                 // groups
  false,             // bias
  true,              // hasScales
  true,              // hasXZero
  true,              // hasWZero
  true,              // hasYZero
  false,             // quantWPerK
  {1, 1},            // inputSize
  {1, 1},            // filterSize
  {0, 0},            // startPad
  {0, 0},            // endPad
  {1, 1},            // stride
  {1, 1},            // dilation
  L"NCHW",           // orderX
  L"NCHW",           // orderW
  L"NCHW",           // orderY
  {-1, -1, -1, -1},  // stridesX
  {-1, -1, -1, -1},  // stridesW
  {-1, -1, -1, -1},  // stridesY
  -1,                // strideB
  L"Int8",           // dataType
};

// =====================================================================================================================
ConvIntOperator::ConvIntOperator(
  const Context& context,
  bool
    disableMetacmds,  // If metacommands should be disabled for this operator.
  const ConvIntParams& params
)
  : m_params(params),
    m_inputType(StringToDataType(params.dataType)),
    m_outputType(params.hasScales ? m_inputType : DML_TENSOR_DATA_TYPE_INT32) {
  if ((m_inputType != DML_TENSOR_DATA_TYPE_INT8) &&
      (m_inputType != DML_TENSOR_DATA_TYPE_UINT8)) {
    std::wstringstream ss;
    ss << "ConvInt only supports Int8 and Uint8, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  constexpr int32 DimCount = 2;

  // Infer the output size from the convolution parameters.
  int32 outputSize[DimCount] = {};

  for (int32 idx = 0; idx < DimCount; ++idx) {
    const int32 dilatedSize =
      (params.filterSize[idx] - 1) * params.dilation[idx] + 1;
    const int32 numerator = params.inputSize[idx] + params.startPad[idx] +
                            params.endPad[idx] - dilatedSize;

    outputSize[idx] = (numerator / params.stride[idx]) + 1;
  }

  // The input, filter, and output tensors are the only required bindings.
  std::vector<int64> shapeX{params.n, params.c};
  std::vector<int64> shapeW{params.k, params.c / params.groups};
  std::vector<int64> shapeY{params.n, params.k};

  for (int32 idx = 0; idx < DimCount; ++idx) {
    shapeX.push_back(params.inputSize[idx]);
    shapeW.push_back(params.filterSize[idx]);
    shapeY.push_back(outputSize[idx]);
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

  m_inputTensorDescVec.resize(ConvIntInputCount);
  m_inputTensorDescVec[ConvIntInputX] =
    CreateTensorDesc(L"X", params.orderX, m_inputType, shapeX, stridesX);
  m_inputTensorDescVec[ConvIntInputW] =
    CreateTensorDesc(L"W", params.orderW, m_inputType, shapeW, stridesW);

  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"Y", params.orderY, m_outputType, shapeY, stridesY)
  );

  // The following tensors are optional.
  const std::vector<int64> shapeWQuant{
    1, (params.quantWPerK ? params.k : 1), 1, 1
  };
  const std::vector<int64> shapeScalar{1, 1, 1, 1};
  const std::vector<int64> autoStrides{-1, -1, -1, -1};

  if (params.hasScales) {
    constexpr DML_TENSOR_DATA_TYPE ScaleType = DML_TENSOR_DATA_TYPE_FLOAT32;
    m_inputTensorDescVec[ConvIntInputXScale] =
      CreateTensorDesc(L"XScale", L"NCHW", ScaleType, shapeScalar, autoStrides);
    m_inputTensorDescVec[ConvIntInputWScale] =
      CreateTensorDesc(L"WScale", L"NCHW", ScaleType, shapeWQuant, autoStrides);
    m_inputTensorDescVec[ConvIntInputYScale] =
      CreateTensorDesc(L"YScale", L"NCHW", ScaleType, shapeScalar, autoStrides);
  }

  if (params.hasXZero) {
    m_inputTensorDescVec[ConvIntInputXZp] =
      CreateTensorDesc(L"XZp", L"NCHW", m_inputType, shapeScalar, autoStrides);
  }

  if (params.hasWZero) {
    m_inputTensorDescVec[ConvIntInputWZp] =
      CreateTensorDesc(L"WZp", L"NCHW", m_inputType, shapeWQuant, autoStrides);
  }

  // These tensors don't exist in the non-quantize verison of this operator.
  if (params.hasScales) {
    if (params.hasYZero) {
      m_inputTensorDescVec[ConvIntInputYZp] = CreateTensorDesc(
        L"YZp", L"NCHW", m_inputType, shapeScalar, autoStrides
      );
    }

    if (params.bias) {
      const std::vector<int64> shapeB{1, params.k, 1, 1};
      const std::vector<int64> stridesB{-1, params.strideB, -1, -1};

      // DirectML expects int32 values for the bias.
      m_inputTensorDescVec[ConvIntInputB] = CreateTensorDesc(
        L"B", L"NCHW", DML_TENSOR_DATA_TYPE_INT32, shapeB, stridesB
      );
    }
  }

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlInputs[ConvIntInputCount] = {};
  DML_TENSOR_DESC* pInputs[ConvIntInputCount] = {};

  for (size_t idx = 0; idx < m_inputTensorDescVec.size(); ++idx) {
    if (m_inputTensorDescVec[idx] != nullptr) {
      ConvertTensorDesc(*m_inputTensorDescVec[idx], dmlInputs + idx);
      pInputs[idx] = &dmlInputs[idx].desc;
    }
  }

  DmlTensorDesc dmlTensorOutput = {};
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorOutput);

  // These arrays must have one value for each convolution area dimension.
  UINT dmlStrides[DimCount] = {};
  UINT dmlDilations[DimCount] = {};
  UINT dmlStartPadding[DimCount] = {};
  UINT dmlEndPadding[DimCount] = {};
  UINT dmlOutputPadding[DimCount] = {};

  for (int32 idx = 0; idx < DimCount; ++idx) {
    dmlStrides[idx] = static_cast<UINT>(params.stride[idx]);
    dmlDilations[idx] = static_cast<UINT>(params.dilation[idx]);
    dmlStartPadding[idx] = static_cast<UINT>(params.startPad[idx]);
    dmlEndPadding[idx] = static_cast<UINT>(params.endPad[idx]);
  }

  // Build and compile the a DirectML operator. DirectML defines two very
  // similar integer matrix multiply operators, the main difference between them
  // is whether or not the output is quantized.
  if (params.hasScales) {
    DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC dmlDesc = {};
    dmlDesc.InputTensor = pInputs[ConvIntInputX];
    dmlDesc.InputScaleTensor = pInputs[ConvIntInputXScale];
    dmlDesc.InputZeroPointTensor = pInputs[ConvIntInputXZp];
    dmlDesc.FilterTensor = pInputs[ConvIntInputW];
    dmlDesc.FilterScaleTensor = pInputs[ConvIntInputWScale];
    dmlDesc.FilterZeroPointTensor = pInputs[ConvIntInputWZp];
    dmlDesc.BiasTensor = pInputs[ConvIntInputB];
    dmlDesc.OutputScaleTensor = pInputs[ConvIntInputYScale];
    dmlDesc.OutputZeroPointTensor = pInputs[ConvIntInputYZp];
    dmlDesc.OutputTensor = &dmlTensorOutput.desc;
    dmlDesc.DimensionCount = 2;
    dmlDesc.Strides = dmlStrides;
    dmlDesc.Dilations = dmlDilations;
    dmlDesc.StartPadding = dmlStartPadding;
    dmlDesc.EndPadding = dmlEndPadding;
    dmlDesc.GroupCount = static_cast<UINT>(params.groups);

    DML_OPERATOR_DESC desc = {};
    desc.Type = DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION;
    desc.Desc = &dmlDesc;

    m_operator = context.CompileOperator(&desc, disableMetacmds, false);
  } else {
    DML_CONVOLUTION_INTEGER_OPERATOR_DESC dmlDesc = {};
    dmlDesc.InputTensor = pInputs[ConvIntInputX];
    dmlDesc.InputZeroPointTensor = pInputs[ConvIntInputXZp];
    dmlDesc.FilterTensor = pInputs[ConvIntInputW];
    dmlDesc.FilterZeroPointTensor = pInputs[ConvIntInputWZp];
    dmlDesc.OutputTensor = &dmlTensorOutput.desc;
    dmlDesc.DimensionCount = 2;
    dmlDesc.Strides = dmlStrides;
    dmlDesc.Dilations = dmlDilations;
    dmlDesc.StartPadding = dmlStartPadding;
    dmlDesc.EndPadding = dmlEndPadding;
    dmlDesc.GroupCount = static_cast<UINT>(params.groups);

    DML_OPERATOR_DESC desc = {};
    desc.Type = DML_OPERATOR_CONVOLUTION_INTEGER;
    desc.Desc = &dmlDesc;

    m_operator = context.CompileOperator(&desc, disableMetacmds, false);
  }
}

// =====================================================================================================================
void ConvIntOperator::BindInputs(
  CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
) const {
  size_t indices[ConvIntInputCount] = {};
  uint32 count = 0;

  if (m_params.hasScales) {
    // Use the DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC binding order.
    indices[count++] = ConvIntInputX;
    indices[count++] = ConvIntInputXScale;
    indices[count++] = ConvIntInputXZp;
    indices[count++] = ConvIntInputW;
    indices[count++] = ConvIntInputWScale;
    indices[count++] = ConvIntInputWZp;
    indices[count++] = ConvIntInputB;
    indices[count++] = ConvIntInputYScale;
    indices[count++] = ConvIntInputYZp;
  } else {
    // Use the DML_OPERATOR_CONVOLUTION_INTEGER binding order.
    indices[count++] = ConvIntInputX;
    indices[count++] = ConvIntInputXZp;
    indices[count++] = ConvIntInputW;
    indices[count++] = ConvIntInputWZp;
  }

  DML_BUFFER_BINDING bindings[ConvIntInputCount] = {};
  DML_BINDING_DESC descs[ConvIntInputCount] = {};

  /*for (size_t idx = 0; idx < count; ++idx) {
    BuildBufferBindDesc(inputs[indices[idx]], bindings + idx, descs + idx);
  }*/

  bindingTable->BindInputs(count, descs);
}

}  // namespace ryzenai::onnx_utils
