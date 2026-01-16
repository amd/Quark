// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/matMul.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for MatMulIntParams
static const MatMulIntParams MatMulIntDefaults = {
  1,                 // m
  1,                 // n
  1,                 // k
  1,                 // batches
  1,                 // channels
  true,              // hasScales
  true,              // hasAZero
  true,              // hasBZero
  true,              // hasYZero
  false,             // quantizeARows
  false,             // quantizeBCols
  false,             // quantizeYRows
  L"NCHW",           // orderA
  L"NCHW",           // orderB
  L"NCHW",           // orderY
  {-1, -1, -1, -1},  // stridesA
  {-1, -1, -1, -1},  // stridesB
  {-1, -1, -1, -1},  // stridesY
  L"Int8",           // dataType
};

// =====================================================================================================================
MatMulIntOperator::MatMulIntOperator(
  const Context& context,
  bool
    disableMetacmds,  // If metacommands should be disabled for this operator.
  const MatMulIntParams& params
)
  : m_params(params),
    m_inputType(StringToDataType(params.dataType)),
    m_outputType(params.hasScales ? m_inputType : DML_TENSOR_DATA_TYPE_INT32) {
  if ((m_inputType != DML_TENSOR_DATA_TYPE_INT8) &&
      (m_inputType != DML_TENSOR_DATA_TYPE_UINT8)) {
    std::wstringstream ss;
    ss << "MatMulInt only supports Int8 and Uint8, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // The A, B, and Y tensors are the only required bindings.
  const std::vector<int64> shapeA{
    params.batches, params.channels, params.m, params.k
  };
  const std::vector<int64> shapeB{
    params.batches, params.channels, params.k, params.n
  };
  const std::vector<int64> shapeY{
    params.batches, params.channels, params.m, params.n
  };

  const std::vector<int64> stridesA(
    params.stridesA, params.stridesA + shapeA.size()
  );
  const std::vector<int64> stridesB(
    params.stridesB, params.stridesB + shapeB.size()
  );
  const std::vector<int64> stridesY(
    params.stridesY, params.stridesY + shapeY.size()
  );

  m_inputTensorDescVec.resize(MatMulIntInputCount);
  m_inputTensorDescVec[MatMulIntInputA] =
    CreateTensorDesc(L"A", params.orderA, m_inputType, shapeA, stridesA);
  m_inputTensorDescVec[MatMulIntInputB] =
    CreateTensorDesc(L"B", params.orderB, m_inputType, shapeB, stridesB);

  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"Y", params.orderY, m_outputType, shapeY, stridesY)
  );

  // The following tensors are optional.
  const std::vector<int64> shapeAQuant{
    1, 1, (params.quantizeARows ? params.m : 1), 1
  };
  const std::vector<int64> shapeBQuant{
    1, 1, 1, (params.quantizeBCols ? params.n : 1)
  };
  const std::vector<int64> shapeYQuant{
    1, 1, (params.quantizeYRows ? params.m : 1), 1
  };
  const std::vector<int64> autoStrides{-1, -1, -1, -1};

  if (params.hasScales) {
    constexpr DML_TENSOR_DATA_TYPE ScaleType = DML_TENSOR_DATA_TYPE_FLOAT32;
    m_inputTensorDescVec[MatMulIntInputAScale] =
      CreateTensorDesc(L"AScale", L"NCHW", ScaleType, shapeAQuant, autoStrides);
    m_inputTensorDescVec[MatMulIntInputBScale] =
      CreateTensorDesc(L"BScale", L"NCHW", ScaleType, shapeBQuant, autoStrides);
    m_inputTensorDescVec[MatMulIntInputYScale] =
      CreateTensorDesc(L"YScale", L"NCHW", ScaleType, shapeYQuant, autoStrides);
  }

  if (params.hasAZero) {
    m_inputTensorDescVec[MatMulIntInputAZp] =
      CreateTensorDesc(L"AZp", L"NCHW", m_inputType, shapeAQuant, autoStrides);
  }

  if (params.hasBZero) {
    m_inputTensorDescVec[MatMulIntInputBZp] =
      CreateTensorDesc(L"BZp", L"NCHW", m_inputType, shapeBQuant, autoStrides);
  }

  // The output zero point doesn't exist in the non-quantize verison of this
  // operator.
  if (params.hasScales && params.hasYZero) {
    m_inputTensorDescVec[MatMulIntInputYZp] =
      CreateTensorDesc(L"YZp", L"NCHW", m_inputType, shapeYQuant, autoStrides);
  }

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlInputs[MatMulIntInputCount] = {};
  DML_TENSOR_DESC* pInputs[MatMulIntInputCount] = {};

  for (size_t idx = 0; idx < m_inputTensorDescVec.size(); ++idx) {
    if (m_inputTensorDescVec[idx] != nullptr) {
      ConvertTensorDesc(*m_inputTensorDescVec[idx], dmlInputs + idx);
      pInputs[idx] = &dmlInputs[idx].desc;
    }
  }

  DmlTensorDesc dmlTensorY = {};
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorY);

  // Build and compile the a DirectML operator. DirectML defines two very
  // similar integer matrix multiply operators, the main difference between them
  // is whether or not the output is quantized.
  if (params.hasScales) {
    DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC dmlDesc = {};
    dmlDesc.ATensor = pInputs[MatMulIntInputA];
    dmlDesc.AScaleTensor = pInputs[MatMulIntInputAScale];
    dmlDesc.AZeroPointTensor = pInputs[MatMulIntInputAZp];
    dmlDesc.BTensor = pInputs[MatMulIntInputB];
    dmlDesc.BScaleTensor = pInputs[MatMulIntInputBScale];
    dmlDesc.BZeroPointTensor = pInputs[MatMulIntInputBZp];
    dmlDesc.OutputScaleTensor = pInputs[MatMulIntInputYScale];
    dmlDesc.OutputZeroPointTensor = pInputs[MatMulIntInputYZp];
    dmlDesc.OutputTensor = &dmlTensorY.desc;

    DML_OPERATOR_DESC desc = {};
    desc.Type = DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY;
    desc.Desc = &dmlDesc;

    m_operator = context.CompileOperator(&desc, disableMetacmds, false);
  } else {
    DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC dmlDesc = {};
    dmlDesc.ATensor = pInputs[MatMulIntInputA];
    dmlDesc.AZeroPointTensor = pInputs[MatMulIntInputAZp];
    dmlDesc.BTensor = pInputs[MatMulIntInputB];
    dmlDesc.BZeroPointTensor = pInputs[MatMulIntInputBZp];
    dmlDesc.OutputTensor = &dmlTensorY.desc;

    DML_OPERATOR_DESC desc = {};
    desc.Type = DML_OPERATOR_MATRIX_MULTIPLY_INTEGER;
    desc.Desc = &dmlDesc;

    m_operator = context.CompileOperator(&desc, disableMetacmds, false);
  }
}

// =====================================================================================================================
void MatMulIntOperator::BindInputs(
  CComPtr<IDMLBindingTable> bindingTable, const TensorVector& inputs
) const {
  size_t indices[MatMulIntInputCount] = {};
  uint32 count = 0;

  if (m_params.hasScales) {
    // Use the DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY binding order.
    indices[count++] = MatMulIntInputA;
    indices[count++] = MatMulIntInputAScale;
    indices[count++] = MatMulIntInputAZp;
    indices[count++] = MatMulIntInputB;
    indices[count++] = MatMulIntInputBScale;
    indices[count++] = MatMulIntInputBZp;
    indices[count++] = MatMulIntInputYScale;
    indices[count++] = MatMulIntInputYZp;
  } else {
    // Use the DML_OPERATOR_MATRIX_MULTIPLY_INTEGER binding order.
    indices[count++] = MatMulIntInputA;
    indices[count++] = MatMulIntInputAZp;
    indices[count++] = MatMulIntInputB;
    indices[count++] = MatMulIntInputBZp;
  }

  DML_BUFFER_BINDING bindings[MatMulIntInputCount] = {};
  DML_BINDING_DESC descs[MatMulIntInputCount] = {};

  /*for (size_t idx = 0; idx < count; ++idx) {
    BuildBufferBindDesc(inputs[indices[idx]], bindings + idx, descs + idx);
  }*/

  bindingTable->BindInputs(count, descs);
}

}  // namespace ryzenai::onnx_utils
