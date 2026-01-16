// Copyright (C) 2021 - 2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/gemm.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for GemmParams
static const GemmParams GemmDefaults = {
  1,             // m
  1,             // n
  1,             // k
  1,             // batches
  1.0f,          // alpha
  1.0f,          // beta
  false,         // transA
  false,         // transB
  false,         // hasC
  L"DHW",        // orderA
  L"DHW",        // orderB
  L"DHW",        // orderC
  L"DHW",        // orderY
  {-1, -1, -1},  // stridesA
  {-1, -1, -1},  // stridesB
  {-1, -1, -1},  // stridesC
  {-1, -1, -1},  // stridesY
  L"Float",      // dataType
  L"None",       // activationFunc
  0,             // activationParam1
  0,             // activationParam2
};

// =====================================================================================================================
GemmOperator::GemmOperator(
  const Context& context,
  bool disableMetacmds,  // If metacommands should be
                         // disabled for this operator.
  const GemmParams& params
)
  : m_gemm(params), m_dataType(StringToDataType(params.dataType)) {
  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::wstringstream ss;
    ss << "Gemm only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // Apply this activation function.
  m_activation.func = StringToActivationFunc(params.activationFunc);
  m_activation.param1 = params.activationParam1;
  m_activation.param2 = params.activationParam2;

  // Build the tensor descriptors. For DirectML's benefit, we must have one
  // pointer for every possible tensor and must use empty pointers as
  // placeholders for unused optional tensors.
  const int64 heightA = m_gemm.transA ? m_gemm.k : m_gemm.m;
  const int64 widthA = m_gemm.transA ? m_gemm.m : m_gemm.k;
  const int64 heightB = m_gemm.transB ? m_gemm.n : m_gemm.k;
  const int64 widthB = m_gemm.transB ? m_gemm.k : m_gemm.n;

  const std::vector<int64> sizeA{m_gemm.batches, heightA, widthA};
  const std::vector<int64> sizeB{m_gemm.batches, heightB, widthB};
  const std::vector<int64> sizeY{m_gemm.batches, m_gemm.m, m_gemm.n};

  const std::vector<int64> stridesA{
    params.stridesA[0], params.stridesA[1], params.stridesA[2]
  };
  const std::vector<int64> stridesB{
    params.stridesB[0], params.stridesB[1], params.stridesB[2]
  };
  const std::vector<int64> stridesC{
    params.stridesC[0], params.stridesC[1], params.stridesC[2]
  };
  const std::vector<int64> stridesY{
    params.stridesY[0], params.stridesY[1], params.stridesY[2]
  };

  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"A", params.orderA, m_dataType, sizeA, stridesA)
  );
  m_inputTensorDescVec.emplace_back(
    CreateTensorDesc(L"B", params.orderB, m_dataType, sizeB, stridesB)
  );
  m_outputTensorDescVec.emplace_back(
    CreateTensorDesc(L"Y", params.orderY, m_dataType, sizeY, stridesY)
  );

  if (params.hasC) {
    // Note that C always has the same size as Y.
    m_inputTensorDescVec.emplace_back(
      CreateTensorDesc(L"C", params.orderC, m_dataType, sizeY, stridesC)
    );
  } else {
    m_inputTensorDescVec.emplace_back(std::shared_ptr<TensorDesc>());
  }

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlTensorA = {};
  DmlTensorDesc dmlTensorB = {};
  DmlTensorDesc dmlTensorC = {};
  DmlTensorDesc dmlTensorY = {};

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorA);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlTensorB);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorY);

  if (params.hasC) {
    ConvertTensorDesc(*m_inputTensorDescVec[2], &dmlTensorC);
  }

  // Build the main DirectML operator descriptor.
  DML_GEMM_OPERATOR_DESC dmlDesc = {};
  dmlDesc.ATensor = &dmlTensorA.desc;
  dmlDesc.BTensor = &dmlTensorB.desc;
  dmlDesc.CTensor = params.hasC ? &dmlTensorC.desc : nullptr;
  dmlDesc.OutputTensor = &dmlTensorY.desc;
  dmlDesc.TransA =
    params.transA ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
  dmlDesc.TransB =
    params.transB ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
  dmlDesc.Alpha = params.alpha;
  dmlDesc.Beta = params.beta;

  // Add on the optional activation func.
  DmlActivationFuncDesc dmlFunc = {};

  if (m_activation.func != ActivationFunc::None) {
    ConvertActivationFuncInfo(m_activation, &dmlFunc);
    dmlDesc.FusedActivation = &dmlFunc.desc;
  }

  // Finally, compile the operator.
  DML_OPERATOR_DESC desc = {};
  desc.Type = DML_OPERATOR_GEMM;
  desc.Desc = &dmlDesc;

  m_operator = context.CompileOperator(
    &desc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
