// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#include "operators/multHeadAttn.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for MhaParams
static const MhaParams MhaDefaults = {
  1,           // batch
  1,           // qSeq
  1,           // kvSeq
  1,           // sizeHeads
  2,           // headCount
  1.0f,        // scale
  L"Self",     // packing
  L"Float16",  // dataType
};

// =====================================================================================================================
// Some basic validation to run over MhaParams.
void MhaOperator::ValidateMhaParams(const MhaParams& params) const {
  if (params.packing == L"Self") {
    if (params.kvSeq != params.qSeq) {
      std::stringstream ss;
      ss << "for Self Attention, q and KV sequence sizes must match";
      throw std::runtime_error(ss.str());
    }
  }

  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::stringstream ss;
    ss << "MHA only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(ss.str());
  }
}

// =====================================================================================================================
MhaOperator::MhaOperator(
  const Context& context,
  bool disableMetacmds,  // If metacommands should be
                         // disabled for this operator.
  const MhaParams& params
)
  : m_dataType(StringToDataType(params.dataType)) {
  ValidateMhaParams(params);
  m_inputTensorDescVec.resize(11);  // required binding sizes for MHA
  m_outputTensorDescVec.resize(3);

  if (params.packing == L"Cross") {
    const std::vector<int64> sizeQ{
      params.batch, params.qSeq, params.headCount * params.sizeHeads
    };

    const std::vector<int64> sizeKV{
      params.batch, params.kvSeq, params.headCount, 2, params.sizeHeads
    };

    const std::vector<int64> stridesQ{-1, -1, -1};
    const std::vector<int64> stridesKV{-1, -1, -1, -1, -1};

    m_inputTensorDescVec.at(0) = CreateTensorDesc(
      L"Q", L"NCDHW", m_dataType, sizeQ,
      stridesQ
    );  // map to Q, KV and QKV tensors
    m_inputTensorDescVec.at(4) =
      CreateTensorDesc(L"KV", L"NCDHW", m_dataType, sizeKV, stridesKV);

  } else if (params.packing == L"Self") {
    const std::vector<int64> sizeQKV{
      params.batch, params.qSeq, params.headCount, 3, params.sizeHeads
    };
    const std::vector<int64> stridesQKV{-1, -1, -1, -1, -1};
    m_inputTensorDescVec.at(5) =
      CreateTensorDesc(L"QKV", L"NCDHW", m_dataType, sizeQKV, stridesQKV);
  }

  const std::vector<int64> sizeOut{
    params.batch, params.qSeq, params.headCount * params.sizeHeads
  };
  const std::vector<int64> stridesOut{-1, -1, -1};
  m_outputTensorDescVec.at(0) =
    CreateTensorDesc(L"Output", L"NCDHW", m_dataType, sizeOut, stridesOut);

  DML_MULTIHEAD_ATTENTION_OPERATOR_DESC dmlDesc = {};

  dmlDesc.HeadCount = params.headCount;
  dmlDesc.Scale = params.scale;

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlTensorQ = {};
  DmlTensorDesc dmlTensorKV = {};
  DmlTensorDesc dmlTensorQKV = {};
  DmlTensorDesc dmlTensorOut = {};

  if (params.packing == L"Cross") {
    ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorQ);
    ConvertTensorDesc(*m_inputTensorDescVec[4], &dmlTensorKV);
    dmlDesc.QueryTensor = &dmlTensorQ.desc;
    dmlDesc.StackedKeyValueTensor = &dmlTensorKV.desc;
  } else if (params.packing == L"Self") {
    ConvertTensorDesc(*m_inputTensorDescVec[5], &dmlTensorQKV);
    dmlDesc.StackedQueryKeyValueTensor = &dmlTensorQKV.desc;
  }
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorOut);
  dmlDesc.OutputTensor = &dmlTensorOut.desc;

  // Finally, compile the operator.
  DML_OPERATOR_DESC desc = {};
  desc.Type = DML_OPERATOR_MULTIHEAD_ATTENTION;
  desc.Desc = &dmlDesc;

  m_operator = context.CompileOperator(
    &desc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16)
  );
}

}  // namespace ryzenai::onnx_utils
