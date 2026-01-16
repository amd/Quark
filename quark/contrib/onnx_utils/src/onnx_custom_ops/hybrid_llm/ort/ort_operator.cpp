// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "ort_operator.hpp"

#include "execution_provider_cpugate.hpp"

using namespace ryzenai::onnx_utils;

OrtOperator::OrtOperator()
  : memory_info_(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    ) {}

OrtOperator::~OrtOperator() {}

void OrtOperator::createOp(
  const OrtKernelInfo* info, const char* op_name, const char* domain,
  int version,
  std::map<std::string, ONNXTensorElementDataType> type_constraints,
  std::vector<Ort::OpAttr> attrs, size_t input_count, size_t output_count
) {
  if (IsEPAvailable()) {
    ep_op_ = CreateCPUOp(
      {op_name, domain, version, std::move(type_constraints), std::move(attrs),
       input_count, output_count}
    );
    return;
  }

  try {
    std::vector<const char*> type_constraint_names;
    std::vector<ONNXTensorElementDataType> type_constraint_values;

    type_constraint_names.reserve(type_constraints.size());
    type_constraint_values.reserve(type_constraints.size());

    for (const auto& [key, val] : type_constraints) {
      type_constraint_names.push_back(key.c_str());
      type_constraint_values.push_back(val);
    }

    op_ = Ort::Op::Create(
      info, op_name, domain, version, type_constraint_names.data(),
      type_constraint_values.data(), type_constraints.size(), attrs.data(),
      attrs.size(), input_count, output_count
    );
    return;
  } catch (...) {
    // When using with VitisAI EP, the ORT ops don't work so suppress the
    // error here and it'll throw a runtime exception during execution if
    // these ops are actually needed.
  }
}

bool OrtOperator::isInitialized() const { return op_ || ep_op_; }

void OrtOperator::invokeOp(
  const OrtKernelContext* context, std::vector<const OrtValue*> input,
  std::vector<OrtValue*> output
) {
  if (op_)
    return op_.Invoke(
      context, input.data(), input.size(), output.data(), output.size()
    );

  if (ep_op_) return ep_op_->Invoke(std::move(input), std::move(output));

  throw std::runtime_error{"CPU Op was not initialized"};
}
