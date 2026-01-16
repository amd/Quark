// Copyright (c) 2025 Advanced Micro Devices, Inc.

#include "execution_provider.hpp"

ryzenai::IExecutionProvider* ryzenai::GetCurrentExecutionProvider() {
  return nullptr;
}

std::shared_ptr<ryzenai::CPUGate::Op>
ryzenai::ExecutionProviderExtensions::CreateCPUOp(
  ryzenai::CPUGate::OpParams&& params
) {
  return nullptr;
}
