// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_PAD
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_PAD

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "operator.hpp"

namespace ryzenai::onnx_utils {

class DynamicPadKernel {
 public:
  DynamicPadKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  void Compute(OrtKernelContext* context);

 protected:
 private:
  std::string node_name_;
  std::vector<std::map<std::string, size_t>> candidates_;
  std::map<int, std::string> dyn_keys_;
  std::string dd_cache_dir_;
  Ort::Logger logger_{nullptr};

  const OrtKernelInfo* ker_info_ = nullptr;

  void read_attributes(const Ort::ConstKernelInfo& info_ptr);

  std::vector<int64_t> get_best_candidate(
    const std::vector<int64_t>& input_shape
  );
};

template <const char* Name, typename Kernel>
struct DynamicPadOp : Operator<Kernel, Name> {
  explicit DynamicPadOp(const Ort::ConstSessionOptions& session_options)
    : Operator<Kernel, Name>(session_options, GetSessionConfigKeys()) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"dd_cache", "external_data_file"};
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_PAD
