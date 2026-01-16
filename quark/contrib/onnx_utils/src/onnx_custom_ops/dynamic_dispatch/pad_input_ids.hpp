// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_PAD
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_PAD

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "external_buffers.hpp"
#include "onnx.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "op_fuser/fusion_rt.hpp"
#include "operator.hpp"
#include "ops/ops_common/dtype_utils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace ryzenai::onnx_utils {

class PadInputIdsKernel {
 public:
  PadInputIdsKernel(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

  void Compute(OrtKernelContext* context);

 protected:
 private:
  std::string node_name_;
  std::vector<std::vector<int64_t>> dimensions_;
  std::string dd_cache_dir_;

  void read_attributes(const Ort::ConstKernelInfo& info_ptr);
};

template <const char* Name, typename Kernel>
struct PadInputIds : Operator<Kernel, Name> {
  explicit PadInputIds(const Ort::ConstSessionOptions& session_options)
    : Operator<Kernel, Name>(session_options, GetSessionConfigKeys()) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {"count", "dd_node", "dd_cache", "external_data_file"};
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_PAD
