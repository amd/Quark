// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_DISPATCH
#define GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_DISPATCH

#include <any>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <type_traits>

#include "custom_ops.hpp"
#include "execution_provider.hpp"
#include "external_buffers.hpp"
#include "external_data.hpp"
#include "lora.hpp"
#include "memory_manager/VAIMemoryManager.hpp"
#include "onnx.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "op_fuser/fusion_rt.hpp"
#include "operator.hpp"
#include "ops/ops_common/dtype_utils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#include "xclbin_container.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

std::string getCacheDirectory(
  const std::unordered_map<std::string, std::string>& session_configs
);

class DynamicDispatchKernelBase : public LoraOpInterface {
 public:
  DynamicDispatchKernelBase(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );

 protected:
  void initialize_fusion_rt(
    const Ort::ConstKernelInfo& info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  void loadLoraData();
  virtual void LoadLora();

  ModelType model_type_ = ModelType::Unknown;
  std::string model_directory_;
  std::string dd_root_;
  std::string xclbin_;
  std::string node_name_;
  std::string compile_fusion_rt_;
  std::string dd_const_key_;
  std::unique_ptr<::OpsFusion::FusionRuntime> rt_;
  // Keep xrt_context and hw_context alive for shared memory manager
  std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_ctx_;
  // Persistent hw_context for shared memory manager
  xrt::hw_context ctx_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<std::string>> dynamic_input_shapes_;
  std::vector<std::vector<bool>> is_inputs_dims_dynamic_;
  // std::unordered_set<std::string> symbolic_dims_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<std::vector<std::string>> dynamic_output_shapes_;
  std::vector<std::vector<bool>> is_outputs_dims_dynamic_;
  std::vector<int64_t> inp_shapes_padding_;
  std::vector<int64_t> out_shapes_padding_;
  std::vector<std::vector<size_t>> external_tensors_;
  ExternalBuffers external_buffers_;
  OpsFusion::Metadata meta_;
  uint32_t past_seq_len_ = 0;
  int64_t local_window_size_ = 0;
  int64_t seq_len_index_ = -1;
  std::vector<OpsFusion::DynamicShapeInfo> dynamic_dim_maps_;
  int64_t find_dynamic_shape_list(
    const OpsFusion::DynamicShapeInfo& partial_shape_info
  );
  bool use_shared_memory_ = false;

 private:
  // if the legacy LLM prefill model support can be dropped, this should be
  // deleted
  bool legacy_llm_prefill_model_ = false;

  void read_attributes(const Ort::ConstKernelInfo& info_ptr);
  void parse_dynamic_dims_string(
    const std::string& dynamic_dims_str, std::vector<std::string>& dims,
    std::vector<bool>& is_dynamic
  );
};

class DynamicDispatchKernel : public DynamicDispatchKernelBase,
                              ExecutionProviderExtensions {
 public:
  using DynamicDispatchKernelBase::DynamicDispatchKernelBase;

  void Compute(OrtKernelContext* context);
};

template <const char* Name, typename Kernel>
struct DynamicDispatch : Operator<Kernel, Name> {
  explicit DynamicDispatch(const Ort::ConstSessionOptions& session_options)
    : Operator<Kernel, Name>(session_options, GetSessionConfigKeys()) {}

  std::unordered_set<std::string> GetSessionConfigKeys() const {
    return {
      "dd_cache",
      "model_name",
      "dd_root",
      "compile_fusion_rt",
      "onnx_custom_ops_const_key",
      "fusion_opt_skip_ext_buf_copy",
      "fusion_opt_stack_size",
      "external_data_file",
      "fusion_opt_io_bind_kv_cache",
      "hybrid_opt_enable_npu_preemption",
      "hybrid_opt_enable_npu_lora"
    };
  }
};

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_DYNAMIC_DISPATCH_DYNAMIC_DISPATCH
