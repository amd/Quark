// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_NPU_OP
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_NPU_OP
#include <onnxruntime_cxx_api.h>
#include <ryzenai/ryzen_mm.h>
#include <xrt/xrt_bo.h>

#include <any>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>

#include "../session_state.hpp"
#include "custom_ops.hpp"
#include "execution_provider.hpp"
#include "lora.hpp"
#include "mladf_version.hpp"
#include "shared_buffer.hpp"
#include "shared_weights.hpp"

struct OrtKernelInfo;

namespace Ort {
struct Logger;
namespace detail {
template <typename T>
struct Unowned;
template <typename T>
struct KernelInfoImpl;
}  // namespace detail
using ConstKernelInfo =
  detail::KernelInfoImpl<detail::Unowned<const OrtKernelInfo>>;
}  // namespace Ort

namespace ryzenai::onnx_utils {

namespace proto {
class Header;
}

template <typename F>
void conditionalTry(F&& f, bool condition, const std::string& label) {
  try {
    f();
  } catch (std::exception& e) {
    if (condition) {
      std::cerr << label << " : " << e.what() << std::endl;
    } else {
      throw;
    }
  } catch (...) {
    if (condition) {
      std::cerr << label << " : failed with unknown error" << std::endl;
    } else {
      throw;
    }
  }
}

// By using this constant we're forcing DD op to always use same set of bos and
// avoid switching between prefill/token sets.
constexpr const auto kGemmBOsSelector = -1;

struct OrtTensor {
  std::vector<int64_t> shape;
  size_t size;
  void* data;
};

class NpuOp : public ExecutionProviderExtensions {
 public:
  explicit NpuOp(
    const OrtKernelInfo* kernel_info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  virtual ~NpuOp();

 protected:
  /**
   * @brief Perform common initialization for all child classes
   *
   */
  std::shared_ptr<proto::Header> initializeNpuOp(
    const char* op_type,
    const std::unordered_map<std::string, std::string>& session_configs,
    Ort::ConstKernelInfo& info
  );

  /**
   * @brief Get the name of the node name
   *
   * @return const std::string&
   */
  const std::string& name() const;

  std::shared_ptr<proto::Header> setExternalData(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  const std::filesystem::path& externalData() const;
  bool useExternalData() const;

  void setFreeAfterPrefill(
    const std::unordered_map<std::string, std::string>& session_configs,
    const proto::Header* header, std::string_view op_type
  );
  bool freeAfterPrefill(std::string_view node_name) const;

  void setLastNode(const proto::Header* header);
  bool lastNode(std::string_view node_name);
  void setDynamicJitFactor(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  float dynamicJitFactor() const;

  void setContinueOnException(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  bool continueOnException() const;

  void setMaxSeqLength(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  std::size_t maxSeqLength() const;

  template <typename F>
  void tryContinueOnException(F&& f) {
    conditionalTry(std::move(f), continue_on_exception_, node_name_);
  }

  size_t getInitPromptSize(
    const std::unordered_map<std::string, std::string>& session_configs
  ) const;

  virtual void initializeKernels() = 0;
  virtual void UpdateSharedBuffer(size_t kernel_size) = 0;

  // For now choose buckets of [1024, 2048, 3072]
  // Most NPU operators have tiling support, however
  // BMM operator does not, so stick to set
  // of shapes it supports and make it coarse-grained
  // so there is not a lot of re-allocs
  // IMPORTANT: assumes all DD operators will have different size
  //            constraints for each of these buckets
  //            i.e. there wont be an operator that will always
  //            allocate same size across any of the buckets
  //            this helps to simplify re-binding xrt::bo logic
  size_t getNPUKernelGranularity(
    std::int64_t prompt_size,
    const std::vector<size_t>& granularity_options = {1024, 2048, 3072, 4096}
  ) {
    assert(
      std::is_sorted(granularity_options.begin(), granularity_options.end()) &&
      "granularity_options should be sorted"
    );

    if (prompt_size > maxSeqLength()) {
      throw std::invalid_argument(
        "Prompt size " + std::to_string(prompt_size) +
        " exceeds max sequence length " + std::to_string(maxSeqLength()) +
        ". Set a higher value with the hybrid_opt_max_seq_length option in the "
        "session options"
      );
    }

    for (size_t option : granularity_options) {
      if (prompt_size <= option) {
        return option;
      }
    }

    return maxSeqLength();
  }

  bool static getPrefillBufferRelease(size_t npu_kernel_size) {
    // For smaller prompt sizes, freeing buffer doesnt
    // impact peak memory that much, so just retain it
    return npu_kernel_size > 1024;
  }

  static inline int runCommandSilently(
    const std::string& cmd, Ort::Logger& ort_logger
  );

  void initializeDynamicDpm(
    const std::unordered_map<std::string, std::string>& session_configs
  );

  void manageDynamicDpmState();

  void resetDpmToDefault();

  void setMladfVersion(const Ort::ConstKernelInfo& info);
  const MladfVersion& mladfVersion() const;

  bool hybrid_opt_enable_dynamic_dpm_ = false;
  Ort::Logger logger_{nullptr};

  void setPreemption(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  bool preemption() const;

  void setQos(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  const std::map<std::string, uint32_t>& qos() const;

  void setPdiName(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  const std::string& pdiName() const;

  std::map<std::string, std::any> getCommonAttrs() const;
  SharedWeights shared_weights_;
  SharedBuffer::Client shared_buffer_;

  void updateAttrsForSharedWeights(
    std::map<std::string, std::any>& attrs
  ) const;

 private:
  bool preemption_ = false;
  std::map<std::string, uint32_t> qos_map_;
  std::string pdi_name_ = "";
  bool free_after_prefill_ = false;
  std::string node_name_;
  std::string last_node_name_ = "";
  std::string global_last_node_name_ = "";
  std::filesystem::path external_data_;
  bool continue_on_exception_ = false;
  size_t max_seq_length_ = 4096;
  float dynamic_jit_ = 1.0;
  bool already_init_dynamic_dpm_ = false;
  MladfVersion mladf_version_;

  struct State {
    bool performance_mode_set = false;
    bool performance_mode_is_reset = false;
  };

  SessionState<State> ss_;

 protected:
  const std::string session_id_;
};

/**
 * @brief This function updates the dynamic JIT buffers based on the dynamic
 * JIT factor, the current buffer size and the new buffer size.
 *
 * @param scale_factor the dynamic JIT scale factor
 * @param buffer_data reference to the buffer
 * @param new_size the requested buffer size
 * @param max_size max buffer size for this operator
 */
void updateJitBuffer(
  float scale_factor, RyzenMM::BufferRef& buffer_data, size_t new_size,
  size_t max_size
);

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_NPU_OP
