// Copyright (c) 2025 Advanced Micro Devices, Inc.

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <ryzenai/ryzen_mm.h>

#include <iomanip>
#include <iostream>
#include <mutex>

#include "execution_provider.hpp"
#include "execution_provider_cpugate.hpp"
#include "onnxruntime/core/providers/shared_library/provider_api.h"
#include "onnxruntime/core/providers/shared_library/provider_bridge_provider.cc"
#include "onnxruntime_cxx_api.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

static_assert(
  ORT_API_VERSION >= 23,
  "RyzenAI EP requires at least ORT 1.23: Upgrade ORT or disable the EP"
);

using namespace onnxruntime;

namespace onnxruntime {
// ORT 1.22: for some reasons it's not exported as should
// since we override the method and never invoke the base,
// it's safe to put abort() here.
common::Status IExternalDataLoader::LoadTensor(
  const Env& env, const std::filesystem::path& data_file_path,
  FileOffsetType data_offset, SafeInt<size_t> data_length, Tensor& tensor
) const {
  abort();
}
}  // namespace onnxruntime

namespace ryzenai {

static constexpr auto domain_name_ = "com.ryzenai";
static constexpr auto tracing_env_key_ = "RYZENAI_EP_TRACING";
static constexpr auto performance_counters_env_key_ =
  "RYZENAI_EP_PERFORMANCE_COUNTERS";
static constexpr auto raiep_prefix_ = "ep.ryzenailightexecutionprovider.";
static constexpr auto vaiep_prefix_ = "ep.vitisaiexecutionprovider.";
static constexpr auto memlimit_key_ = "memory_limit_mb";
static constexpr auto token_backend_key_ = "hybrid_opt_token_backend";
static constexpr auto vendor_name_ = "AMD";
static constexpr auto vendor_id_amd_ = 0x1002;
static constexpr auto vendor_id_amd_xilinx_ = 0x1022;

static const OrtApi* api_ = nullptr;

struct Allocator : OrtAllocator {
  Allocator(RyzenMM::Allocator allocator, Ort::ConstMemoryInfo mi)
    : allocator_(std::move(allocator)), mi_(std::move(mi)) {
    static_cast<OrtAllocator&>(*this) = {ORT_API_VERSION};

    OrtAllocator::Alloc = &Allocator::Alloc;
    OrtAllocator::Free = &Allocator::Free;
    OrtAllocator::Info = &Allocator::Info;
    OrtAllocator::Reserve = &Allocator::Reserve;
  }

  static void* Alloc(struct OrtAllocator* this_ptr, size_t size) {
    return size ? static_cast<Allocator*>(this_ptr)
                    ->allocator_.AllocateBuffer(size)
                    .CreateUnmanagedBuffer()
                : nullptr;
  }

  static void Free(struct OrtAllocator* this_ptr, void* ptr) {
    if (nullptr != ptr) ryzenai::RyzenMM::FreeUnmanagedBuffer(ptr);
  };

  static const struct OrtMemoryInfo* Info(const struct OrtAllocator* this_ptr) {
    return static_cast<const Allocator*>(this_ptr)->mi_;
  }

  static void* Reserve(struct OrtAllocator* this_ptr, size_t size) {
    return Alloc(this_ptr, size);
  }

 private:
  const RyzenMM::Allocator allocator_;
  const Ort::ConstMemoryInfo mi_;
};

struct Ep : OrtEp, ryzenai::IExecutionProvider {
  Ep(const OrtApi* api, Ort::ConstSessionOptions session_options)
    : api_(api),
      ep_api_(api->GetEpApi()),
      session_options_(std::move(session_options)) {
    static_cast<OrtEp&>(*this) = {ORT_API_VERSION};

    OrtEp::GetName = [](auto) noexcept { return ONNX_UTILS_RYZENAI_EP_NAME; };

    OrtEp::GetCapability =
      [](auto self, auto graph, auto graph_support_info) noexcept {
        return static_cast<Ep*>(self)->GetCapability(graph, graph_support_info);
      };

    OrtEp::Compile = [](
                       auto self, auto graphs, auto fused_nodes, auto count,
                       auto node_compute_infos, auto ep_context_nodes
                     ) noexcept {
      return static_cast<Ep*>(self)->Compile(
        graphs, fused_nodes, count, node_compute_infos, ep_context_nodes
      );
    };

    OrtEp::ReleaseNodeComputeInfos = [](
                                       auto self, auto node_compute_infos,
                                       auto num_node_compute_infos
                                     ) noexcept {
      return static_cast<Ep*>(self)->ReleaseNodeComputeInfos(
        node_compute_infos, num_node_compute_infos
      );
    };

    OrtEp::OnRunStart = [](auto self, auto run_options) noexcept {
      return static_cast<Ep*>(self)->OnRunStart(run_options);
    };

    OrtEp::OnRunEnd =
      [](auto self, auto run_options, auto sync_stream) noexcept {
        return static_cast<Ep*>(self)->OnRunEnd(run_options, sync_stream);
      };

    OrtEp::CreateAllocator =
      [](auto self, auto meminfo, auto allocator) noexcept {
        return static_cast<Ep*>(self)->CreateAllocator(meminfo, allocator);
      };

    if (const auto opt = GetSessionOptionValue(memlimit_key_)) {
      const auto val = std::atol(opt->c_str());

      RyzenMM::Config::SetMemoryLimit(static_cast<size_t>(val) * 1024 * 1024);
    }

    CurrentThreadInstance.store(this, std::memory_order_relaxed);

    if (tracing_enabled_) {
      std::cout << "RyzenAI EP Session " << session_id_ << " created."
                << std::endl;

      if (OrtKeyValuePairs* entries = nullptr;
          nullptr ==
          api_->GetSessionOptionsConfigEntries(session_options_, &entries)) {
        const char* const* keys = nullptr;
        const char* const* vals = nullptr;
        size_t cnt = 0;

        api_->GetKeyValuePairs(entries, &keys, &vals, &cnt);

        for (auto i = 0u; i < cnt; ++i) {
          std::cout << "\t" << keys[i] << " = " << vals[i] << std::endl;
        }

        api_->ReleaseKeyValuePairs(entries);
      }
    }
  }

  ~Ep() {
    if (tracing_enabled_) {
      std::cout << "RyzenAI EP Session " << session_id_ << " destroyed."
                << std::endl;
    }

    CurrentThreadInstance.store(nullptr, std::memory_order_relaxed);
  }

  inline static thread_local std::atomic<Ep*> CurrentThreadInstance{nullptr};

  RYZENAI_EP_NOINLINE OrtStatus* GetCapability(
    const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info
  ) noexcept {
    size_t num_nodes = 0;

    if (auto s = api_->Graph_GetNumNodes(graph, &num_nodes)) return s;

    std::vector<const OrtNode*> nodes(num_nodes);

    if (auto s = api_->Graph_GetNodes(graph, nodes.data(), nodes.size()))
      return s;

    for (const auto n : nodes) {
      const char *name = nullptr, *op_type = nullptr, *domain = nullptr;
      size_t node_id = 0, num_inputs = 0, num_outputs = 0;

      if (auto s = api_->Node_GetName(n, &name)) return s;
      if (auto s = api_->Node_GetOperatorType(n, &op_type)) return s;
      if (auto s = api_->Node_GetDomain(n, &domain)) return s;
      if (auto s = api_->Node_GetId(n, &node_id)) return s;
      if (auto s = api_->Node_GetNumInputs(n, &num_inputs)) return s;
      if (auto s = api_->Node_GetNumOutputs(n, &num_outputs)) return s;

      auto& ni = snapshot_->nodes[std::string{name ? name : ""}];

      ni.index = node_id;
      ni.op_type = op_type ? op_type : "";
      ni.domain = domain ? domain : "";

      if (num_inputs) {
        std::vector<const OrtValueInfo*> inputs(num_inputs);

        if (auto s = api_->Node_GetInputs(n, inputs.data(), inputs.size()))
          return s;

        for (size_t i = 0; i < num_inputs; ++i) {
          const OrtValueInfo* vi = inputs[i];

          if (nullptr == vi) continue;

          const OrtNode* producer_node = nullptr;
          size_t producer_out_idx = 0;

          if (auto s = api_->ValueInfo_GetValueProducer(
                vi, &producer_node, &producer_out_idx
              ))
            return s;

          if (producer_node) {
            const char* prod_name = nullptr;

            if (auto s = api_->Node_GetName(producer_node, &prod_name))
              return s;

            ni.inputs[std::string{prod_name ? prod_name : ""}] = std::make_pair(
              static_cast<int>(producer_out_idx), static_cast<int>(i)
            );
          }
        }
      }

      if (num_outputs) {
        std::vector<const OrtValueInfo*> outputs(num_outputs);

        if (auto s = api_->Node_GetOutputs(n, outputs.data(), outputs.size()))
          return s;

        for (size_t j = 0; j < num_outputs; ++j) {
          const OrtValueInfo* vo = outputs[j];

          if (nullptr == vo) continue;

          size_t num_consumers = 0;

          if (auto s = api_->ValueInfo_GetValueNumConsumers(vo, &num_consumers))
            return s;

          if (num_consumers) {
            std::vector<const OrtNode*> consumers(num_consumers);
            std::vector<int64_t> input_indices(num_consumers);

            if (auto s = api_->ValueInfo_GetValueConsumers(
                  vo, consumers.data(), input_indices.data(), num_consumers
                ))
              return s;

            for (size_t k = 0; k < num_consumers; ++k) {
              const char* cons_name = nullptr;

              if (auto s = api_->Node_GetName(consumers[k], &cons_name))
                return s;

              ni.outputs[std::string{cons_name ? cons_name : ""}] =
                std::make_pair(
                  static_cast<int>(j), static_cast<int>(input_indices[k])
                );
            }
          }
        }
      }

      if (domain && std::strcmp(domain, domain_name_) == 0) {
        if (auto s =
              ep_api_->EpGraphSupportInfo_AddSingleNode(graph_support_info, n))
          return s;
      }
    }

    // snap.Dump();
    return nullptr;
  }

  OrtStatus* Compile(
    const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
    OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes
  ) noexcept {
    *node_compute_infos = nullptr;
    *ep_context_nodes = nullptr;

    return nullptr;
  }

  void ReleaseNodeComputeInfos(
    OrtNodeComputeInfo** node_compute_infos, size_t num_node_compute_infos
  ) noexcept {}

  OrtStatus* OnRunStart(const ::OrtRunOptions* run_options) noexcept {
    // simlation of absence of 'IExecutionProvider::OnSessionInitializationEnd'
    if (0 == runs_.fetch_add(1, std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(cpu_gate_guard_);
      if (cpu_gate_) cpu_gate_->Initialize();
    }

    return nullptr;
  }

  OrtStatus* OnRunEnd(
    const ::OrtRunOptions* run_options, _In_ bool sync_stream
  ) noexcept {
    return nullptr;
  }

  OrtStatus* CreateAllocator(
    const OrtMemoryInfo* meminfo, OrtAllocator** allocator
  ) noexcept {
    if (IsGPUMemoryRequired())
      *allocator = new Allocator(
        ryzenai::RyzenMM::GPUAllocator<'REPG'>(), Ort::ConstMemoryInfo(meminfo)
      );
    else
      *allocator = new Allocator(
        ryzenai::RyzenMM::NPUAllocator<'REPN'>(), Ort::ConstMemoryInfo(meminfo)
      );

    return nullptr;
  }

  bool IsGPUMemoryRequired() const {
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
    bool gpu_prefill = false;
#else
    bool gpu_prefill = true;
#endif

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
    bool gpu_token = true;

    if (const auto val = GetSessionOptionValue(token_backend_key_)) {
      if ("npu" == val.value()) gpu_token = false;
    }
#else
    bool gpu_token = false;
#endif

    return gpu_prefill || gpu_token;
  }

  // ryzenai::IExecutionProvider impl

  const std::string& GetSessionID() const override { return session_id_; }

  CPUGate::Interface& GetCPUGate() override {
    std::lock_guard<std::mutex> lock(cpu_gate_guard_);

    if (!cpu_gate_) cpu_gate_ = CPUGate::CreateInstance();

    return *cpu_gate_;
  }

  static std::string _GetEnvString(const char* key) {
    const auto* val = std::getenv(key);
    return val == nullptr ? "" : std::string{val};
  }

  static bool _IsFeatureEnabledViaEnvironment(const char* key) {
    auto value = _GetEnvString(key);
    // this will not work for encodings with multi-byte characters
    std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return std::tolower(c); }
    );

    if (value.empty() || value == "no" || value == "false" || value == "0") {
      return false;
    }

    return true;
  }

  bool IsPerformanceCountersEnabled() override {
    static const bool enabled =
      _IsFeatureEnabledViaEnvironment(performance_counters_env_key_);
    return enabled;
  }

  void RecordDuration(Metric m, Clock::duration d) override {
    const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();

    performance_metrics_[size_t(m)].fetch_add(ns, std::memory_order_relaxed);
  }

  void QueryPerformanceCounters(
    const char* const** names, uint32_t* count, uint64_t* values
  ) const override {
    if (names) *names = (const char* const*)MetricNames;
    if (count) *count = uint32_t(Metric::Max);
    if (values)
      for (auto i = 0; i < uint32_t(Metric::Max); i++)
        values[i] = performance_metrics_[i].load(std::memory_order_relaxed);
  }

  std::optional<std::string> GetSessionOptionValue(std::string key) const {
    const auto rai_key = raiep_prefix_ + key;
    const auto vai_key = vaiep_prefix_ + key;

    if (session_options_.HasConfigEntry(rai_key.c_str())) {
      return session_options_.GetConfigEntry(rai_key.c_str());
    } else if (session_options_.HasConfigEntry(vai_key.c_str())) {
      return session_options_.GetConfigEntry(vai_key.c_str());
    }

    return std::nullopt;
  }

 private:
  const OrtApi* api_{nullptr};
  const OrtEpApi* ep_api_{nullptr};

  const Ort::ConstSessionOptions session_options_;

  inline static std::atomic<unsigned> next_session_id_{0};
  const std::string session_id_{std::to_string(next_session_id_++)};

  const bool tracing_enabled_ =
    _IsFeatureEnabledViaEnvironment(tracing_env_key_);

  std::shared_ptr<CPUGate::Interface> cpu_gate_;
  std::mutex cpu_gate_guard_;

  std::array<std::atomic_uint64_t, size_t(Metric::Max)> performance_metrics_;

  std::atomic_uint64_t runs_{0};

 private:
  struct Snapshot {
    struct NodeInfo {
      size_t index;
      std::string op_type, domain;
      std::unordered_map<std::string, std::pair<int, int>> inputs, outputs;
    };

    std::unordered_map<std::string, NodeInfo> nodes;

    void Dump() const {
      std::cout << "================ Snapshot Dump (sorted by index) "
                   "================\n";

      std::vector<std::pair<std::string, const Snapshot::NodeInfo*>> sorted;
      sorted.reserve(nodes.size());
      for (const auto& [name, ni] : nodes) sorted.emplace_back(name, &ni);

      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second->index < b.second->index;
      });

      for (const auto& [name, ni_ptr] : sorted) {
        const auto& ni = *ni_ptr;
        std::cout << "Node[" << std::setw(3) << ni.index << "] "
                  << (name.empty() ? "<unnamed>" : name) << "\n";
        std::cout << "  OpType : " << ni.op_type << "\n";
        std::cout << "  Domain : " << ni.domain << "\n";

        if (!ni.inputs.empty()) {
          std::cout << "  Inputs:\n";
          for (const auto& [src_name, pair] : ni.inputs) {
            std::cout << "    from "
                      << (src_name.empty() ? "<unnamed>" : src_name)
                      << " (src_out=" << pair.first
                      << ", dst_in=" << pair.second << ")\n";
          }
        }

        if (!ni.outputs.empty()) {
          std::cout << "  Outputs:\n";
          for (const auto& [dst_name, pair] : ni.outputs) {
            std::cout << "    to "
                      << (dst_name.empty() ? "<unnamed>" : dst_name)
                      << " (src_out=" << pair.first
                      << ", dst_in=" << pair.second << ")\n";
          }
        }

        std::cout << "---------------------------------------------------------"
                     "------\n";
      }

      std::cout << "Total nodes: " << sorted.size() << "\n";
      std::cout
        << "================================================================\n";
    }
  };

  mutable std::shared_ptr<Snapshot> snapshot_ = std::make_shared<Snapshot>();

  friend struct ExecutionProvider;
};

struct EpFactory : OrtEpFactory {
  EpFactory(const OrtApi* api) : api_(api), ep_api_(api_->GetEpApi()) {
    static_cast<OrtEpFactory&>(*this) = {ORT_API_VERSION};

    OrtEpFactory::GetName = [](auto) noexcept {
      return ONNX_UTILS_RYZENAI_EP_NAME;
    };

    OrtEpFactory::GetVendor = [](auto) noexcept { return vendor_name_; };

    OrtEpFactory::GetSupportedDevices =
      [](
        auto self, auto devices, auto num_devices, auto ep_devices,
        auto max_ep_devices, auto num_ep_devices
      ) noexcept {
        return static_cast<EpFactory*>(self)->GetSupportedDevices(
          devices, num_devices, ep_devices, max_ep_devices, num_ep_devices
        );
      };

    OrtEpFactory::CreateEp = [](
                               auto self, auto devices, auto ep_metadata_pairs,
                               auto num_devices, auto session_options,
                               auto logger, auto ep
                             ) noexcept {
      return static_cast<EpFactory*>(self)->CreateEp(
        devices, ep_metadata_pairs, num_devices, session_options, logger, ep
      );
    };

    OrtEpFactory::ReleaseEp = [](auto self, auto ep) noexcept {
      return static_cast<EpFactory*>(self)->ReleaseEp(ep);
    };

    OrtEpFactory::GetVendorId = [](auto) noexcept -> uint32_t {
      return vendor_id_amd_;  // AMD
    };

    OrtEpFactory::GetVersion = [](auto) noexcept -> const char* {
      return "1.6.0";
    };

    OrtEpFactory::CreateAllocator =
      [](auto self, auto mi, auto kv, auto allocator) noexcept -> OrtStatus* {
      return static_cast<EpFactory*>(self)->CreateAllocator(mi, kv, allocator);
    };

    OrtEpFactory::ReleaseAllocator = [](auto self, auto allocator) noexcept {
      return static_cast<EpFactory*>(self)->ReleaseAllocator(allocator);
    };

    OrtEpFactory::CreateDataTransfer = [](auto self, auto dt) noexcept {
      return static_cast<EpFactory*>(self)->CreateDataTransfer(dt);
    };

    OrtEpFactory::IsStreamAware = [](auto) noexcept { return false; };
  }

  RYZENAI_EP_NOINLINE OrtStatus* GetSupportedDevices(
    const OrtHardwareDevice* const* devices, size_t num_devices,
    OrtEpDevice** ep_devices, size_t max_ep_devices, size_t* num_ep_devices
  ) noexcept {
    size_t out_count = 0;

    for (auto i = 0; i < num_devices; i++) {
      const OrtHardwareDevice* hw = devices[i];
      const auto typ = api_->HardwareDevice_Type(hw);
      const auto vid = api_->HardwareDevice_VendorId(hw);

      const auto supported =
        (OrtHardwareDeviceType::OrtHardwareDeviceType_GPU == typ &&
         vendor_id_amd_ == vid)  // AMD
        || (OrtHardwareDeviceType::OrtHardwareDeviceType_NPU == typ &&
            vendor_id_amd_xilinx_ == vid)  // AMD-Xilinx
        ;

      if (!supported) continue;

      if (out_count == max_ep_devices)
        return api_->CreateStatus(
          ORT_FAIL, "GetSupportedDevices: output buffer too small"
        );

      OrtEpDevice* ep_dev = nullptr;

      if (auto st =
            ep_api_->CreateEpDevice(this, hw, nullptr, nullptr, &ep_dev)) {
        return st;
      }

      static OrtMemoryInfo *meminfo_rw{nullptr}, *meminfo_ro{nullptr},
        *meminfo_ha{nullptr};

      if (!meminfo_rw) {
        // TODO: make it non-leaking
        api_->CreateMemoryInfo_V2(
          "RMM", OrtMemoryInfoDeviceType_NPU, vendor_id_amd_xilinx_, 0,
          OrtDeviceMemoryType_DEFAULT, 0, OrtAllocatorType::OrtDeviceAllocator,
          &meminfo_rw
        );

        api_->CreateMemoryInfo_V2(
          "RMM-RO", OrtMemoryInfoDeviceType_NPU, vendor_id_amd_xilinx_, 0,
          OrtDeviceMemoryType_DEFAULT, 0,
          OrtAllocatorType::OrtReadOnlyAllocator, &meminfo_ro
        );

        api_->CreateMemoryInfo_V2(
          "RMM-HA", OrtMemoryInfoDeviceType_NPU, vendor_id_amd_xilinx_, 0,
          OrtDeviceMemoryType_HOST_ACCESSIBLE, 0,
          OrtAllocatorType::OrtDeviceAllocator, &meminfo_ha
        );
      }

      ep_api_->EpDevice_AddAllocatorInfo(ep_dev, meminfo_rw);
      ep_api_->EpDevice_AddAllocatorInfo(ep_dev, meminfo_ro);
      ep_api_->EpDevice_AddAllocatorInfo(ep_dev, meminfo_ha);

      ep_devices[out_count++] = ep_dev;
    }

    *num_ep_devices = out_count;
    return nullptr;
  }

  OrtStatus* CreateEp(
    const OrtHardwareDevice* const*, const OrtKeyValuePairs* const*, size_t,
    const OrtSessionOptions* session_options, const OrtLogger*, OrtEp** ep
  ) noexcept {
    // TODO: double check that it's not really needed and remove this hack
    api_->AddSessionConfigEntry(
      (OrtSessionOptions*)session_options, "custom_allocator", "RMM"
    );

    *ep = new Ep(api_, Ort::ConstSessionOptions(session_options));

    ep_instances_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  void ReleaseEp(struct OrtEp* ep) noexcept {
    delete static_cast<Ep*>(ep);

    if (1 == ep_instances_.fetch_sub(1, std::memory_order_relaxed)) {
      ryzenai::RyzenMM::Cleanup();
    }
  }

  OrtStatus* CreateAllocator(
    const OrtMemoryInfo* mi, const OrtKeyValuePairs*, OrtAllocator** allocator
  ) noexcept {
    *allocator = new Allocator(
      ryzenai::RyzenMM::NPUAllocator<'REPN'>(), Ort::ConstMemoryInfo(mi)
    );

    return nullptr;
  }

  void ReleaseAllocator(OrtAllocator* allocator) noexcept {
    delete static_cast<Allocator*>(allocator);
  }

  OrtStatus* CreateDataTransfer(OrtDataTransferImpl** data_transfer) noexcept {
    auto* impl = new OrtDataTransferImpl{ORT_API_VERSION};

    impl->Release = [](OrtDataTransferImpl* p) noexcept { delete p; };
    impl->CanCopy = [](auto, auto, auto) noexcept -> bool { return true; };
    impl->CopyTensors = [](
                          auto, const OrtValue** src_tensors,
                          OrtValue** dst_tensors, OrtSyncStream** streams,
                          size_t num_tensors
                        ) noexcept -> OrtStatus* {
      for (size_t i = 0; i < num_tensors; ++i) {
        const void* src_data = nullptr;
        void* dst_data = nullptr;
        size_t bytes;

        ryzenai::api_->GetTensorData(src_tensors[i], &src_data);
        ryzenai::api_->GetTensorMutableData(dst_tensors[i], &dst_data);
        ryzenai::api_->GetTensorSizeInBytes(src_tensors[i], &bytes);

        memcpy(dst_data, src_data, bytes);
      }

      return nullptr;
    };

    *data_transfer = impl;
    return nullptr;
  }

 private:
  const OrtApi* api_{nullptr};
  const OrtEpApi* ep_api_{nullptr};

  std::atomic_uint64_t ep_instances_{0};
};

struct ExecutionProvider : Ep, onnxruntime::IExecutionProvider {
  ExecutionProvider(const OrtSessionOptions& session_options)
    : Ep(ryzenai::api_, Ort::ConstSessionOptions(&session_options)),
      onnxruntime::IExecutionProvider(ONNX_UTILS_RYZENAI_EP_NAME) {}

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
    const onnxruntime::GraphViewer& gv, const IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& graph_optimizer_registry,
    IResourceAccountant* resource_accountant = nullptr
  ) const override {
    auto& snap = *snapshot_;

    std::vector<std::unique_ptr<ComputeCapability>> caps;

    for (const auto& node : gv.Nodes()) {
      auto& ni = snap.nodes[node.Name()];

      ni.index = node.Index();
      ni.op_type = node.OpType();
      ni.domain = node.Domain();

      for (auto e = node.InputEdgesBegin(); e != node.InputEdgesEnd(); ++e)
        ni.inputs[e->GetNode().Name()] =
          std::make_pair(e->GetSrcArgIndex(), e->GetDstArgIndex());

      for (auto e = node.OutputEdgesBegin(); e != node.OutputEdgesEnd(); ++e)
        ni.outputs[e->GetNode().Name()] =
          std::make_pair(e->GetSrcArgIndex(), e->GetDstArgIndex());

      if ("com.ryzenai" == ni.domain) {
        auto sub = IndexedSubGraph::Create();
        sub->Nodes().push_back(node.Index());
        caps.emplace_back(ComputeCapability::Create(std::move(sub)));
      }
    }

    return caps;
  }

  common::Status OnSessionInitializationEnd() override {
    GetCPUGate().Initialize();
    return Status::OK();
  }

  template <typename T>
  struct Allocator : IAllocator, T {
    using IAllocator::IAllocator;

    void* Alloc(size_t size) override {
      return (0 == size) ? nullptr
                         : T::AllocateBuffer(size).CreateUnmanagedBuffer();
    }

    void Free(void* p) override {
      if (nullptr != p) ryzenai::RyzenMM::FreeUnmanagedBuffer(p);
    }
  };

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    OrtMemoryInfo mi{"RMM", OrtAllocatorType::OrtDeviceAllocator};

    if (IsGPUMemoryRequired())
      return {
        std::make_shared<Allocator<ryzenai::RyzenMM::GPUAllocator<'REPG'>>>(mi)
      };
    else
      return {
        std::make_shared<Allocator<ryzenai::RyzenMM::NPUAllocator<'REPN'>>>(mi)
      };
  }

  std::unique_ptr<onnxruntime::IExternalDataLoader>
  GetExternalDataLoader() const override {
    struct ExternalDataLoader : onnxruntime::IExternalDataLoader {
      bool CanLoad(const OrtMemoryInfo&) const override { return true; }

      common::Status LoadTensor(
        const Env& env, const std::filesystem::path& data_file_path,
        FileOffsetType data_offset, SafeInt<size_t> data_length, Tensor& tensor
      ) const override {
        return env.ReadFileIntoBuffer(
          data_file_path.native().c_str(), data_offset, data_length,
          gsl::make_span((char*)tensor.MutableDataRaw(), tensor.SizeInBytes())
        );
      }
    };

    return std::make_unique<ExternalDataLoader>();
  }
};

IExecutionProvider* GetCurrentExecutionProvider() {
  return Ep::CurrentThreadInstance.load(std::memory_order_relaxed);
}

std::shared_ptr<CPUGate::Op> ExecutionProviderExtensions::CreateCPUOp(
  CPUGate::OpParams&& params
) {
  return ep_ ? ep_->GetCPUGate().CreateOp(std::move(params)) : nullptr;
}

namespace onnx_utils {
extern const char* ExecutionProvider;
}  // namespace onnx_utils

static std::atomic_bool ep_type_was_explicitly_set_{false};

}  // namespace ryzenai

extern "C" {
RYZENAI_EP_EXPORT_API const onnxruntime::Provider* ORT_API_CALL
GetProvider(void) {
  struct Provider : onnxruntime::Provider {
    void Initialize() override {}
    void Shutdown() override { ryzenai::RyzenMM::Cleanup(); }

    Status CreateIExecutionProvider(
      const OrtHardwareDevice* const*, const OrtKeyValuePairs* const*, size_t,
      ProviderOptions&, const OrtSessionOptions& session_options,
      const OrtLogger&, std::unique_ptr<onnxruntime::IExecutionProvider>& ep
    ) override {
      ryzenai::api_->AddSessionConfigEntry(
        (OrtSessionOptions*)&session_options, "custom_allocator", "RMM"
      );

      ep = std::make_unique<ryzenai::ExecutionProvider>(session_options);
      return Status::OK();
    }
  };

  static Provider provider;

  return &provider;
}

RYZENAI_EP_EXPORT_API OrtStatus* ORT_API_CALL CreateEpFactories(
  _In_ const char* registered_name, _In_ const OrtApiBase* ort_api_base,
  _In_ const OrtLogger* ort_logger, _Inout_ OrtEpFactory** factories,
  _In_ size_t max_factories, _Out_ size_t* num_factories
) {
  if (!ryzenai::ep_type_was_explicitly_set_)
    ryzenai::onnx_utils::ExecutionProvider = ONNX_UTILS_RYZENAI_EP_NAME;

  Ort::InitApi((ryzenai::api_ = ort_api_base->GetApi(ORT_API_VERSION)));

  *factories = new ryzenai::EpFactory(ryzenai::api_);
  *num_factories = 1;
  return nullptr;
}

RYZENAI_EP_EXPORT_API OrtStatus* ORT_API_CALL
ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<ryzenai::EpFactory*>(factory);
  return nullptr;
}

RYZENAI_EP_EXPORT_API void RyzenAI_SetExecutionProviderType(
  const char* ep_name
) {
  ryzenai::onnx_utils::ExecutionProvider = ep_name;
  ryzenai::ep_type_was_explicitly_set_ = true;
}

RYZENAI_EP_EXPORT_API void RyzenAI_SetSessionOptions(
  OrtSessionOptions* options
) {
  ryzenai::api_->AddSessionConfigEntry(
    options, "session.inter_op.allow_spinning", "0"
  );
  ryzenai::api_->AddSessionConfigEntry(
    options, "session.intra_op.allow_spinning", "0"
  );
}

RYZENAI_EP_EXPORT_API void RyzenAI_Shutdown() { ryzenai::RyzenMM::Cleanup(); }

RYZENAI_EP_EXPORT_API bool
RyzenAI_QueryCurrentExecutionProviderPerformanceCounters(
  const char* const** names, uint32_t* count, uint64_t* values
) {
  if (const auto ep =
        ryzenai::Ep::CurrentThreadInstance.load(std::memory_order_relaxed)) {
    ep->QueryPerformanceCounters(names, count, values);
    return true;
  }

  return false;
}
}  // extern "C"
