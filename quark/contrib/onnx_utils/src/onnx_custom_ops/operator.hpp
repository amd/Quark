// Copyright (c) 2024 Advanced Micro Devices, Inc.

#ifndef GUARD_ONNX_CUSTOM_OPS_OPERATOR
#define GUARD_ONNX_CUSTOM_OPS_OPERATOR

#include <deque>
#include <filesystem>
#include <mutex>

#include "execution_provider.hpp"
#include "onnxruntime_lite_custom_op.h"

namespace ryzenai::onnx_utils {

extern const char* ExecutionProvider;

template <typename Kernel, const char* Name>
struct Operator : Ort::CustomOpBase<Operator<Kernel, Name>, Kernel> {
  Operator(
    Ort::ConstSessionOptions session_options,
    std::unordered_set<std::string> keys = {}
  )
    : session_options_(std::move(session_options)),
      session_options_keys_(std::move(keys)) {}

  virtual ~Operator() = default;

  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new Kernel(api, info, buildSessionConfigs());
  };

  constexpr const char* GetName() const noexcept { return Name; }

  const char* GetExecutionProviderType() const noexcept {
    return ExecutionProvider;
  }

  virtual size_t GetInputTypeCount() const noexcept { return 1; }

  virtual size_t GetOutputTypeCount() const noexcept { return 1; }

  virtual ONNXTensorElementDataType
  GetInputType(size_t /* index */) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  virtual ONNXTensorElementDataType
  GetOutputType(size_t /* index */) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  virtual OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  virtual OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return false;  // heterogenous
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return false;  // heterogeneous
  }

 private:
  std::unordered_map<std::string, std::string> buildSessionConfigs() const {
    std::unordered_map<std::string, std::string> session_configs;

    for (const auto& key : session_options_keys_) {
      const auto epkey = "ep.ryzenailightexecutionprovider." + key;
      const auto epkey_vitisai = "ep.vitisaiexecutionprovider." + key;

      if (session_options_.HasConfigEntry(epkey.c_str())) {
        session_configs[key] = session_options_.GetConfigEntry(epkey.c_str());
      } else if (session_options_.HasConfigEntry(epkey_vitisai.c_str())) {
        session_configs[key] =
          session_options_.GetConfigEntry(epkey_vitisai.c_str());
      } else if (session_options_.HasConfigEntry(key.c_str())) {
        session_configs[key] = session_options_.GetConfigEntry(key.c_str());
      } else {
        session_configs[key] = "";
      }
    }

    // in ep mode absolute paths are not constructed on oga side
    if (session_options_.HasConfigEntry("model_root")) {
      const auto model_root = session_options_.GetConfigEntry("model_root");
      const auto make_path_model_relative = [&](const char* key) {
        auto it = session_configs.find(key);

        if (it == session_configs.end() || it->second.empty()) return;

        it->second = (std::filesystem::path{model_root} / it->second)
                       .lexically_normal()
                       .string();
      };

      make_path_model_relative("external_data_file");
    }

    {  // "session_id" must be set because SessionState relies on it
      auto session_id = std::string{"default"};

      if (session_options_.HasConfigEntry("session_id"))
        // if provided from outside, let's use it
        session_id = session_options_.GetConfigEntry("session_id");
      else if (const auto ep = GetCurrentExecutionProvider())
        // EP must have on if EP flow is available
        session_id = ep->GetSessionID();

      session_configs["session_id"] = std::move(session_id);
    }

    // this is not working for some reason
    // GetSessionConfigs(this->session_configs_, session_options);

    return session_configs;
  }

 private:
  const Ort::ConstSessionOptions session_options_;
  const std::unordered_set<std::string> session_options_keys_;
};

template <typename T>
static T* GetCustomOp(const Ort::ConstSessionOptions& options) {
  static std::mutex mtx;
  static std::deque<std::unique_ptr<T>> instances;
  std::lock_guard<std::mutex> lock(mtx);
  instances.push_back(std::make_unique<T>(options));
  return instances.back().get();
}

}  // namespace ryzenai::onnx_utils

#endif  // GUARD_ONNX_CUSTOM_OPS_OPERATOR
