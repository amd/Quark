// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "npu_op.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "../../custom_ops.hpp"
#include "external_data.hpp"
#include "npu_utils.hpp"
#include "onnxruntime_cxx_api.h"
#include "ort.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

NpuOp::NpuOp(
  const OrtKernelInfo* kernel_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : ss_(session_configs), session_id_(session_configs.at("session_id")) {
  Ort::ConstKernelInfo info{kernel_info};
  node_name_ = info.GetNodeName();
  logger_ = info.GetLogger();
}

NpuOp::~NpuOp() { resetDpmToDefault(); }

std::shared_ptr<proto::Header> NpuOp::initializeNpuOp(
  const char* op_type,
  const std::unordered_map<std::string, std::string>& session_configs,
  Ort::ConstKernelInfo& info
) {
  shared_buffer_ = SharedBuffer::Client(op_type, session_id_, name());

  initializeDynamicDpm(session_configs);
  setMaxSeqLength(session_configs);
  setDynamicJitFactor(session_configs);
  setContinueOnException(session_configs);
  setPreemption(session_configs);
  setQos(session_configs);
  setPdiName(session_configs);
  setMladfVersion(info);

  auto header = setExternalData(session_configs);
  setFreeAfterPrefill(session_configs, header.get(), op_type);
  setLastNode(header.get());

  Lora::enableLora(session_configs);

  auto model_key = getAttribute<std::string>(info, "model_hash", "");
  shared_weights_.setModelKey(model_key);

  return header;
}

const std::string& NpuOp::name() const { return node_name_; }

void NpuOp::setFreeAfterPrefill(
  const std::unordered_map<std::string, std::string>& session_configs,
  const proto::Header* header, std::string_view op_type
) {
  free_after_prefill_ =
    session_configs.at("hybrid_opt_free_after_prefill") == "1";
  last_node_name_ = header->op_metadata().at(op_type).last();
}

bool NpuOp::freeAfterPrefill(std::string_view node_name) const {
  return free_after_prefill_ && node_name == last_node_name_;
}

void NpuOp::setLastNode(const proto::Header* header) {
  auto& last_operators = (header->layers().rbegin())->operators();
  auto it = last_operators.rbegin();
  global_last_node_name_ = it != last_operators.rend() ? *it : "";
}

bool NpuOp::lastNode(std::string_view node_name) {
  return node_name == global_last_node_name_;
}

void NpuOp::setDynamicJitFactor(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_dynamic_jit_factor");
  if (!value.empty()) {
    float user_value;
    try {
      user_value = std::stof(value);
    } catch (const std::invalid_argument&) {
      user_value = 1.0;
    }
    if (user_value >= 0.0 && user_value <= 1.0) {
      dynamic_jit_ = user_value;
    }
  }
}
float NpuOp::dynamicJitFactor() const { return dynamic_jit_; }

std::shared_ptr<proto::Header> NpuOp::setExternalData(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto header = proto::getHeader(session_configs);

  const auto& external_data_file = session_configs.at("external_data_file");
  if (header->external_data().npu()) {
    external_data_ = fs::path(external_data_file).parent_path() /
                     header->external_data().filename();
  }

  return header;
}

const fs::path& NpuOp::externalData() const { return external_data_; }

bool NpuOp::useExternalData() const {
  return !external_data_.empty() && !shared_weights_.ready();
}

size_t NpuOp::getInitPromptSize(
  const std::unordered_map<std::string, std::string>& session_configs
) const {
  size_t init_prompt_size = kSeqLengthDefault;
  if (session_configs.count("hybrid_opt_init_prompt_size") &&
      session_configs.at("hybrid_opt_init_prompt_size") != "") {
    int prompt_size =
      std::stoi(session_configs.at("hybrid_opt_init_prompt_size"));

    if (0 < prompt_size && prompt_size <= maxSeqLength()) {
      init_prompt_size = prompt_size;
    }
  }

  return init_prompt_size;
}

void updateJitBuffer(
  float scale_factor, RyzenMM::BufferRef& buffer_data, size_t new_size,
  size_t max_size
) {
  RyzenMM::NPUAllocator<'NPJT'> allocator;
  const auto buffer_size = buffer_data ? buffer_data.Size() : 0;
  if (scale_factor > 0) {
    if (new_size > buffer_size || new_size < buffer_size * scale_factor) {
      buffer_data = {};
      buffer_data = allocator.AllocateBuffer(new_size);
    }
  } else {
    if (!buffer_data) {
      buffer_data = allocator.AllocateBuffer(max_size);
    }
  }
}

int NpuOp::runCommandSilently(const std::string& cmd, Ort::Logger& ort_logger) {
#ifdef _WIN32
  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  ZeroMemory(&pi, sizeof(pi));

  // Run via cmd.exe /C <command> to capture the usual shell semantics
  std::string command = "cmd.exe /C " + cmd;

  BOOL success = CreateProcessA(
    nullptr, command.data(), nullptr, nullptr, FALSE,
    CREATE_NO_WINDOW,  // No console window
    nullptr, nullptr, &si, &pi
  );

  if (!success) {
    DWORD error_code = GetLastError();
    std::stringstream ss;
    ss << "[runCommandSilently] CreateProcessA failed for command: " << cmd
       << ". Error code: " << error_code;
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, ss.str().c_str());
    return static_cast<int>(error_code);
  }

  // Wait until child process exits.
  WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD exit_code = 0;
  if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
    std::stringstream ss;
    ss << "[runCommandSilently] Could not get exit code for command: " << cmd;
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, ss.str().c_str());
    exit_code = static_cast<DWORD>(-1);
  }

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return static_cast<int>(exit_code);

#else
  static constexpr const char* kSilentRedirect = " >/dev/null 2>&1";
  std::string silent_cmd = cmd + kSilentRedirect;
  int result = system(silent_cmd.c_str());
  if (result != 0) {
    std::stringstream ss;
    ss << "[runCommandSilently] Command failed: " << cmd;
    ORT_CXX_LOG(ort_logger, ORT_LOGGING_LEVEL_ERROR, ss.str().c_str());
  }
  return result;
#endif
}

void NpuOp::initializeDynamicDpm(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  if (already_init_dynamic_dpm_) {
    return;
  }
  already_init_dynamic_dpm_ = true;

  auto it = session_configs.find("hybrid_opt_enable_dynamic_dpm");
  if (it != session_configs.end() && !it->second.empty()) {
    hybrid_opt_enable_dynamic_dpm_ = (it->second == "1");
  }
}

void NpuOp::manageDynamicDpmState() {
  if (!hybrid_opt_enable_dynamic_dpm_) {
    return;
  }

  if (!ss_->performance_mode_set) {
    runCommandSilently(
      "C:\\Windows\\System32\\AMD\\xrt-smi configure --pmode performance",
      this->logger_
    );
    ss_->performance_mode_set = true;
  } else if (lastNode(node_name_)) {
    runCommandSilently(
      "C:\\Windows\\System32\\AMD\\xrt-smi configure --pmode balanced",
      this->logger_
    );
    ss_->performance_mode_set = false;
  }
}

void NpuOp::resetDpmToDefault() {
  if (!ss_->performance_mode_is_reset && hybrid_opt_enable_dynamic_dpm_) {
    runCommandSilently(
      "C:\\Windows\\System32\\AMD\\xrt-smi configure --pmode default",
      this->logger_
    );
    ss_->performance_mode_is_reset = true;
  }
}

void NpuOp::setContinueOnException(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_continue_on_exception");
  continue_on_exception_ = value.empty() ? false : value == "1";
}
bool NpuOp::continueOnException() const { return continue_on_exception_; }

void NpuOp::setMaxSeqLength(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& value = session_configs.at("hybrid_opt_max_seq_length");
  if (!value.empty()) {
    auto seq_len = std::stoull(value);
    auto is_power_of_two = !(seq_len == 0) && !(seq_len & (seq_len - 1));
    if (!is_power_of_two) {
      throw std::invalid_argument(
        "hybrid_opt_max_seq_length must be a power of two"
      );
    }
    max_seq_length_ = seq_len;
  }
}
std::size_t NpuOp::maxSeqLength() const { return max_seq_length_; };

void NpuOp::setMladfVersion(const Ort::ConstKernelInfo& info) {
  std::string is_bfp16;
  try {
    is_bfp16 = info.GetAttribute<std::string>("is_bfp16");
  } catch (const Ort::Exception&) {
    mladf_version_ = "v1";
  }

  if (is_bfp16 == "weights") {
    mladf_version_ = "v2";
  } else if (is_bfp16.empty()) {
    mladf_version_ = "v1";
  } else {
    throw std::runtime_error("Unsupported is_bfp16 specified: " + is_bfp16);
  }
}

void NpuOp::setPreemption(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_enable_npu_preemption");
  preemption_ = value.empty() ? false : value == "1";
}
bool NpuOp::preemption() const { return preemption_; }

void NpuOp::setQos(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto value = session_configs.at("hybrid_opt_enable_npu_qos");
  auto qos_enable = value.empty() || !preemption_ ? false : value == "1";

  if (qos_enable) {
    qos_map_["is_preemptible"] = 1;
  }
}
const std::map<std::string, uint32_t>& NpuOp::qos() const { return qos_map_; }

const std::string& NpuOp::mladfVersion() const { return mladf_version_; }

void NpuOp::setPdiName(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  pdi_name_ = session_configs.at("hybrid_opt_npu_pdi_name");
}

const std::string& NpuOp::pdiName() const { return pdi_name_; }

std::map<std::string, std::any> NpuOp::getCommonAttrs() const {
  return {
    {"op_version", mladfVersion()},
    {"pdi_name", pdiName()},
    {"preemption", preemption()},
    {"lora", Lora::isEnabled()},
    {"qos", qos()}
  };
}

void NpuOp::updateAttrsForSharedWeights(
  std::map<std::string, std::any>& attrs
) const {
  if (shared_weights_.ready()) {
    attrs["skip_create_weights"] = 1;
  }
}

}  // namespace ryzenai::onnx_utils
