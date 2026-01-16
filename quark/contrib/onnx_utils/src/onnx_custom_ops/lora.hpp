// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "external_data.hpp"
#include "lora_op_interface.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"

using LoraOp = ryzenai::onnx_utils::LoraOpInterface;

namespace ryzenai::RyzenMM {
struct Allocator;
struct BufferRef;
}  // namespace ryzenai::RyzenMM

namespace ryzenai::onnx_utils {

// Enum for LoRA data type (prefill or token phase)
enum class LoraDataType { Prefill, Token };

class Lora {
 public:
  static void enableLora(
    const std::unordered_map<std::string, std::string>& session_configs
  );
  static bool isEnabled();
  static std::filesystem::path header(const std::string& type = "prefill");
  static std::filesystem::path data(const std::string& type = "prefill");

  static const proto::Header* prefillHeader();
  static const proto::Header* tokenHeader();
  static void releasePrefillHeader();
  static void releaseTokenHeader();

  static void setLoraAdapter(const std::string& lora_name);
  static void setLoraAdapterFromBuffer(
    const std::string& lora_name, const void* prefill_proto_ptr,
    size_t prefill_proto_size, const void* prefill_bin_ptr,
    size_t prefill_bin_size, const void* token_proto_ptr,
    size_t token_proto_size, const void* token_bin_ptr, size_t token_bin_size
  );
  static const std::string& getLoraName();

  // Buffer-based data access
  static bool isFromBuffer();

  // Unified data loading (works with both file and buffer modes)
  static void loadBinData(
    void* dest, size_t size, uint64_t offset,
    LoraDataType type = LoraDataType::Prefill
  );

  // LoRA op registration
  static void addLoraOp(LoraOp* op);

 private:
  // Trigger LoadLora on all registered ops
  static void loadAllLoraOps();
  inline static bool enable_ = false;
  inline static std::string lora_name_ = "base";
  inline static std::filesystem::path model_dir_;
  inline static std::shared_ptr<proto::Header> prefill_header_{nullptr};
  inline static std::shared_ptr<proto::Header> token_header_{nullptr};

  // Buffer-based lora data (pointers to external buffers - valid during
  // LoraComputeFromBuffer)
  inline static bool from_buffer_ = false;
  inline static const void* prefill_bin_data_ = nullptr;
  inline static size_t prefill_bin_size_ = 0;
  inline static const void* token_bin_data_ = nullptr;
  inline static size_t token_bin_size_ = 0;

  // Registered LoRA ops
  inline static std::vector<LoraOp*> lora_ops_;
};

class LoraInterface {
 public:
  virtual void zeroesAllLoraData() const = 0;
  virtual void syncLora(void* rt = nullptr) = 0;
  void setLora(const std::string& prefix);
  const std::string& getLoraName() const;
  static const size_t getMaxRank();

 private:
  std::string lora_name_ = "base";
  inline static size_t kMaxRank_ = 128;
};

class LoraBuffer : public LoraInterface {
 public:
  LoraBuffer() = default;
  virtual ~LoraBuffer();
  void addBo(
    size_t k, size_t n,
    ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* gemm,
    RyzenMM::Allocator allocator
  );
  void* data(int index) const;
  void zeroesLoraData(int index = 0) const;
  void zeroesAllLoraData() const;
  void syncLora(void* rt = nullptr);
  const xrt::bo& getBo(int index) const;
  size_t getBoSize(int index) const;

 private:
  std::vector<void*> buffers_;
  std::vector<xrt::bo> xrt_bos_;
  std::vector<size_t> bo_sizes_;
};

}  // namespace ryzenai::onnx_utils
