// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "lora.hpp"

#include <xrt/xrt_bo.h>

#include "custom_ops.hpp"
#include "onnxruntime_cxx_api.h"
#include "ryzenai/ryzen_mm.h"

namespace fs = std::filesystem;

void LoraCompute(const char* lora_name) {
  ryzenai::onnx_utils::Lora::setLoraAdapter(lora_name);
}

void LoraComputeFromBuffer(
  const char* lora_name, const void* prefill_proto_ptr,
  size_t prefill_proto_size, const void* prefill_bin_ptr,
  size_t prefill_bin_size, const void* token_proto_ptr, size_t token_proto_size,
  const void* token_bin_ptr, size_t token_bin_size
) {
  ryzenai::onnx_utils::Lora::setLoraAdapterFromBuffer(
    lora_name, prefill_proto_ptr, prefill_proto_size, prefill_bin_ptr,
    prefill_bin_size, token_proto_ptr, token_proto_size, token_bin_ptr,
    token_bin_size
  );
}

namespace ryzenai::onnx_utils {

void Lora::enableLora(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  try {
    auto value = session_configs.at("hybrid_opt_enable_npu_lora");
    enable_ = value.empty() ? false : value == "1";
    model_dir_ =
      fs::path(session_configs.at("external_data_file")).parent_path();
  } catch (const Ort::Exception&) {
    // continue with default value
  }
}

bool Lora::isEnabled() { return enable_; }

void Lora::setLoraAdapter(const std::string& lora_name) {
  if (lora_name == "base") {
    lora_name_ = lora_name;
    std::cout << "--Successfully set to base model with no lora data"
              << std::endl;
    return;
  }
  // check if model_dir/prefix.bin and model_dir/prefix.pb.bin exist
  // TODO(chiz): remove hardcoding of prefill here
  fs::path data_file = model_dir_ / fs::path(lora_name + "_prefill.bin");
  fs::path header_file = model_dir_ / fs::path(lora_name + "_prefill.pb.bin");

  if (!fs::exists(data_file) || !fs::exists(header_file)) {
    std::cerr << "--Failed to set new Lora adapter: " << lora_name << ""
              << std::endl;
    std::cerr << "--Required files do not exist: " << data_file.string()
              << " and " << header_file.string() << std::endl;
    if (lora_name_ == "base") {
      std::cerr << "--Fallback to base model" << std::endl;
    } else {
      std::cerr << "--Fallback to previous adapter: " << lora_name_
                << std::endl;
    }
  } else {
    std::cout << "--Successfully set new Lora adapter: " << lora_name
              << std::endl;
    lora_name_ = lora_name;
  }

  prefill_header_ = proto::getHeader(header_file.string());
  token_header_ = proto::getHeader(header("token").string());

  // Reset buffer mode since we're loading from file
  from_buffer_ = false;
  prefill_bin_data_ = nullptr;
  prefill_bin_size_ = 0;
  token_bin_data_ = nullptr;
  token_bin_size_ = 0;

  loadAllLoraOps();
}

void Lora::setLoraAdapterFromBuffer(
  const std::string& lora_name, const void* prefill_proto_ptr,
  size_t prefill_proto_size, const void* prefill_bin_ptr,
  size_t prefill_bin_size, const void* token_proto_ptr, size_t token_proto_size,
  const void* token_bin_ptr, size_t token_bin_size
) {
  // Check if lora_name is "base" or all buffers are empty - this means "base"
  // mode (unload LoRA)
  bool is_base = (lora_name == "base");
  bool all_empty = (prefill_proto_ptr == nullptr || prefill_proto_size == 0) &&
                   (prefill_bin_ptr == nullptr || prefill_bin_size == 0) &&
                   (token_proto_ptr == nullptr || token_proto_size == 0) &&
                   (token_bin_ptr == nullptr || token_bin_size == 0);

  if (is_base || all_empty) {
    // Switch to base mode - unload LoRA
    lora_name_ = "base";
    from_buffer_ = false;
    prefill_bin_data_ = nullptr;
    prefill_bin_size_ = 0;
    token_bin_data_ = nullptr;
    token_bin_size_ = 0;
    prefill_header_.reset();
    token_header_.reset();
    std::cout
      << "--Successfully set to base model with no lora data (from buffer API)"
      << std::endl;
    loadAllLoraOps();
    return;
  }

  // Validate all buffers are provided for LoRA mode
  if (prefill_proto_ptr == nullptr || prefill_proto_size == 0) {
    std::cerr << "--Failed to set LoRA adapter from buffer: invalid prefill "
                 "proto buffer"
              << std::endl;
    return;
  }

  if (prefill_bin_ptr == nullptr || prefill_bin_size == 0) {
    std::cerr
      << "--Failed to set LoRA adapter from buffer: invalid prefill bin buffer"
      << std::endl;
    return;
  }

  if (token_proto_ptr == nullptr || token_proto_size == 0) {
    std::cerr
      << "--Failed to set LoRA adapter from buffer: invalid token proto buffer"
      << std::endl;
    return;
  }

  if (token_bin_ptr == nullptr || token_bin_size == 0) {
    std::cerr
      << "--Failed to set LoRA adapter from buffer: invalid token bin buffer"
      << std::endl;
    return;
  }

  // Parse prefill and token headers from buffers
  prefill_header_ = proto::getHeader(prefill_proto_ptr, prefill_proto_size);
  token_header_ = proto::getHeader(token_proto_ptr, token_proto_size);

  // Store bin data pointers (data will be loaded during LoadLora() which is
  // called next)
  from_buffer_ = true;
  prefill_bin_data_ = prefill_bin_ptr;
  prefill_bin_size_ = prefill_bin_size;
  token_bin_data_ = token_bin_ptr;
  token_bin_size_ = token_bin_size;
  lora_name_ = lora_name;

  std::cout << "--Successfully set LoRA adapter '" << lora_name
            << "' from buffer"
            << " (prefill proto: " << prefill_proto_size << " bytes"
            << ", prefill bin: " << prefill_bin_size << " bytes"
            << ", token proto: " << token_proto_size << " bytes"
            << ", token bin: " << token_bin_size << " bytes)" << std::endl;

  loadAllLoraOps();
}

void Lora::addLoraOp(LoraOp* op) {
  if (isEnabled()) {
    lora_ops_.push_back(op);
  }
}

void Lora::loadAllLoraOps() {
  for (auto* op : lora_ops_) {
    op->LoadLora();
  }
}

const proto::Header* Lora::prefillHeader() { return prefill_header_.get(); }

const proto::Header* Lora::tokenHeader() { return token_header_.get(); }

void Lora::releasePrefillHeader() {
  if (prefill_header_ != nullptr) {
    prefill_header_.reset();
  }
}

void Lora::releaseTokenHeader() {
  if (token_header_ != nullptr) {
    token_header_.reset();
  }
}

const std::string& Lora::getLoraName() { return lora_name_; }

bool Lora::isFromBuffer() { return from_buffer_; }

void Lora::loadBinData(
  void* dest, size_t size, uint64_t offset, LoraDataType type
) {
  const bool is_prefill = (type == LoraDataType::Prefill);
  const char* type_name = is_prefill ? "prefill" : "token";

  if (from_buffer_) {
    // Load from external buffer pointer
    const void* bin_data = is_prefill ? prefill_bin_data_ : token_bin_data_;
    size_t bin_size = is_prefill ? prefill_bin_size_ : token_bin_size_;

    if (offset + size > bin_size) {
      throw std::runtime_error(
        std::string(
          "Lora::loadBinData: offset + size exceeds buffer size for "
          "type: "
        ) +
        type_name
      );
    }
    memcpy(dest, static_cast<const uint8_t*>(bin_data) + offset, size);
  } else {
    // Load from file
    loadBin(dest, data(type_name).string(), size, offset);
  }
}

std::filesystem::path Lora::header(const std::string& type) {
  auto header_path = model_dir_ / (lora_name_ + "_" + type + ".pb.bin");
  return header_path;
}

std::filesystem::path Lora::data(const std::string& type) {
  auto data_path = model_dir_ / (lora_name_ + "_" + type + ".bin");
  return data_path;
}

/**
 * @brief Compute the LoRA BO size based on the K, Rank, and N values. This is
 * copied from DynamicDispatch as-is.
 *
 * @param K
 * @param Rank
 * @param N
 * @return size_t
 */
inline size_t calculateLoraBoSize(int64_t K, int Rank, int64_t N) {
  // Calculate the size of the lora_a.
  // Lora_a comes from a matrix of shape (K, Rank) and applied with padding, in
  // the type of BF16.
  const int lora_a_size = 2 * 2 * 2 * (1 + K / 128) * 4 * (Rank / 16) * 8 * 8;
  const int a_data_band = sizeof(uint16_t);
  const int lora_a_data_size = lora_a_size * a_data_band;

  // Calculate the size of the lora_b.
  // Lora_b comes from a matrix of shape (Rank, N) and applied with padding, in
  // the type of BFP16.
  constexpr int ebs = 8;  // const value
  const int b_data_band = sizeof(uint8_t);
  const int lora_b_data_size = ((Rank * N) / ebs) * (ebs + 1) * b_data_band;
  return lora_a_data_size + lora_b_data_size;
}

void LoraBuffer::addBo(
  size_t k, size_t n,
  ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* gemm,
  RyzenMM::Allocator allocator
) {
  const auto kMaxRank = 128;
  const auto bo_size = calculateLoraBoSize(k, kMaxRank, n);
  bo_sizes_.push_back(bo_size);
#ifdef _WIN32
  auto* buf = _aligned_malloc(bo_size, 4096);
#else
  auto* buf = aligned_alloc(4096, bo_size);
#endif

  memset(buf, 0, bo_size);
  buffers_.push_back(buf);
  xrt_bos_.push_back(gemm->bind_bo(buffers_.back(), bo_size));
}

LoraBuffer::~LoraBuffer() {
  std::for_each(buffers_.begin(), buffers_.end(), [&](void* buf) {
#ifdef _WIN32
    _aligned_free(buf);
#else
    free(buf);
#endif
  });
  buffers_.clear();
  xrt_bos_.clear();
  bo_sizes_.clear();
}

void LoraBuffer::syncLora(void* rt) {
  for (auto& bo : xrt_bos_) {
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
}

void LoraBuffer::zeroesLoraData(int index) const {
  memset(buffers_.at(index), 0, bo_sizes_.at(index));
}

void LoraBuffer::zeroesAllLoraData() const {
  auto total_bufs = buffers_.size();
  if (total_bufs != bo_sizes_.size() || total_bufs != xrt_bos_.size()) {
    throw std::runtime_error("LoraBuffer: Inconsistent buffer sizes");
  }
  for (size_t i = 0; i < total_bufs; ++i) {
    memset(buffers_.at(i), 0, bo_sizes_.at(i));
  }
}

void* LoraBuffer::data(int index) const { return buffers_.at(index); }

void LoraInterface::setLora(const std::string& lora_name) {
  lora_name_ = lora_name;
}

const std::string& LoraInterface::getLoraName() const { return lora_name_; }
const size_t LoraInterface::getMaxRank() { return kMaxRank_; }

const xrt::bo& LoraBuffer::getBo(int index) const { return xrt_bos_.at(index); }

size_t LoraBuffer::getBoSize(int index) const { return bo_sizes_.at(index); }

}  // namespace ryzenai::onnx_utils
