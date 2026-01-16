// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once

#include <future>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "external_data.pb.h"

namespace ryzenai::onnx_utils {

namespace proto {
class Header;
}

struct ExternalTensorInfo {
  uint64_t size;
  uint64_t offset;
  // for now, assume that all external tensors are in the same file
  // std::string filename;
};

ExternalTensorInfo getExternalTensorInfo(
  const proto::Header* header, const std::string& node_name, int tensor_index
);

void loadBin(
  void* bin_data, const std::string& fname, size_t size, uint64_t offset
);

namespace proto {

/**
 * @brief Get the proto header from a file
 *
 * @param external_data_path path to the header.pb.bin file
 * @return Header
 */
std::shared_ptr<Header> getHeader(std::string_view external_data_path);
/**
 * @brief Get the proto header from in-memory
 *
 * @param header pointer to the header in memory
 * @param header_size size of the header in bytes
 * @return Header
 */
std::shared_ptr<Header> getHeader(const void* header, size_t header_size);
/**
 * @brief Auto read the header from the session_config options either from file
 * or from in-memory.
 *
 * @param session_configs the inference session's config object
 * @return Header
 */
std::shared_ptr<Header> getHeader(
  const std::unordered_map<std::string, std::string>& session_configs
);

void saveHeader(const proto::Header* header, std::string_view path);

uint64_t getNpuMaxSize(const proto::Header* header, std::string_view op_type);

}  // namespace proto

}  // namespace ryzenai::onnx_utils
