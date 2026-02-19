// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif
#include <fstream>

#include "external_data.hpp"

namespace ryzenai::onnx_utils {

ExternalTensorInfo getExternalTensorInfo(
  const proto::Header* header, const std::string& node_name, int tensor_index
) {
  if (header->operators().find(node_name) == header->operators().end()) {
    return {0, 0};
  }
  if (tensor_index < 0) {
    auto size = header->operators().at(node_name).data().size();
    tensor_index = size + tensor_index;
  }

  const auto& tensor =
    header->operators().at(node_name).data().at(tensor_index);
  return {tensor.size(), tensor.offset()};
}

void loadBin(
  void* bindata, const std::string& fname, size_t size, uint64_t offset
) {
#if defined(_WIN32)
  HANDLE f_handle = CreateFile(
    fname.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL, nullptr
  );
  DWORD read = 0;
  OVERLAPPED ol{};
  // Set the offset from the start of the file
  ol.Offset = static_cast<std::uint32_t>(offset);
  ol.OffsetHigh = offset >> 32;
  auto status = ReadFile(f_handle, bindata, size, &read, &ol);
  if (!status) {
    auto error_code = GetLastError();
    if (error_code != ERROR_IO_PENDING) {
      CloseHandle(f_handle);
      throw std::runtime_error(
        "Readfile from " + fname + " failed: " + std::to_string(error_code)
      );
    }
  }
  CloseHandle(f_handle);
#else
  int fd = open(fname.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error(
      "Failed to open file: " + fname + " (" + std::strerror(errno) + ")"
    );
  }

  off_t seek_result = lseek64(fd, static_cast<off64_t>(offset), SEEK_SET);
  if (seek_result == -1) {
    close(fd);
    throw std::runtime_error(
      "Failed to seek to offset " + std::to_string(offset) +
      " in file: " + fname + " (" + std::strerror(errno) + ")"
    );
  }

  ssize_t bytes_read = read(fd, bindata, size);
  if (bytes_read == -1) {
    close(fd);
    throw std::runtime_error(
      "Failed to read from file: " + fname + " (" + std::strerror(errno) + ")"
    );
  }

  if (static_cast<size_t>(bytes_read) != size) {
    close(fd);
    throw std::runtime_error(
      "Failed to read " + std::to_string(size) + " bytes from file: " + fname +
      " (only read " + std::to_string(bytes_read) + " bytes)"
    );
  }

  close(fd);
#endif
}

namespace proto {

std::shared_ptr<Header> getHeader(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  const auto& external_data_file = session_configs.at("external_data_file");

  // Used for 2 purposes
  // 1. load external header file if needed
  // 2. get parent path to find where JIT weights are
  if (external_data_file.empty()) {
    throw std::invalid_argument(
      "external_data_file not specified in the session options"
    );
  }

  const auto& external_data_blob_size_str =
    session_configs.at("external_data_blob_size");

  Header header;
  size_t external_data_blob_size = 0;

  if (!external_data_blob_size_str.empty()) {
    external_data_blob_size = std::stoull(external_data_blob_size_str);
  }

  if (external_data_blob_size) {
    const void* external_data_blob =
      (const void*)std::stoull(session_configs.at("external_data_blob"));

    return getHeader(external_data_blob, external_data_blob_size);
  }
  return getHeader(external_data_file);
}

std::shared_ptr<Header> getHeader(std::string_view external_data_path) {
  auto header = std::make_shared<Header>();

  if (std::fstream header_stream(
        external_data_path.data(), std::ios::in | std::ios::binary
      );
      !header->ParseFromIstream(&header_stream)) {
    throw std::invalid_argument(
      "Cannot read header from " + std::string{external_data_path}
    );
  }

  return header;
}

std::shared_ptr<Header> getHeader(const void* header_addr, size_t header_size) {
  auto header = std::make_shared<Header>();
  if (!header->ParseFromArray(header_addr, header_size)) {
    throw std::invalid_argument("Cannot read in-memory header");
  }
  return header;
}

void saveHeader(const Header* header, std::string_view path) {
  std::fstream fout(
    path.data(), std::ios::out | std::ios::trunc | std::ios::binary
  );
  if (!fout.is_open()) {
    throw std::invalid_argument("Cannot open " + std::string{path});
  }
  header->SerializeToOstream(&fout);
}

uint64_t getNpuMaxSize(const Header* header, std::string_view op_type) {
  return header->op_metadata().at(op_type).max_npu_buffer_size();
}

}  // namespace proto

}  // namespace ryzenai::onnx_utils
