// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <xrt/xrt_bo.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ryzenai::onnx_utils::SharedBuffer {
// Ordered set of buffer's slices size requirements.
// Example: {"in": 1024, "out": 2048}
//
using Requirements = std::vector<std::pair<std::string, size_t>>;

struct Span {
  void* ptr;
  size_t len;
};

// Lightweight interface to a single instance of shared buffer object.
// Represents an user-specific view of the shared buffer. The shared buffer is
// allocated when the first Client instance is created and deallocated when the
// last Client instance is destroyed. The shared buffer grows as much as needed
// to fullfill all users' requirements. Each instance must be created with
// unique combination of group_name (or buffer view name) and node_name (or
// client name).
//
struct Client {
  Client() = default;
  Client(
    const std::string& group_name, const std::string& session_id,
    const std::string& node_name
  );
  ~Client() = default;

  // The way to inform shared buffer system about buffer sizes required.
  // Must be called before any other method.
  // Might cause buffer reallocation which might take time.
  // Buffer reallocation invalidates all other buffers.
  //
  void Update(Requirements reqs);

  // The way to ask shared buffer whether existing buffer (key+ptr+len) must be
  // "rebound". "key" must be one of previously provided to "Update" method. If
  // provided buffer is still valid, std::nullopt is returned. Otherwise, valid
  // pointer and length are returned and user is expected to use those.
  //
  std::optional<Span> Validate(
    const std::string& key, const void* ptr, size_t len
  );

  // Handy wrapper for xrt::bo's
  //
  inline auto Validate(const std::string& key, xrt::bo bo) {
    return Validate(key, bo ? bo.map() : nullptr, bo ? bo.size() : 0);
  }

  // Returns current valid pointer for "key". Works similar to "Validate(key,
  // nullptr, 0)" but does not log about rebinding.
  //
  Span Get(const std::string& key);

  // Clears backend buffer.
  // Next call to Update will allocate a new buffer.
  //
  void Reset();

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace ryzenai::onnx_utils::SharedBuffer
