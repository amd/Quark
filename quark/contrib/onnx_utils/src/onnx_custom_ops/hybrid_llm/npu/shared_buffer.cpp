// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "shared_buffer.hpp"

#include <ryzenai/ryzen_mm.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <set>
#include <unordered_map>

namespace ryzenai::onnx_utils::SharedBuffer {
struct Core {
  void Update(
    const std::string& group_name, const std::string& node_name,
    Requirements reqs
  ) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (groups_[group_name].Update(node_name, std::move(reqs))) {
      if (!buffer_might_need_update_ && tracing_)
        fprintf(
          stderr, "SharedBuffer[%s, %s]: buffer might be updated\n",
          group_name.c_str(), node_name.c_str()
        );

      buffer_might_need_update_ = true;
    }
  }

  const auto GetBuffer(
    const std::string& group_name, const std::string& node_name
  ) {
    if (buffer_might_need_update_ || !buffer_) {
      buffer_might_need_update_ = false;

      auto required_buffer_size = size_t(0);

      for (const auto& [_, tmp] : groups_)
        required_buffer_size =
          (std::max)(required_buffer_size, tmp.allocations_total);

      const auto current_buffer_size = buffer_ ? buffer_.Size() : 0;

      if ((!buffer_shrinking_ && current_buffer_size < required_buffer_size) ||
          (buffer_shrinking_ && current_buffer_size != required_buffer_size)) {
        if (tracing_)
          fprintf(
            stderr, "SharedBuffer[%s, %s]: reallocation %llu -> %llu\n",
            group_name.c_str(), node_name.c_str(), current_buffer_size,
            required_buffer_size
          );

        buffer_ = {};
        buffer_ = allocator_.AllocateBuffer(required_buffer_size);

        if (with_forced_rebinds_)
          for (auto& [group_name_to_rebind, group] : groups_)
            for (auto& [key_to_rebind, _] : group.allocations_map)
              forced_rebinds_.emplace(
                std::make_pair(group_name_to_rebind, key_to_rebind)
              );
      }
    }

    return buffer_.Data<uint8_t>();
  }

  std::optional<Span> Validate(
    const std::string& group_name, const std::string& node_name,
    const std::string& key, const void* ptr, size_t len
  ) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto& group = groups_[group_name];
    auto& alloc = group.allocations_map.at(key);
    const auto buffer = GetBuffer(group_name, node_name);
    const auto expected_ptr = static_cast<void*>(buffer + alloc.offset);
    const auto expected_len = alloc.length;
    const auto already_okay = ptr == expected_ptr && len >= expected_len;
    const auto force_rebind =
      forced_rebinds_.erase(std::make_pair(group_name, key)) > 0;

    if (already_okay && !force_rebind) return std::nullopt;

    if (tracing_) {
      if (!already_okay)
        fprintf(
          stderr,
          "SharedBuffer[%s, %s]: rebind '%s' from 0x%p:%llu to 0x%p:%llu\n",
          group_name.c_str(), node_name.c_str(), key.c_str(), ptr, len,
          expected_ptr, expected_len
        );
      else
        fprintf(
          stderr, "SharedBuffer[%s, %s]: forced rebind of '%s' to 0x%p:%llu\n",
          group_name.c_str(), node_name.c_str(), key.c_str(), expected_ptr,
          expected_len
        );
    }

    return Span{expected_ptr, expected_len};
  }

  Span Get(
    const std::string& group_name, const std::string& node_name,
    const std::string& key
  ) {
    std::lock_guard<std::mutex> lock(mtx_);

    const auto& group = groups_[group_name];
    const auto& alloc = group.allocations_map.at(key);

    return Span{
      static_cast<void*>(GetBuffer(group_name, node_name) + alloc.offset),
      alloc.length
    };
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (buffer_ && tracing_)
      fprintf(
        stderr, "SharedBuffer: RESET (%f MB)\n",
        buffer_ ? double(buffer_.Size()) / 1024 / 1024 : 0.0
      );

    buffer_ = {};
  }

  ~Core() {
    if (!tracing_) return;

    uint64_t total_required = 0;

    static const auto ToMB = [](size_t bytes) {
      return double(bytes) / 1024 / 1024;
    };

    for (auto& [group_name, group] : groups_) {
      total_required = std::max(total_required, group.allocations_total);

      fprintf(
        stderr, "SharedBuffer: group '%s' required %f MB for slices: ",
        group_name.c_str(), ToMB(group.allocations_total)
      );

      auto first = true;

      for (const auto& [key, alloc] : group.allocations_map) {
        if (!first)
          fprintf(stderr, ", ");
        else
          first = false;

        fprintf(stderr, "\"%s\" (%f MB)", key.c_str(), ToMB(alloc.length));
      }

      fprintf(stderr, "\n");
    }

    fprintf(
      stderr, "SharedBuffer: %f MB buffer was required\n", ToMB(total_required)
    );
  }

 private:
  struct Group {
    struct Allocation {
      size_t offset = 0, length = 0;
      bool rebind = true;
      inline bool operator==(const Allocation& other) const noexcept {
        return std::tie(offset, length) == std::tie(other.offset, other.length);
      }
    };

    std::unordered_map<std::string, Requirements>
      requirements_map;  // node_name is key
    std::unordered_map<std::string, Allocation>
      allocations_map;  // ("in", "out" are keys here

    size_t allocations_total = 0;

    bool Update(const std::string& node_name, Requirements reqs) {
      const auto& stored_reqs = requirements_map[node_name] = std::move(reqs);
      const auto prev_allocations_map = std::exchange(allocations_map, {});

      // in allocation map entry length must be max required by a node in the
      // group

      for (const auto& [_, tmp] : requirements_map)
        for (const auto& [key, val] : tmp) {
          auto& entry = allocations_map[key];
          entry.length = (std::max)(entry.length, val);
          if (prev_allocations_map.count(key))
            entry.rebind = prev_allocations_map.at(key).rebind;
          else
            entry.rebind = true;
        }

      // following block assigns offsets and calculates total length

      allocations_total = 0;

      for (const auto& [key, _] : stored_reqs) {
        if (allocations_map.count(key) == 0) continue;

        auto& entry = allocations_map[key];

        entry.offset = allocations_total;
        allocations_total += entry.length;
      }

      return allocations_map != prev_allocations_map;
    }
  };

 private:
  std::mutex mtx_;
  std::unordered_map<std::string, Group> groups_;  // group_name is key
  std::set<std::pair<std::string, std::string>>
    forced_rebinds_;  // group name + key pairs that must be rebound

  RyzenMM::NPUAllocator<'SHBU'> allocator_;
  RyzenMM::BufferRef buffer_;
  bool buffer_might_need_update_{true};

 private:
  // TODO: use following methods from EP
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

  const bool tracing_ =
    _IsFeatureEnabledViaEnvironment("RYZENAI_EP_SHARED_BUFFER_TRACING");

  const bool buffer_shrinking_ =
    _IsFeatureEnabledViaEnvironment("RYZENAI_EP_SHARED_BUFFER_SHRINKING");

  // TODO: make it work without this
  const bool with_forced_rebinds_ = !_IsFeatureEnabledViaEnvironment(
    "RYZENAI_EP_SHARED_BUFFER_DISABLE_FORCED_REBINDS"
  );
};

std::weak_ptr<Core> weak_core_;

struct Client::Impl {
  inline Impl(const std::string& group_name, const std::string& node_name)
    : group_name_(group_name), node_name_(node_name) {
    if (!(core_ = weak_core_.lock()))
      weak_core_ = core_ = std::make_shared<Core>();
  }

  inline void Update(Requirements reqs) {
    core_->Update(group_name_, node_name_, std::move(reqs));
  }

  inline std::optional<Span> Validate(
    const std::string& key, const void* ptr, size_t len
  ) {
    return core_->Validate(group_name_, node_name_, key, ptr, len);
  }

  inline Span Get(const std::string& key) {
    return core_->Get(group_name_, node_name_, key);
  }

  inline void Reset() { core_->Reset(); }

 private:
  const std::string group_name_, node_name_;
  std::shared_ptr<Core> core_;
};

Client::Client(
  const std::string& group_name, const std::string& session_id,
  const std::string& node_name
)
  : impl_(
      std::make_shared<Client::Impl>(group_name, session_id + "/" + node_name)
    ) {}

void Client::Update(Requirements reqs) { impl_->Update(std::move(reqs)); }

std::optional<Span> Client::Validate(
  const std::string& key, const void* ptr, size_t len
) {
  return impl_->Validate(key, ptr, len);
}

Span Client::Get(const std::string& key) { return impl_->Get(key); }

void Client::Reset() { impl_->Reset(); }

}  // namespace ryzenai::onnx_utils::SharedBuffer
