// Copyright (c) 2025 Advanced Micro Devices, Inc.

#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ryzenai {
template <typename T>
struct SessionState {
  SessionState() = default;

  SessionState(
    const std::unordered_map<std::string, std::string>& session_configs
  ) {
    if (!session_configs.count("session_id")) {
      throw std::runtime_error{"SessionState: session_id is missing"};
    }

    const auto& session_id = session_configs.at("session_id");

    {
      std::unique_lock<std::mutex> lock(weaks_mutex_);
      auto& weak =
        weaks_[session_id];  // operator[] inserts a default-constructed element
                             // if 'session_id' isn't present

      if (!(strong_ = weak.lock())) {
        weak = strong_ = std::make_shared<T>();
      }
    }
  }

  explicit operator bool() const { return static_cast<bool>(strong_); }

  T* operator->() { return strong_.get(); }
  T* operator->() const { return strong_.get(); }
  T& operator*() { return *strong_; }

 private:
  inline static std::unordered_map<std::string, std::weak_ptr<T>> weaks_;
  inline static std::mutex weaks_mutex_;

  std::shared_ptr<T> strong_;
};
};  // namespace ryzenai
