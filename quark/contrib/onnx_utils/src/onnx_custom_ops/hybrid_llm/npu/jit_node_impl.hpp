// Copyright (c) 2024 Advanced Micro Devices, Inc.

#pragma once

#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "jit_node.hpp"
#include "ryzenai/ryzen_mm.h"

namespace ryzenai::onnx_utils {

template <typename T>
JitNode<T>::JitNode(
  JitNode<T>* op_inter,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : ss_(session_configs) {
  ss_->topo_order.push_back(this);
  ss_->jit_futures.emplace_back();
  idx_ = ss_->offset;

  ss_->offset += 1;

  setReadAhead(session_configs);
}

template <typename T>
void JitNode<T>::loadFirstData() {
  loadJitData(idx_, true);
}

template <typename T>
void JitNode<T>::loadJitData(int idx, bool is_constructor) {
  if (idx < ss_->read_ahead || !ss_->jit_enable) {
    ss_->topo_order[idx]->readDataImpl(ss_->read_offset);
    bo_idx_ = ss_->read_offset;
    // std::cout << "Read in " << idx << " to " << read_offset_ << std::endl;
    ss_->read_offset = ss_->jit_enable
                         ? (ss_->read_offset + 1) % ss_->read_ahead
                         : (ss_->read_offset + 1);
    if (is_constructor) {
      ss_->jit_state.push_back(JitState::Read);
    } else {
      ss_->jit_state[idx] = JitState::Read;
    }

    if (idx < kJitLoadAhead || !ss_->jit_enable) {
      ss_->topo_order[idx]->loadDataImpl(ss_->load_offset);
      ss_->jit_state[idx] = JitState::Loaded;
      // std::cout << "Loaded in " << idx << " to " << load_offset_ <<
      // std::endl;
      ss_->load_offset = ss_->jit_enable
                           ? (ss_->load_offset + 1) % ss_->read_ahead
                           : (ss_->load_offset + 1);
    }

  } else if (is_constructor) {
    ss_->jit_state.push_back(JitState::Unloaded);
  }
}

template <typename T>
void JitNode<T>::loadData() const {
  auto idx = (idx_ + kJitLoadAhead) % ss_->jit_state.size();
  auto bo_offset = ss_->load_offset;
  // std::cout << "Initiating load in node " << idx << std::endl;
  auto state = ss_->jit_state[idx];
  if (state == JitState::Loaded) {
    return;
  }
  if (state == JitState::Reading) {
    ss_->jit_futures[idx].get();
  } else if (state == JitState::Unloaded && kJitLoadAhead == 0) {
    // in this case, weightReady() will handle loading data
    return;
  } else if (state != JitState::Read) {
    throw std::invalid_argument(
      "Loading weights that haven't been read? " + std::to_string(idx_) + " " +
      std::to_string(static_cast<int>(state))
    );
  }
  ss_->jit_state[idx] = JitState::Loading;
  ss_->load_offset = (ss_->load_offset + 1) % ss_->read_ahead;
  ss_->jit_futures[idx] =
    std::async(std::launch::async, [idx, bo_offset, ss = ss_]() {
      ss->topo_order[idx]->loadDataImpl(bo_offset);
      ss->topo_order[idx]->bo_idx_ = bo_offset;
      ss->jit_state[idx] = JitState::Loaded;
      // std::cout << "Loaded in node " << idx << " to bo " << bo_offset
      //           << std::endl;
    });
}

template <typename T>
void JitNode<T>::readData() const {
  if (ss_->read_ahead < 0) {
    ss_->read_ahead = ss_->jit_state.size();
  }
  // we cannot read more than all of them so clamp it at the number of nodes
  auto max_read_ahead =
    std::min(static_cast<size_t>(ss_->read_ahead), ss_->jit_state.size());
  auto idx = (idx_ + max_read_ahead - 1) % ss_->jit_state.size();
  auto bo_offset = ss_->read_offset;
  // std::cout << "Initiating read in node " << idx << std::endl;
  auto state = ss_->jit_state[idx];
  if (state == JitState::Read || state == JitState::Loaded) {
    return;
  }
  if (state != JitState::Unloaded) {
    throw std::invalid_argument(
      "Reading weights that haven't been unloaded? " + std::to_string(idx_) +
      " " + std::to_string(static_cast<int>(state))
    );
  }
  ss_->jit_state[idx] = JitState::Reading;
  ss_->read_offset = (ss_->read_offset + 1) % ss_->read_ahead;
  ss_->jit_futures[idx] =
    std::async(std::launch::async, [idx, bo_offset, ss = ss_]() {
      ss->topo_order[idx]->readDataImpl(bo_offset);
      ss->topo_order[idx]->bo_idx_ = bo_offset;
      ss->jit_state[idx] = JitState::Read;
      // std::cout << "Read in node " << idx << " to bo " << bo_offset
      //           << std::endl;
    });
}

template <typename T>
void JitNode<T>::unloadData() const {
  // std::cout << "Initiating unloading in node " << idx_ << std::endl;
  auto idx = idx_;
  if (ss_->jit_state[idx] != JitState::Loaded) {
    throw std::invalid_argument("Weights should have been loaded to unload");
  }
  if (!isJitEnabled()) {
    return;
  }
  ss_->jit_state[idx] = JitState::Unloading;
  ss_->jit_futures[idx] = std::async(std::launch::async, [idx, ss = ss_]() {
    ss->topo_order[idx]->unloadDataImpl(ss->topo_order[idx]->bo_idx_);
    // std::cout << "Unloaded in node " << idx << " from bo "
    //           << JitNode::topo_order_[idx]->bo_idx_ << std::endl;
    ss->jit_state[idx] = JitState::Unloaded;
  });
}

template <typename T>
int JitNode<T>::weightsReady() const {
  auto state = ss_->jit_state[idx_];
  switch (state) {
    case JitState::Loaded:
      return bo_idx_;
    case JitState::Loading:
      // std::cout << "Waiting for weights to load\n";
      ss_->jit_futures[idx_].get();
      return bo_idx_;
    case JitState::Read:
      if (kJitLoadAhead != 0) {
        std::cout << "Read\n";
        return -1;
      }
      loadData();
      ss_->jit_futures[idx_].get();
      return bo_idx_;
    case JitState::Reading:
      // std::cout << "Waiting for weights to read\n";
      if (kJitLoadAhead != 0) {
        std::cout << "Read\n";
        return -1;
      }
      ss_->jit_futures[idx_].get();
      loadData();
      ss_->jit_futures[idx_].get();
      return bo_idx_;
    case JitState::Unloaded:
      std::cout << "unloaded\n";
      return -1;
    case JitState::Unloading:
      std::cout << "unloading\n";
      ss_->jit_futures[idx_].get();
      return -1;

    default:
      throw std::invalid_argument("Unhandled switch");
  }
}

template <typename T>
void JitNode<T>::setReadAhead(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto& option = session_configs.at("hybrid_opt_npu_read_ahead");
  auto option_int = option.empty() ? kJitReadAhead : std::stoi(option);
  if (option_int < 0) {
    ss_->jit_enable = false;
  }
  // allow setting negative number here. It gets overwritten during readData()
  ss_->read_ahead =
    option_int > kJitLoadAhead || option_int < 0 ? option_int : kJitReadAhead;
}

template <typename T>
int JitNode<T>::readAhead() const {
  return ss_->read_ahead;
}

template <typename T>
bool JitNode<T>::isJitEnabled() const {
  return ss_->jit_enable;
}

template <typename T>
void JitNode<T>::initializeSharedBuffers(
  std::vector<RyzenMM::BufferRef>& buffers
) const {
  if (ss_->read_ahead < 0) {
    buffers.emplace_back();
  } else if (buffers.empty()) {
    buffers.resize(ss_->read_ahead);
  }
}

}  // namespace ryzenai::onnx_utils
