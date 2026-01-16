// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "shared_weights.hpp"

#include <algorithm>

#include "op_fuser/fusion_rt.hpp"
#include "ort.hpp"
#include "utils/utils.hpp"
#include "utils/weak.hpp"

namespace ryzenai::onnx_utils {

bool SharedWeights::isEnabled() const {
  return !weights_.empty() && !model_key_.empty();
}

bool SharedWeights::ready() const {
  return isEnabled() && std::all_of(
                          weights_.begin(), weights_.end(),
                          [](const SharedWeightsInfo& w) { return w.addr != 0; }
                        );
}
void SharedWeights::setModelKey(std::string key) {
  model_key_ = std::move(key);
}

void SharedWeights::setWeightsCount(size_t count) { weights_.resize(count); }

void SharedWeights::setWeightKey(int index, std::string key) {
  weights_.at(index).key = std::move(key);
}

void SharedWeights::setWeightKey(
  int index, const Ort::ConstKernelInfo& info, const char* attr_name
) {
  weights_.at(index).key = getAttribute<std::string>(info, attr_name, "");
}

void SharedWeights::setWeightAddr(int index) {
  if (weights_.at(index).addr != 0) {
    return;
  }
  if (vitis::ai::WeakStore<std::string, OpsFusion::BoWithMap>::has(
        model_key_
      )) {
    auto weak_store_bo =
      vitis::ai::WeakStore<std::string, OpsFusion::BoWithMap>::get(model_key_);
    auto const_bo = weak_store_bo->buffer;
    if (weak_store_bo->op_offset.find(weights_.at(index).key) !=
        weak_store_bo->op_offset.end()) {
      weights_.at(index).addr =
        const_bo->address() + weak_store_bo->op_offset[weights_.at(index).key];
    }
  }
}

size_t SharedWeights::weightAddr(int index) const {
  return weights_.at(index).addr;
}

}  // namespace ryzenai::onnx_utils
