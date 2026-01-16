// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "pad_input_ids.hpp"

#include <filesystem>
#include <limits>

#include "dynamic_dispatch.hpp"
#include "onnx.hpp"

namespace ryzenai::onnx_utils {

namespace {

template <typename T, typename U>
void printIterable(const T& iter, const U& name) {
  std::cout << name << ": ";
  for (const auto& item : iter) {
    std::cout << item << ",";
  }
  std::cout << "\n";
}

}  // namespace

PadInputIdsKernel::PadInputIdsKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
) {
  auto info_ptr = Ort::ConstKernelInfo(info);

  dd_cache_dir_ = getCacheDirectory(session_configs) + "/.cache";

  node_name_ = info_ptr.GetNodeName();
  read_attributes(info_ptr);
}

void PadInputIdsKernel::read_attributes(const Ort::ConstKernelInfo& info_ptr) {
  auto count = info_ptr.GetAttribute<int64_t>("count");
  if (count > 0) {
    dimensions_.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto key = "dimension_" + std::to_string(i);
      dimensions_.emplace_back(info_ptr.GetAttributes<int64_t>(key.c_str()));
    }
    return;
  }

  std::string meta_json;
  auto dd_node = info_ptr.GetAttribute<std::string>("dd_node");
  const auto op_dir =
    std::filesystem::path{dd_cache_dir_ + "/" + dd_node + "_meta.json"};
  if (std::filesystem::exists(op_dir)) {
    meta_json = op_dir.string();
  } else {
    throw std::invalid_argument(
      "Cannot find metajson in pad custom op: " + op_dir.string()
    );
  }
  auto meta = OpsFusion::load_meta_json(meta_json);
  auto& shape_list = meta.dynamic_shape_list;

  for (const auto& entry : shape_list) {
    std::vector<int64_t> dimension = {
      1, int64_t(entry.dyn_shapes.at("sequence_length_padded"))
    };
    dimensions_.push_back(dimension);
  }
}

void PadInputIdsKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto* input_data = ctx.GetInput(0).GetTensorData<int64_t>();
  auto input_shape = ctx.GetInput(0).GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() != 2) {
    throw std::invalid_argument("Input shape must be 2D for now");
  }

  std::vector<int64_t> best_candidate(
    input_shape.size(), std::numeric_limits<int64_t>::max()
  );
  bool found = false;
  for (const auto& candidate : dimensions_) {
    bool valid_candidate = true;
    for (auto i = 0U; i < candidate.size(); i++) {
      if (candidate[i] < input_shape[i] || candidate[i] > best_candidate[i]) {
        valid_candidate = false;
        break;
      }
    }
    if (valid_candidate) {
      best_candidate = candidate;
      found = true;
    }
  }

  if (!found) {
    int index = 0;
    for (const auto& candidate : dimensions_) {
      printIterable(candidate, index);
      index++;
    }
    printIterable(input_shape, "Input_shape");
    throw std::invalid_argument("Found no padding candidate");
  }

  auto output_tensor = getOutputTensor(ctx, 0, best_candidate);
  for (int i = 0; i < best_candidate[0]; ++i) {
    for (int j = 0; j < best_candidate[1]; ++j) {
      auto value = i < input_shape[0] && j < input_shape[1]
                     ? input_data[i * input_shape[1] + j]
                     : 0;
      static_cast<int64_t*>(output_tensor.data)[i * best_candidate[1] + j] =
        value;
    }
  }
}

}  // namespace ryzenai::onnx_utils
