// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "dynamic_pad.hpp"

#include <chrono>
#include <filesystem>
#include <limits>
#include <variant>

#include "dynamic_dispatch.hpp"
#include "onnx.hpp"
#include "ryzenai/onnx_utils/string.hpp"

#ifdef _PROFILE_DYNPAD_
#define _T(func)                                                          \
  {                                                                       \
    auto start = std::chrono::high_resolution_clock::now();               \
    func;                                                                 \
    auto end = std::chrono::high_resolution_clock::now();                 \
    std::chrono::duration<double, std::milli> elapsed = end - start;      \
    std::cout << "Total time: " << elapsed.count() << " ms" << std::endl; \
  }
#else
#define _T(func) func
#endif

using TypedPtr = std::variant<
  float*, double*, int8_t*, int16_t*, int32_t*, int64_t*, uint8_t*, uint16_t*,
  uint32_t*, uint64_t*>;

template <typename T>
TypedPtr OnnxTypeToTypedPtr(T ptr, ONNXTensorElementDataType dtype) {
  using RawPtr = std::remove_cv_t<std::remove_pointer_t<T>>;
  RawPtr* raw_ptr = const_cast<RawPtr*>(reinterpret_cast<const RawPtr*>(ptr));

  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return static_cast<int8_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return static_cast<int16_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return static_cast<int32_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return static_cast<int64_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return static_cast<uint8_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return static_cast<uint16_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return static_cast<uint32_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return static_cast<uint64_t*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return static_cast<int16_t*>(raw_ptr);  // here we use int16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return static_cast<float*>(raw_ptr);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return static_cast<double*>(raw_ptr);
    default:
      throw std::runtime_error("Unsupported dtype for DynamicPad: " + dtype);
  }
}

namespace ryzenai::onnx_utils {

/**
 * @brief Convert offset of a tensor to indexes
 *
 * @param idx offset
 * @param shape shape of the tensor
 *
 * @return indexes
 */
static std::vector<int64_t> offset2indexes(
  int64_t idx, const std::vector<int64_t>& shape
) {
  std::vector<int64_t> ret(shape.size());
  int64_t i = shape.size() - 1;
  while (idx > 0) {
    ret[i] = idx % shape[i];
    idx = idx / shape[i];
    i--;
  }
  return ret;
}

/**
 * @brief Convert indexes to offset
 *
 * @param indexes indexes
 * @param shape shape of the tensor
 *
 * @return offset
 */
static int64_t indexes2offset(
  const std::vector<int64_t>& indexes, const std::vector<int64_t>& shape
) {
  int64_t offset = 0;
  for (size_t i = 0; i < indexes.size(); i++) {
    offset = offset * shape[i] + indexes[i];
  }
  return offset;
}

/**
 * @brief Get size of a tensor
 *
 * @param shape shape of the tensor
 *
 * @return size
 */
static size_t shape2size(const std::vector<int64_t>& shape) {
  size_t size = 1;
  for (auto& dim : shape) {
    size *= dim;
  }
  return size;
}

/**
 * @brief Pad a tensor(optimized)
 *
 * @param output output tensor
 * @param input input tensor
 * @param input_shape shape of the input tensor
 * @param output_shape shape of the output tensor
 * @param pad_value value to pad
 *
 * @return true if success, false otherwise
 */
template <typename T>
static bool pad_opt(
  T* output, const T* input, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, const T pad_value = 0
) {
  // optimize for SD cases
  if (input_shape.size() == 4 && output_shape.size() == 4 &&
      output_shape[0] == input_shape[0]) {
    if (output_shape[1] == input_shape[1] &&
        output_shape[2] == input_shape[2] &&
        output_shape[3] == input_shape[3]) {
      // output shape same with input, just copy
      auto out_size = shape2size(output_shape);
      std::memcpy(output, input, out_size * sizeof(T));
      return true;
    } else if (output_shape[1] == input_shape[1]) {
      // (const1, const2, w, h) -> (const1, const2, w_pad, h_pad)
      int64_t outer = output_shape[0] * output_shape[1];
      for (int64_t k = 0; k < outer; k++) {
        for (int64_t j = 0; j < input_shape[2]; j++) {
          auto in_offset =
            k * input_shape[2] * input_shape[3] + j * input_shape[3];
          auto out_offset =
            k * output_shape[2] * output_shape[3] + j * output_shape[3];
          std::memcpy(
            output + out_offset, input + in_offset, input_shape[3] * sizeof(T)
          );
          std::fill(
            output + out_offset + input_shape[3],
            output + out_offset + output_shape[3], pad_value
          );
        }
        std::fill(
          output + k * output_shape[2] * output_shape[3] +
            input_shape[2] * output_shape[3],
          output + (k + 1) * output_shape[2] * output_shape[3], pad_value
        );
      }
      return true;
    } else if (output_shape[3] == input_shape[3]) {
      // (const1, x, y const2) -> (const1, x_pad,  y_pad, const2)
      for (int64_t i = 0; i < output_shape[0]; i++) {
        for (int64_t j = 0; j < input_shape[1]; j++) {
          auto in_offset =
            i * input_shape[1] * input_shape[2] * input_shape[3] +
            j * input_shape[2] * input_shape[3];
          auto out_offset =
            i * output_shape[1] * output_shape[2] * output_shape[3] +
            j * output_shape[2] * output_shape[3];
          std::memcpy(
            output + out_offset, input + in_offset,
            input_shape[2] * input_shape[3] * sizeof(T)
          );
          std::fill(
            output + out_offset + input_shape[2] * output_shape[3],
            output + out_offset + output_shape[2] * output_shape[3], pad_value
          );
        }
        std::fill(
          output + i * output_shape[1] * output_shape[2] * output_shape[3] +
            input_shape[1] * output_shape[2] * output_shape[3],
          output +
            (i + 1) * output_shape[1] * output_shape[2] * output_shape[3],
          pad_value
        );
      }
      return true;
    }
  }
  // optimize for LLM cases
  if (input_shape.size() == 2 && output_shape.size() == 2 &&
      output_shape[0] == input_shape[0]) {
    //(1, seq_len) -> (1, seq_len_pad)
    for (int64_t i = 0; i < output_shape[0]; i++) {
      auto in_offset = i * input_shape[1];
      auto out_offset = i * output_shape[1];
      std::memcpy(
        output + out_offset, input + in_offset, input_shape[1] * sizeof(T)
      );
      std::fill(
        output + out_offset + input_shape[1],
        output + out_offset + output_shape[1], pad_value
      );
    }
    return true;
  }
  // not supported yet
  return false;
}

/**
 * @brief Pad a tensor
 *
 * @param output output tensor
 * @param input input tensor
 * @param input_shape shape of the input tensor
 * @param output_shape shape of the output tensor
 * @param pad_value value to pad
 */
template <typename T>
static void pad(
  T* output, const T* input, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> output_shape, const T pad_value = 0
) {
  // check if there is an optimized implementation
  if (pad_opt(output, input, input_shape, output_shape, pad_value)) return;

  // fall back to the naive implementation
  auto out_size = shape2size(output_shape);
  std::fill(output, output + out_size, pad_value);
  auto in_size = shape2size(input_shape);
  for (size_t i = 0; i < in_size; i++) {
    auto indexes = offset2indexes(i, input_shape);
    auto out_offset = indexes2offset(indexes, output_shape);
    output[out_offset] = input[i];
  }
}
DynamicPadKernel::DynamicPadKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
) {
  ker_info_ = info;
  auto info_ptr = Ort::ConstKernelInfo(info);
  logger_ = info_ptr.GetLogger();
  dd_cache_dir_ = getCacheDirectory(session_configs);
  node_name_ = info_ptr.GetNodeName();

  read_attributes(info_ptr);
}

void DynamicPadKernel::read_attributes(const Ort::ConstKernelInfo& info_ptr) {
  // if there are multiple meta files, we need a way to find the right one. In
  // this case, we use dd_name attribute. Otherwise, we just pick the first one.
  std::string dd_name;
  try {
    dd_name = info_ptr.GetAttribute<std::string>("dd_name");
  } catch (const std::exception& e) {
    // continue with empty value
  }

  // find meta.json
  std::filesystem::path path;
  for (const auto& entry : std::filesystem::directory_iterator(dd_cache_dir_)) {
    if (endsWith(entry.path().string(), dd_name + "_meta.json")) {
      path = entry.path();
      break;
    }
  }
  if (path.empty()) {
    throw std::invalid_argument("Cannot find meta.json in pad custom op");
  } else {
    std::stringstream msg;
    msg << "[DynamicPad] meta.json found: " << path.string();
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, msg.str().c_str());
  }
  // save candidates
  auto meta = OpsFusion::load_meta_json(path.string());
  auto& shape_list = meta.dynamic_shape_list;
  for (auto& entry : shape_list) {
    candidates_.push_back(entry.dyn_shapes);
  }

  // save dynamic keys
  auto dyn_dim_str = info_ptr.GetAttribute<std::string>("input_shape");
  dyn_keys_.clear();
  std::istringstream dim_stream(dyn_dim_str);
  std::string dim;
  int index = 0;
  while (std::getline(dim_stream, dim, ',')) {
    dim.erase(dim.begin(), std::find_if(dim.begin(), dim.end(), [](int ch) {
                return !std::isspace(ch);
              }));
    dim.erase(
      std::find_if(
        dim.rbegin(), dim.rend(), [](int ch) { return !std::isspace(ch); }
      ).base(),
      dim.end()
    );
    if (std::any_of(dim.begin(), dim.end(), ::isalpha)) {
      dyn_keys_[index] = dim;
    }
    index++;
  }
}

/**
 * @brief Inference the best output shape from dynamic list candidates
 */
std::vector<int64_t> DynamicPadKernel::get_best_candidate(
  const std::vector<int64_t>& input_shape
) {
  // function to verify candidate
  // return score: smaller is better
  auto verify_candidate =
    [&](std::map<std::string, size_t>& dyn_shape) -> int64_t {
    int64_t score = 1;
    for (auto& [i, dyn_key] : dyn_keys_) {
      if (input_shape[i] > dyn_shape[dyn_key]) {
        return std::numeric_limits<int64_t>::max();
      }
      score *= dyn_shape[dyn_key];
    }
    return score;
  };

  // iterate over candidates
  int64_t cur_score = std::numeric_limits<int64_t>::max();
  std::map<std::string, size_t>* best_candidate = nullptr;
  for (auto& dyn_shape : candidates_) {
    int64_t score = verify_candidate(dyn_shape);
    if (score < cur_score) {
      cur_score = score;
      best_candidate = &dyn_shape;
    }
  }

  if (best_candidate == nullptr) {
    throw std::runtime_error("No valid dynamic shape found.");
  }
  // Update output shape
  auto output_shape = input_shape;
  for (auto& [i, dyn_key] : dyn_keys_) {
    output_shape[i] = best_candidate->at(dyn_key);
  }
  return output_shape;
}

void DynamicPadKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  const auto input_tensor = ctx.GetInput(0);
  const auto* input_data = input_tensor.GetTensorRawData();
  auto input_dtype = input_tensor.GetTensorTypeAndShapeInfo().GetElementType();
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  TypedPtr typed = OnnxTypeToTypedPtr(input_data, input_dtype);

  auto output_shape = get_best_candidate(input_shape);
  auto output_tensor = ctx.GetOutput(0, output_shape);
  auto* output_data = output_tensor.GetTensorMutableRawData();

  auto log_pad_shape = [&]() {
    std::stringstream msg;
    msg << "[DynamicPad] input shape: (";
    for (size_t i = 0; i < input_shape.size(); i++) {
      msg << input_shape[i];
      msg << (i < input_shape.size() - 1) ? ", " : ")";
    }
    msg << ", padded shape: (";
    for (size_t i = 0; i < output_shape.size(); i++) {
      msg << output_shape[i];
      msg << (i < output_shape.size() - 1) ? ", " : ")";
    }

    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, msg.str().c_str());
  };
  log_pad_shape();

  std::visit(
    [&](auto&& arg) {
      using T = std::remove_pointer_t<std::decay_t<decltype(arg)>>;
      _T(pad(
        static_cast<T*>(output_data), static_cast<const T*>(input_data),
        input_shape, output_shape
      ));
    },
    typed
  );
}

}  // namespace ryzenai::onnx_utils
