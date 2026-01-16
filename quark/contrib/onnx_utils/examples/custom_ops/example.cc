// Copyright (c) 2024 Advanced Micro Devices, Inc.

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <ratio>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
// The Windows header needs to be imported first for the executable to compile
// clang-format off
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h> // NOLINT(misc-include-cleaner)
#include <processthreadsapi.h>
#include <Psapi.h>
// clang-format on
#endif  // _WIN32

#include "ryzenai/onnx_utils/custom_ops_options.hpp"
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
#include "ryzenai/onnx_utils/custom_allocator.hpp"
#endif
#include "ryzenai/onnx_utils/string.hpp"

using RandomGenerator = std::mt19937;

constexpr auto kDebugPrintNum = 10;
constexpr auto kSeed = 42;
constexpr auto kMaxRandomIntValue = 255;

namespace {

/**
 * @brief Similar to std::is_same, returns true if T is any of the types in the
 * variadic expression
 *
 * @tparam T base type to check
 * @tparam Ts list of types to compare against
 */
template <typename T, typename... Ts>
// NOLINTNEXTLINE(readability-identifier-naming)
struct is_any : std::disjunction<std::is_same<T, Ts>...> {};

template <typename T, typename... Ts>
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr bool is_any_v = is_any<T, Ts...>::value;

#ifdef _WIN32
std::pair<std::size_t, std::size_t> getPeakMemory() {
  if (PROCESS_MEMORY_COUNTERS pmc;
      GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return {pmc.PeakWorkingSetSize, pmc.PeakPagefileUsage};
  }
  return {0, 0};
}
#endif  // _WIN32

std::vector<const char*> getInputNames(const Ort::Session& session) {
  const Ort::AllocatorWithDefaultOptions allocator;
  const size_t node_count = session.GetInputCount();
  std::vector<const char*> out(node_count);
  for (size_t i = 0; i < node_count; i++) {
    auto tmp = session.GetInputNameAllocated(i, allocator);
    out[i] = tmp.get();
  }
  return out;
}

std::vector<const char*> getOutputNames(const Ort::Session& session) {
  const Ort::AllocatorWithDefaultOptions allocator;
  const size_t node_count = session.GetOutputCount();
  std::vector<const char*> out(node_count);
  for (size_t i = 0; i < node_count; i++) {
    auto tmp = session.GetOutputNameAllocated(i, allocator);
    out[i] = tmp.get();
  }
  return out;
}

std::vector<std::vector<int64_t>> getInputShapes(const Ort::Session& session) {
  const size_t node_count = session.GetInputCount();
  std::vector<std::vector<int64_t>> out(node_count);
  for (size_t i = 0; i < node_count; i++) {
    out[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
  }
  return out;
}

std::vector<std::vector<int64_t>> getOutputShapes(const Ort::Session& session) {
  const size_t node_count = session.GetOutputCount();
  std::vector<std::vector<int64_t>> out(node_count);
  for (size_t i = 0; i < node_count; i++) {
    out[i] =
      session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
  }
  return out;
}

std::string tensorElementTypeToString(ONNXTensorElementDataType elem) {
  switch (elem) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ";
    default:
      return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED";
  }
}

// pretty prints a shape dimension vector
std::string printShape(const std::vector<int64_t>& vec) {
  std::stringstream ss("");
  ss << "[ ";
  for (size_t i = 0; i < vec.size() - 1; i++) {
    ss << vec[i] << " x ";
  }
  ss << vec[vec.size() - 1] << " ]";
  return ss.str();
}

template <typename T, typename Allocator>
auto createInputTensor(
  const OrtMemoryInfo* const memory_info, std::vector<T> input_data,
  std::vector<int64_t> input_shapes, Allocator* allocator,
  bool use_custom_allocator
) {
  if (use_custom_allocator) {
    auto tensor = Ort::Value::CreateTensor<T>(
      allocator, input_shapes.data(), input_shapes.size()
    );
    std::memcpy(
      tensor.template GetTensorMutableData<T>(), input_data.data(),
      input_data.size() * sizeof(T)
    );
    return tensor;
  }
  return Ort::Value::CreateTensor<T>(
    memory_info, input_data.data(), input_data.size(), input_shapes.data(),
    input_shapes.size()
  );
}

void printIoSummary(
  Ort::TypeInfo type_info, std::string_view name, std::string_view label
) {
  std::cout << label << " name: " << name << "\n";

  std::cout << label << " tensor ";
  std::cout << tensorElementTypeToString(
                 type_info.GetTensorTypeAndShapeInfo().GetElementType()
               )
            << "\n";

  std::cout << label << " dimension Count: "
            << type_info.GetTensorTypeAndShapeInfo().GetDimensionsCount();

  auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
  if (!shape.empty()) {
    std::cout << label << " shape: " << printShape(shape) << "\n";
  } else {
    std::cout << label << " shape: []\n";
  }
}

std::vector<std::string> printInputSummary(
  size_t count, const Ort::Session& session, OrtAllocator* allocator
) {
  std::vector<std::string> input_names(count);
  for (auto i = 0U; i < count; i++) {
    input_names[i] = session.GetInputNameAllocated(i, allocator).get();
    printIoSummary(session.GetInputTypeInfo(i), input_names[i], "Input");
  }

  return input_names;
}

std::vector<std::string> printOutputSummary(
  size_t count, const Ort::Session& session, OrtAllocator* allocator
) {
  std::vector<std::string> output_names(count);
  for (auto i = 0U; i < count; i++) {
    output_names[i] = session.GetOutputNameAllocated(i, allocator).get();
    printIoSummary(session.GetOutputTypeInfo(i), output_names[i], "Output");
  }

  return output_names;
}

template <typename T, typename Generator>
std::vector<T> buildRandomData(
  const Ort::ShapeInferContext::Ints& shape, Generator gen
) {
  size_t raw_data_size = 1;
  for (const auto& elem : shape) {
    // assuming shape of 1 if shape is -1
    raw_data_size *= elem == -1 ? 1 : elem;
  }
  raw_data_size = std::max(raw_data_size, shape.size());

  std::vector<T> random_data;
  random_data.reserve(raw_data_size);

  if constexpr (is_any_v<T, float, Ort::Float16_t>) {
    // NOLINTNEXTLINE(misc-const-correctness)
    std::uniform_real_distribution dist{0.0F, 1.0F};
    for (int i = 0; i < raw_data_size; ++i) {
      const auto random_value = dist(gen);
      random_data.push_back(static_cast<T>(random_value));
    }
  } else if constexpr (is_any_v<T, int32_t, int64_t>) {
    // NOLINTNEXTLINE(misc-const-correctness)
    std::uniform_int_distribution dist{0, kMaxRandomIntValue};
    for (int i = 0; i < raw_data_size; ++i) {
      const auto random_value = dist(gen);
      random_data.push_back(random_value);
    }
  } else {
    static_assert(!sizeof(T), "Invalid type to buildRandomData");
  }

  return random_data;
}

template <typename T, typename U = T>
void printOrtValue(
  const Ort::Value& value, int index, const std::string& label
) {
  const auto* array = value.GetTensorData<T>();
  std::cout << label << ":[" << index << "]\n";

  for (auto i = 0; i < kDebugPrintNum; i++) {
    std::cout << static_cast<U>(array[i]) << " ";
  }
  std::cout << "\n";
}

int entryPoint(const std::vector<std::string>& args) {
  if (args.size() < 3) {
    std::cerr
      << "Usage: example_custom_ops dll_path model [external_data_file]\n";
    return -1;
  }

#ifdef _WIN32
  auto [start_working, start_page] = getPeakMemory();
#endif  // _WIN32

  const auto& dll_path = args[1];
  const auto& onnx_model = args[2];

#ifndef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  bool constexpr kUseCustomAllocator = false;
#else
  bool constexpr kUseCustomAllocator = true;
#endif  //  ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY

  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  const RandomGenerator gen(kSeed);
  const Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeMatMulExample"};

#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  ryzenai::onnx_utils::CustomAllocator custom_allocator(kUseCustomAllocator);
  auto* allocator_ptr = &custom_allocator;
  if constexpr (kUseCustomAllocator) {
    Ort::GetApi().RegisterAllocator(env, &custom_allocator);
  }
  const auto* memory_info = custom_allocator.Info();
#else
  OrtAllocator* allocator_ptr = nullptr;
  auto memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY

  Ort::SessionOptions session_options;

  // Use the CPU execution provider
  session_options.SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_DISABLE_ALL
  );
#ifdef ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  if constexpr (kUseCustomAllocator) {
    session_options.DisableMemPattern();
    session_options.AddConfigEntry("session.use_env_allocators", "1");
    session_options.AddConfigEntry(
      "custom_allocator",
      std::to_string(reinterpret_cast<std::uintptr_t>(&custom_allocator))
        .c_str()
    );
  }
#endif  // ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY

  if (args.size() > 3) {
    session_options.AddConfigEntry("external_data_file", args[3].c_str());
  }

#ifdef _WIN32
  const auto library_filename = ryzenai::onnx_utils::toWideString(dll_path);
  const auto model_path = ryzenai::onnx_utils::toWideString(onnx_model);
#else
  const auto library_filename = dll_path;
  const auto model_path = onnx_model;
#endif
  session_options.RegisterCustomOpsLibrary(library_filename.c_str());

  // Create the session
  Ort::Session session(env, model_path.c_str(), session_options);

  std::cout << "Model loaded successfully.\n\n";

  // Get input and output names
  const Ort::AllocatorWithDefaultOptions allocator;

  const size_t input_count = session.GetInputCount();
  std::cout << "Total input count: " << input_count << "\n";

  auto inNames = printInputSummary(input_count, session, allocator);
  std::vector<const char*> input_names(input_count, nullptr);
  for (uint32_t i = 0; i < input_count; i++) {
    input_names[i] = inNames[i].c_str();
  }

  const size_t output_count = session.GetOutputCount();
  std::cout << "Total output count: " << output_count << "\n";

  auto outNames = printOutputSummary(output_count, session, allocator);
  std::vector<const char*> output_names(output_count, nullptr);
  for (uint32_t i = 0; i < output_count; i++) {
    output_names[i] = outNames[i].c_str();
  }

  std::vector<std::vector<int64_t>> input_shapes = getInputShapes(session);
  const std::vector<std::vector<int64_t>> output_shapes =
    getOutputShapes(session);

  // Prepare random input data
  std::vector<Ort::Value> input_tensors;

  for (int i = 0; i < input_count; i++) {
    auto data_type =
      session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

    for (int y = 0; y < input_shapes[i].size(); y++) {
      // if input shape is -1
      input_shapes[i][y] = input_shapes[i][y] == -1 ? 1 : input_shapes[i][y];
    }

    switch (data_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        auto random_data =
          buildRandomData<float, RandomGenerator>(input_shapes[i], gen);
        input_tensors.push_back(
          createInputTensor<float>(
            memory_info, random_data, input_shapes[i], allocator_ptr,
            kUseCustomAllocator
          )
        );
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        auto random_data = buildRandomData<Ort::Float16_t, RandomGenerator>(
          input_shapes[i], gen
        );
        input_tensors.push_back(
          createInputTensor<Ort::Float16_t>(
            memory_info, random_data, input_shapes[i], allocator_ptr,
            kUseCustomAllocator
          )
        );
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        auto random_data =
          buildRandomData<int64_t, RandomGenerator>(input_shapes[i], gen);
        input_tensors.push_back(
          createInputTensor<std::int64_t>(
            memory_info, random_data, input_shapes[i], allocator_ptr,
            kUseCustomAllocator
          )
        );
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        auto random_data =
          buildRandomData<int32_t, RandomGenerator>(input_shapes[i], gen);
        input_tensors.push_back(
          createInputTensor<std::int32_t>(
            memory_info, random_data, input_shapes[i], allocator_ptr,
            kUseCustomAllocator
          )
        );
        break;
      }
      default:
        std::cerr << "Unsupported data type " +
                       tensorElementTypeToString(data_type)
                  << "\n";
        return -1;
    }
  }

  for (int i = 0; i < input_count; i++) {
    // double-check the dimensions of the input tensor
    assert(
      input_tensors[i].IsTensor() &&
      input_tensors[i].GetTensorTypeAndShapeInfo().GetShape() == input_shapes[i]
    );
  }

  for (int i = 0; i < input_count; i++) {
    auto ort_type =
      input_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();

    switch (ort_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        printOrtValue<Ort::Float16_t, float>(input_tensors[i], i, "Input");
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        printOrtValue<int32_t>(input_tensors[i], i, "Input");
        break;
      }
      default:
        std::cerr << "Only Int32 and Float16 are supported input types\n";
        return -1;
    }
  }

  std::cout << "Running...\n";
  // Execute the model
  // TODO(varunsh): we can also use the custom allocator for the output tensors
  std::vector<Ort::Value> output_tensors = session.Run(
    Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
    input_count, output_names.data(), output_count
  );

  //// output_tensor from previous run becomes input to this run
  // std::vector<Ort::Value> output_tensors2 = session.Run(
  //   Ort::RunOptions{nullptr}, input_names.data(), output_tensors.data(),
  //   input_count, output_names.data(), output_count);

  //  Process output
  for (int i = 0; i < output_count; i++) {
    auto ort_type =
      output_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();

    switch (ort_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        printOrtValue<Ort::Float16_t, float>(output_tensors[i], i, "Output");
        break;
      }
      default:
        std::cerr << "Only Float16 are supported output types\n";
        return -1;
    }
  }

  std::cout << "Ran successfully\n";

#ifdef _WIN32
  auto [end_working, end_page] = getPeakMemory();
  auto relative_peak_working =
    static_cast<double>(end_working - start_working) / std::giga::num;
  auto relative_peak_page =
    static_cast<double>(end_page - start_page) / std::giga::num;
  std::cout << "PeakWorkingSetSize: " << relative_peak_working << " GB\n";
  std::cout << "PeakPagefileUsage: " << relative_peak_page << " GB\n";
#endif  // _WIN32

  return 0;
}

}  // anonymous namespace

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  try {
    const std::vector<std::string> args(argv, argv + argc);
    return entryPoint(args);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return -1;
  } catch (...) {
    std::cerr << "Unknown exception occurred\n";
    return -1;
  }
}
