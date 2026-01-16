// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "corelib.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "external_data.hpp"
#include "onnxruntime_cxx_api.h"
#include "operator.hpp"
#include "ort.hpp"
#include "ryzenai/corelib.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

using ryzenai::corelib::DataType;
using ryzenai::corelib::QuantizedConstTensor;
using ryzenai::corelib::QuantizedPlaceholderTensor;

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

using SessionConfigs = std::unordered_map<std::string, std::string>;

struct CoreLibKernelBase {
 protected:
  CoreLibKernelBase(
    const OrtKernelInfo* info, const SessionConfigs& session_configs
  )
    : kinfo_(info), session_configs_(session_configs) {}

  template <typename R>
  R GetAttribute(const char* name, std::optional<R> def = std::nullopt) const {
    return ryzenai::onnx_utils::getAttribute<R>(kinfo_, name, def);
  }

  template <typename R>
  std::vector<R> GetAttributes(
    const char* name, std::optional<std::vector<R>> def = std::nullopt
  ) const {
    return ryzenai::onnx_utils::getAttributes<R>(kinfo_, name, def);
  }

  struct ExternalData {
    proto::Header header;
    fs::path path;
  };

  std::shared_ptr<ExternalData> LoadExternalData() {
    if (external_data_) return external_data_;

    if (0 == session_configs_.count("external_data_file")) return {};

    auto path = fs::path{session_configs_.at("external_data_file")};

    if (path.empty()) return {};

    proto::Header header = proto::getHeader(path.string());

    external_data_ = std::make_shared<ExternalData>(
      ExternalData{std::move(header), std::move(path)}
    );

    return external_data_;
  }

  std::optional<Ort::ConstValue> GetConstInputTensor(size_t index) {
    int is_constant = 0;
    auto res = kinfo_.GetTensorConstantInput(index, &is_constant);
    if (is_constant) return res;
    return std::nullopt;
  }

 protected:
  const Ort::ConstKernelInfo kinfo_;
  const SessionConfigs& session_configs_;

 private:
  std::shared_ptr<ExternalData> external_data_;
};

struct CoreLibMatmulKernelBase : CoreLibKernelBase {
  using CoreLibKernelBase::CoreLibKernelBase;

 protected:
  template <typename T>
  static auto ExtractShape(const T& tensor) {
    const auto values = tensor.GetTensorTypeAndShapeInfo().GetShape();

    // TODO: confirm that it's okay to just drop the first element which is
    // usually '1'
    // varunsh: The first element refers to batch and it has been 1 for our
    // cases.
    if (values.size() == 3 && values[0] == 1)
      return corelib::Shape(std::next(values.begin()), values.end());

    return corelib::Shape(values.begin(), values.end());
  };

  auto MakeOutputShape(const Ort::ConstValue& input) const {
    const auto input_shape = input.GetTensorTypeAndShapeInfo().GetShape();
    return std::vector<int64_t>{input_shape[0], input_shape[1], n_};
  }

 protected:
  const int64_t block_size_ = GetAttribute<int64_t>("block_size");
  const int64_t k_ = GetAttribute<int64_t>("K");
  const int64_t n_ = GetAttribute<int64_t>("N");
  const bool cast_input_ =
    !GetAttributes<int64_t>("hybrid_llm_cast_input", {}).empty();
  const bool cast_output_ =
    !GetAttributes<int64_t>("hybrid_llm_cast_output", {}).empty();
  const bool non_bfp16_weights_ =
    GetAttribute<std::string>("is_bfp16", "") != "weights";

  corelib::MatMulOperator op_;
};

struct CoreLibMatmulKernel_Gemm : CoreLibMatmulKernelBase {
  CoreLibMatmulKernel_Gemm(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const SessionConfigs& session_configs
  )
    : CoreLibMatmulKernelBase(info, session_configs) {
    corelib::Tensor cl_weights, cl_bias = corelib::MissingTensor;

    if (auto data = LoadExternalData()) {
      if (data->header.external_data().npu()) {
        const auto dataPath =
          data->path.parent_path() / data->header.external_data().filename();
        const auto& op = data->header.operators().at(kinfo_.GetNodeName());

        // op.PrintDebugString();

        if (op.data_size() == 5) {
          const auto& weights = op.data(4);

          cl_weights = corelib::ConstFileTensor{
            corelib::DataType::Native,
            {weights.shape().begin(), weights.shape().end()},
            dataPath.string(),
            weights.offset(),
            weights_attrs_
          };
        }
      }
    }

    // if we have prepacked NPU weights
    if (!cl_weights && kinfo_.GetInputCount() > 5) {
      const auto weights = GetConstInputTensor(5);

      cl_weights = corelib::ConstTensor{
        corelib::DataType::Native, ExtractShape(*weights),
        (void*)weights->GetTensorRawData(), weights_attrs_
      };
    }

    if (!cl_weights) {
      const auto weights = GetConstInputTensor(1);
      const auto scales = GetConstInputTensor(2);
      const auto zeros = GetConstInputTensor(3);
      const auto bias = GetConstInputTensor(4);

      cl_weights = corelib::QuantizedConstTensor{
        {{corelib::DataType::Int4,
          {k_, n_},
          (void*)weights->GetTensorData<void>(),
          weights_attrs_},
         {corelib::DataType::FP16,
          {k_ / block_size_, n_},
          (void*)scales->GetTensorData<void>()},
         {corelib::DataType::Int4,
          {k_ / block_size_, n_},
          (void*)zeros->GetTensorData<void>()},
         unsigned(block_size_)}
      };

      cl_bias = corelib::ConstTensor{
        corelib::DataType::FP16, ExtractShape(*bias),
        (void*)bias->GetTensorRawData()
      };
    }

    op_ = corelib::MatMulOperator{
      {input_, std::move(cl_weights), output_, std::move(cl_bias)}
    };
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    auto input = ctx.GetInput(0);
    auto output = ctx.GetOutput(0, MakeOutputShape(input));

    input_.SetData(ExtractShape(input), (void*)input.GetTensorRawData());
    output_.SetData(ExtractShape(output), output.GetTensorMutableRawData());

    op_.Compute();
  }

 private:
  corelib::PlaceholderTensor input_{
    cast_input_ ? corelib::DataType::FP16 : corelib::DataType::BF16
  };
  corelib::PlaceholderTensor output_{
    cast_output_ ? corelib::DataType::FP16 : corelib::DataType::BF16
  };
  corelib::Dictionary weights_attrs_{
    // if set to true, older `v1` op_version is used
    {"non_bfp16", non_bfp16_weights_}
  };
};

struct CoreLibMatmulKernel_QGemm : CoreLibMatmulKernelBase {
  CoreLibMatmulKernel_QGemm(
    const OrtApi& ort_api, const OrtKernelInfo* info,
    const SessionConfigs& session_configs
  )
    : CoreLibMatmulKernelBase(info, session_configs) {
    // TODO: load weights and pass it to following function
    QuantizedConstTensor weights{
      {{DataType::Int4, {}, nullptr},
       {DataType::FP32, {}, nullptr},
       {DataType::UInt16, {}, nullptr},
       128}
    };

    input_ = QuantizedPlaceholderTensor{
      {DataType::UInt16, DataType::FP32, DataType::UInt16, 128}
    };

    // TODO: pass input scales and zeropoint value to following calls
    input_.SetScales({}, nullptr);
    input_.SetZeroes({}, nullptr);

    output_ = QuantizedPlaceholderTensor{
      {DataType::UInt16, DataType::FP32, DataType::UInt16, 128}
    };

    // TODO: pass output scales and zeropoint value to following calls
    input_.SetScales({}, nullptr);
    input_.SetZeroes({}, nullptr);

    op_ = ryzenai::corelib::MatMulOperator{
      {input_, weights, output_, ryzenai::corelib::MissingTensor}
    };
  }

  void Compute(OrtKernelContext* context) {
    // TODO: get input from context and pass it following function
    input_.SetData({}, nullptr);

    // TODO: get output from context and pass it following function
    output_.SetData({}, nullptr);

    op_.Compute();
  }

  QuantizedPlaceholderTensor input_, output_;
};

static struct AutoCleanupCoreLib {
  ~AutoCleanupCoreLib() { ryzenai_corelib_cleanup(); }
} autoCleanupCoreLib_;

template <const char* Name, typename Kernel>
struct CoreLibCustomOp : Operator<Kernel, Name> {
  CoreLibCustomOp(const Ort::ConstSessionOptions& session_options)
    : Operator(session_options, {"external_data_file"}) {}
};

std::vector<const OrtCustomOp*> create_corelib_ops(
  const Ort::ConstSessionOptions& options
) {
  static const char matmulname[] = "MatMul";
  static const char qmatmulname[] = "QMatMul";
  const auto matmul =
    GetCustomOp<CoreLibCustomOp<matmulname, CoreLibMatmulKernel_Gemm>>(options);
  const auto qmatmul =
    GetCustomOp<CoreLibCustomOp<qmatmulname, CoreLibMatmulKernel_QGemm>>(
      options
    );
  return {matmul, qmatmul};
}
}  // namespace ryzenai::onnx_utils
