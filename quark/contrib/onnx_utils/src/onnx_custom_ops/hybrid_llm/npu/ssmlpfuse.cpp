// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "ssmlpfuse.hpp"

#include "jit_node_impl.hpp"
#include "lora.hpp"
#include "npu_utils.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "profiling/profiling.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "ort.hpp"

using NPUTensor = ::Tensor;

namespace ryzenai::onnx_utils {

template class JitNode<AMDSSMlpFuseKernel>;

AMDSSMlpFuseKernel::AMDSSMlpFuseKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : JitNode(this, session_configs),
    NpuOp(k_info, session_configs),
    ss_(session_configs) {
#ifdef NPU_SS_MLP_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif

  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  auto header = initializeNpuOp(op_type_, session_configs, info);
  auto num_inputs = info.GetInputCount();
  auto up_wts_tensor = info.GetInputName(num_inputs - 2);

  shared_weights_.setWeightsCount(1);
  shared_weights_.setWeightKey(0, info, "ssmlp_wts_hash");
  this->initializeSharedBuffers(ss_->ssmlp_bo_data_);
  ss_->ssmlp_bo_data_.emplace_back();

  ort_cast_bf16_to_fp16_.construct(info);
  ort_cast_fp16_to_bf16_.construct(info);
  ort_cast_fp32_to_fp16_.construct(info);

  // kernel objects for sslrn 1 and 2
  initializeKernels();

  manageDynamicDpmState();

  ssmlp_K_ = info.GetAttribute<int64_t>("gate_K");
  ssmlp_N_ = info.GetAttribute<int64_t>("gate_N");

  int total_input_num = info.GetInputCount();
  auto expected_packed_input_num = 16;
  const bool packed_consts = (expected_packed_input_num == total_input_num);
  if (packed_consts) {
    int is_constant = 0;
    int tensor_offset = -3;
    Ort::ConstValue ssmlp_packed_consts =
      info.GetTensorConstantInput(total_input_num - 3, &is_constant);
    if (is_constant) {
      if (ssmlp_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount() >
            1 &&
          false) {
        // TODO: not supported now.
        auto ssmlp_weights_data_ =
          info.GetTensorConstantInput(13, &is_constant);
        const auto* ssmlp_wts_data_ort =
          ssmlp_weights_data_.GetTensorData<uint8_t>();

        auto dimensions_wts =
          ssmlp_weights_data_.GetTensorTypeAndShapeInfo().GetShape();
      } else {
        if (!shared_weights_.ready()) {
          jit_tensor_ssmlp_ =
            getExternalTensorInfo(header.get(), name(), tensor_offset);
          ss_->jit_max_bo_size_ssmlp_ =
            proto::getNpuMaxSize(header.get(), op_type_);
          // load weights after all BO sizes set
          this->loadFirstData();
        }
      }
    }
  }

  cnt_ = ss_->instances__++;
  try {
    input_cast_indices_ = info.GetAttributes<int64_t>("hybrid_llm_cast_input");
  } catch (const Ort::Exception&) {
    input_cast_indices_ = {0, 1};
  }
}

AMDSSMlpFuseKernel::~AMDSSMlpFuseKernel() {
  shared_buffer_.Reset();

#ifdef NPU_SS_MLP_PROFILE_EN
  std::ostringstream os;

  int event_id = 0;

  for (const auto& measurement : measurements_) {
    os << name() << ",NPUSSMLP," << event_id << ","
       << MillisecondsFp{measurement}.count() << "\n";

    event_id++;
  }

  std::cout << os.str() << std::flush;
#endif
  ss_->instances__--;
  if (ss_->instances__ == 0) {
    ss_->ssmlp_.reset();
  }
}

void AMDSSMlpFuseKernel::readDataImpl(int idx) {
  if (shared_weights_.ready()) return;

  updateJitBuffer(
    dynamicJitFactor(), ss_->ssmlp_bo_data_[idx], jit_tensor_ssmlp_.size,
    ss_->jit_max_bo_size_ssmlp_
  );

  const auto external_data = externalData().string();
  loadBin(
    ss_->ssmlp_bo_data_[idx].Data(), external_data, jit_tensor_ssmlp_.size,
    jit_tensor_ssmlp_.offset
  );
}

void AMDSSMlpFuseKernel::loadDataImpl(int idx) {
  // if we're reading everything, we don't need to use a shared buffer
  const bool use_shared_buffer = this->isJitEnabled();

  if (use_shared_buffer) {
    ss_->ssmlp_weights_.clear();
  }
  ss_->ssmlp_weights_.push_back(ss_->ssmlp_->bind_bo(
    ss_->ssmlp_bo_data_[idx].Data(), alignTo4096(jit_tensor_ssmlp_.size)
  ));
}

void AMDSSMlpFuseKernel::initializeKernels() {
  if (!ss_->ssmlp_) {
    auto attr_ssmlp = getCommonAttrs();
    const MladfVersion version = MladfVersion::aie4_v1;
    attr_ssmlp["op_version"] = version.str();
    attr_ssmlp["activation_type"] = std::string("silu");
    attr_ssmlp["activation_alpha"] = 0;
    attr_ssmlp["activation_beta"] = 0;
    attr_ssmlp["swiglu_limit"] = 0;
    attr_ssmlp["group_size"] = 128;
    ss_->ssmlp_ =
      std::make_unique<ryzenai::dynamic_dispatch::transformer::ssmlp<
        uint16_t, uint8_t, uint16_t>>(true, attr_ssmlp);
  }
  auto attr_add_1 = getCommonAttrs();
  attr_add_1.insert({{"skip_create_output", 1}, {"skip_create_input", 1}});
  ss_->add_ =
    std::make_unique<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr_add_1
    );
  ss_->seq_len_ = 0;
}

void AMDSSMlpFuseKernel::UpdateSharedBuffer(size_t kernel_size) {
  std::vector<size_t> a_shape_g = {
    1, kernel_size, static_cast<size_t>(ssmlp_K_)
  };
  std::vector<size_t> b_shape_g = {
    static_cast<size_t>(ssmlp_K_), static_cast<size_t>(ssmlp_N_)
  };
  std::vector<size_t> c_shape_g = {
    1, kernel_size, static_cast<size_t>(ssmlp_N_)
  };

  NPUTensor input_tensor_g = {nullptr, a_shape_g, "bfloat16"};
  NPUTensor wts_tensor_g = {nullptr, b_shape_g, "bfloat16"};
  NPUTensor out_tensor_g = {nullptr, c_shape_g, "bfloat16"};
  NPUTensor placeholder;

  std::vector<NPUTensor> inputs_g = {input_tensor_g, wts_tensor_g,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_g};
  std::vector<NPUTensor> inputs_add = {input_tensor_g, input_tensor_g};
  std::vector<NPUTensor> outputs_add = {input_tensor_g};
  auto size_map_add = get_NPU_tensor_size(
    ss_->add_->get_buffer_reqs(inputs_add, outputs_add), mladfVersion()
  );
  std::vector<size_t> a_shape = {2, kernel_size, ssmlp_K_};
  std::vector<size_t> out_shape = {2, kernel_size, ssmlp_K_};
  Tensor input0_tensor = {nullptr, a_shape, "bfloat16"};
  Tensor input1_tensor = {nullptr, {ssmlp_K_}, "bfloat16"};
  Tensor input2_tensor = {nullptr, {ssmlp_K_ * 2, ssmlp_N_}, "bfloat16"};
  Tensor input3_tensor = {nullptr, {ssmlp_N_, ssmlp_K_}, "bfloat16"};
  Tensor input4_tensor = {nullptr, {ssmlp_K_}, "bfloat16"};
  Tensor out_tensor = {nullptr, out_shape, "bfloat16"};

  std::vector<Tensor> input_tensors = {
    input0_tensor, input1_tensor, input2_tensor, input3_tensor, input4_tensor
  };
  std::vector<Tensor> output_tensors = {out_tensor};

  std::vector<OpArgMap> arg_map =
    ss_->ssmlp_->get_buffer_reqs(input_tensors, output_tensors);
  auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());
  size_t input_bo_size = size_map["in0"];
  ssmlp_scratch_bo_size_ = size_map["scratch"];
  size_t output_bo_size = size_map["out"];

  SharedBuffer::Requirements shared_buffer_reqs{
    {"add", alignTo4096(size_map_add["in0"])},
    {"add1", alignTo4096(size_map_add["in0"])},
    {"ssmlp_in", alignTo4096(input_bo_size)},
    {"ssmlp_out", alignTo4096(output_bo_size)}
  };

  if (ssmlp_scratch_bo_size_ > 0) {
    shared_buffer_reqs.emplace_back(
      "scratch", alignTo4096(ssmlp_scratch_bo_size_)
    );
  }

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}

void AMDSSMlpFuseKernel::execute_ssmlp(
  const uint16_t* input_data, std::vector<int64_t>& input_shape,
  const uint16_t* skip_data, std::vector<int64_t>& skip_shape, int grp_size,
  int run_cnt, bool sync_input
) {
  auto input_elements = ssmlp_seq_len_ * ssmlp_K_;
  auto skip_elements = input_elements;

  uint16_t* input_bo_map = ss_->ssmlp_inputs_.map<uint16_t*>();
  memcpy(
    (void*)input_bo_map, (void*)input_data, input_elements * sizeof(uint16_t)
  );
  uint16_t* skip_bo_map = ss_->ssmlp_inputs_.map<uint16_t*>() + input_elements;
  memcpy(
    (void*)skip_bo_map, (void*)skip_data, skip_elements * sizeof(uint16_t)
  );
  Tensor input_tensor = {
    input_bo_map, {2, ssmlp_seq_len_, ssmlp_K_}, "bfloat16"
  };
  ss_->ssmlp_->pad_input(input_tensor);

  ss_->ssmlp_inputs_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ss_->ssmlp_weights_[run_cnt].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<NPUBufferSpan> inputs = {
    {ss_->ssmlp_inputs_, 0, ss_->ssmlp_inputs_.size()},
    {ss_->ssmlp_weights_[run_cnt], 0, ss_->ssmlp_weights_[run_cnt].size()}
  };

  if (ssmlp_scratch_bo_size_ > 0) {
    NPUBufferSpan span;
    span.bo = ss_->ssmlp_scratch_;
    span.offset = 0;
    span.size = ss_->ssmlp_scratch_.size();
    inputs.push_back(span);
  }

  std::vector<NPUBufferSpan> outputs = {
    {ss_->ssmlp_outputs_, 0, ss_->ssmlp_outputs_.size()}
  };

  PROFILING_START(execute_ssmlp)
  if (true) {
    conditionalTry([&]() { ss_->ssmlp_->run(inputs, outputs); }, true, name());
  } else {
    std::map<std::string, std::any> attr = {};
    auto run = ss_->ssmlp_->create_run(inputs, outputs, attr);
    run.value().start();
  }
  PROFILING_END(execute_ssmlp, false, name().c_str())

  ss_->ssmlp_outputs_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint16_t* output_bo_map = ss_->ssmlp_outputs_.map<uint16_t*>();
  Tensor output_tensor = {
    output_bo_map, {2, ssmlp_seq_len_, ssmlp_K_}, "bfloat16"
  };
  ss_->ssmlp_->pad_output(output_tensor);
}

// Kernel Compute
void AMDSSMlpFuseKernel::Compute(OrtKernelContext* context) {
  PROFILING_START(Compute)
  PROFILING_START(setup)
#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif
  shared_weights_.setWeightAddr(0);

  int cnt = cnt_;
  bool wait_for_data = false;

  if (useExternalData()) {
    auto t1 = std::chrono::high_resolution_clock::now();

    this->readData();
    wait_for_data = true;
  }

  Ort::KernelContext ctx(context);
  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  initializeKernels();

  PROFILING_END(setup, false, name().c_str())

  PROFILING_START(input_format)
  auto input = ctx.GetInput(0);  // Input
  auto skip = ctx.GetInput(1);   // skip

  auto dimensions_input = input.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_skip = skip.GetTensorTypeAndShapeInfo().GetShape();

  size_t B = dimensions_input[0];  // Batch
  size_t M = dimensions_input[1];  // Seq len
  size_t K = dimensions_input[2];  // Hidden size = Num_heads * Head_size
  const size_t num_elements = B * M * K;

  auto npu_kernel_size = getNPUKernelGranularity(M);

  ssmlp_seq_len_ = M;

  UpdateSharedBuffer(npu_kernel_size);

  auto in_data = input.GetTensorData<uint16_t>();
  auto skip_data = skip.GetTensorData<uint16_t>();

  std::vector<bool> input_cast = {false, false};
  bool output_cast = false;

  uint16_t* input_ptr = nullptr;
  uint16_t* skip_ptr = nullptr;
  if (!input_cast_indices_.empty()) {
    for (int i = 0; i < input_cast_indices_.size(); i++) input_cast[i] = true;
  }

  if (auto rebind = shared_buffer_.Validate("add", ss_->add_->get_inputs()[0]))
    ss_->add_->create_bo(rebind->ptr, rebind->len, 0);

  if (auto rebind = shared_buffer_.Validate("add1", ss_->add_->get_inputs()[1]))
    ss_->add_->create_bo(rebind->ptr, rebind->len, 1);

  bool need_op0_copy = true;
  // Create NPU input tensor
  auto input_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  if (input_cast[0]) {
    RecordDuration(Metric::Casting, [&]() {
      auto add_op0 = ss_->add_->get_inputs()[0];
      uint16_t* add_input_0_map = add_op0.template map<uint16_t*>();

      ort_cast_fp16_to_bf16_.execute(
        (Ort::BFloat16_t*)add_input_0_map, input, context
      );
      input_ptr = add_input_0_map;
      need_op0_copy = false;

      const size_t add0_bo_size = input_count * sizeof(std::uint16_t);
      const size_t add0_bo_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        add_op0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      });
    });
  } else
    input_ptr = (uint16_t*)in_data;

  if (input_cast[1]) {
    RecordDuration(Metric::Casting, [&]() {
      // Create NPU skip tensor
      auto input_count_skip =
        skip.GetTensorTypeAndShapeInfo().GetElementCount();
      npu_skip_buffer_.resize(input_count_skip);

      ort_cast_fp16_to_bf16_.execute(
        (Ort::BFloat16_t*)npu_skip_buffer_.data(), skip, context
      );
      skip_ptr = (uint16_t*)npu_skip_buffer_.data();
    });
  } else
    skip_ptr = (uint16_t*)skip_data;

  PROFILING_END(input_format, false, name().c_str())
#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point input_format_start = Clock::now();
#endif

#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point input_format_end = Clock::now();
#endif
  std::vector<size_t> a_shape = {M, K};

  if (wait_for_data) {
    auto ready = this->weightsReady();
    if (ready < 0) {
      std::cerr << "Disable NPU JIT with `hybrid_opt_npu_read_ahead='-1'` in "
                   "session options\n";
      throw std::invalid_argument("Weights not loaded in " + name());
    }
    if (this->isJitEnabled()) {
      cnt = 0;
    }
  }

  if (auto res = shared_buffer_.Validate("ssmlp_in", ss_->ssmlp_inputs_)) {
    ss_->ssmlp_inputs_ = ss_->ssmlp_->bind_bo(res->ptr, res->len);
  }

  if (auto res = shared_buffer_.Validate("ssmlp_out", ss_->ssmlp_outputs_)) {
    ss_->ssmlp_outputs_ = ss_->ssmlp_->bind_bo(res->ptr, res->len);
  }

  if (ssmlp_scratch_bo_size_ > 0) {
    if (auto res = shared_buffer_.Validate("scratch", ss_->ssmlp_scratch_)) {
      if (res->len > 0) {
        ss_->ssmlp_scratch_ = ss_->ssmlp_->bind_bo(res->ptr, res->len);
      }
    }
  }

  std::vector<size_t> in_shape = {2, ssmlp_seq_len_, ssmlp_K_};
  std::vector<size_t> out_shape = {2, ssmlp_seq_len_, ssmlp_K_};
  Tensor input0_tensor = {nullptr, in_shape, "bfloat16"};
  Tensor input1_tensor = {nullptr, {ssmlp_K_}, "bfloat16"};
  Tensor input2_tensor = {nullptr, {ssmlp_K_ * 2, ssmlp_N_}, "bfloat16"};
  Tensor input3_tensor = {nullptr, {ssmlp_N_, ssmlp_K_}, "bfloat16"};
  Tensor input4_tensor = {nullptr, {ssmlp_K_}, "bfloat16"};
  Tensor out_tensor = {nullptr, out_shape, "bfloat16"};

  std::vector<Tensor> input_tensors = {
    input0_tensor, input1_tensor, input2_tensor, input3_tensor, input4_tensor
  };
  std::vector<Tensor> output_tensors = {out_tensor};
  ss_->ssmlp_->set_tensor_shape(input_tensors, output_tensors);

  execute_ssmlp(
    input_ptr, dimensions_input, skip_ptr, dimensions_skip, 128, cnt, true
  );

  uint16_t* output_data_input =
    ss_->ssmlp_outputs_.map<uint16_t*>() + ssmlp_seq_len_ * ssmlp_K_;
  auto output_input = ctx.GetOutput(0, dimensions_input);
  auto input_out_data = output_input.GetTensorMutableData<uint16_t>();
  if (!output_cast_indices_.empty()) {
    RecordDuration(Metric::Casting, [&]() {
      ort_cast_bf16_to_fp16_.execute(
        (Ort::Float16_t*)input_out_data, (Ort::BFloat16_t*)output_data_input,
        dimensions_input, context
      );
    });
  } else {
    MemCpy(
      input_out_data, output_data_input,
      ssmlp_seq_len_ * ssmlp_K_ * sizeof(uint16_t)
    );
  }

  if (num_outputs == 2) {
    uint16_t* output_data_skip = ss_->ssmlp_outputs_.map<uint16_t*>();
    auto output_skip = ctx.GetOutput(1, dimensions_skip);
    auto skip_out_data = output_skip.GetTensorMutableData<uint16_t>();
    if (!output_cast_indices_.empty()) {
      RecordDuration(Metric::Casting, [&]() {
        ort_cast_bf16_to_fp16_.execute(
          (Ort::Float16_t*)skip_out_data, (Ort::BFloat16_t*)output_data_skip,
          dimensions_input, context
        );
      });
    } else {
      MemCpy(
        skip_out_data, output_data_skip,
        ssmlp_seq_len_ * ssmlp_K_ * sizeof(uint16_t)
      );
    }
  }

  if (freeAfterPrefill(name())) {
    ss_->ssmlp_.reset();
    if (lastNode(name()) && getPrefillBufferRelease(npu_kernel_size)) {
      shared_buffer_.Reset();
    }
  }

  if (useExternalData()) {
    this->loadData();
    this->unloadData();
  }
  PROFILING_END(Compute, false, name().c_str())
}

}  // namespace ryzenai::onnx_utils
