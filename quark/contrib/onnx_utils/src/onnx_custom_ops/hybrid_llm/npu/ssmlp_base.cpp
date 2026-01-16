// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "ssmlp_base.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "common.hpp"
#include "jit_node_impl.hpp"
#include "lora.hpp"
#include "profiling/profiling.hpp"
#include "ssgmlp.hpp"
#include "ssmlp.hpp"

namespace ryzenai::onnx_utils {

using NPUTensor = ::Tensor;

template class SSMLPBase<AMDSSGMLPKernel>;
template class SSMLPBase<JitNode<AMDSSGMLPKernel>>;

template class SSMLPBase<AMDSSMLPKernel>;
template class SSMLPBase<JitNode<AMDSSMLPKernel>>;

template <typename T>
SSMLPBase<T>::SSMLPBase(
  JitNode<T>* op_inter, const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs,
  bool is_ssgmlp
)
  : JitNode<T>(op_inter, session_configs),
    NpuOp(k_info, session_configs),
    is_ssgmlp_(is_ssgmlp),
    ss_(session_configs) {
#ifdef NPU_SS_MLP_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif

  op_type_ = is_ssgmlp_ ? ssgmlp_op_type_ : ssmlp_op_type_;

  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  auto header = initializeNpuOp(op_type_, session_configs, info);
  auto num_inputs = info.GetInputCount();
  auto up_wts_tensor = info.GetInputName(num_inputs - 2);
  gate_up_fused_ = up_wts_tensor == "";
  // info.GetInputName(9) + ".bin";
  if (!gate_up_fused_) {
    shared_weights_.setWeightsCount(3);
    shared_weights_.setWeightKey(0, info, "gate_wts_hash");
    shared_weights_.setWeightKey(1, info, "up_wts_hash");
    shared_weights_.setWeightKey(2, info, "down_wts_hash");
    this->initializeSharedBuffers(ss_->bo_data_gate_);
    this->initializeSharedBuffers(ss_->bo_data_up_);
    this->initializeSharedBuffers(ss_->bo_data_down_);
  } else {
    shared_weights_.setWeightsCount(2);
    shared_weights_.setWeightKey(0, info, "gate_up_wts_hash");
    shared_weights_.setWeightKey(1, info, "down_wts_hash");
    this->initializeSharedBuffers(ss_->bo_data_gate_);
    this->initializeSharedBuffers(ss_->bo_data_down_);
  }

  ort_cast_bf16_to_fp16_.construct(info);
  ort_cast_fp16_to_bf16_.construct(info);
  ort_cast_fp32_to_fp16_.construct(info);

  epsilon_ = info.GetAttribute<float>("epsilon");
  ort_sslrn_.construct(info);

  if (is_ssgmlp_) {
    ort_slrn_.construct(info);
  }

  // kernel objects for sslrn 1 and 2
  initializeKernels();

  manageDynamicDpmState();

  int is_constant = 0;
  int ssln1_wts_idx = is_ssgmlp_ ? 3 : 2;
  m_weights = info.GetTensorConstantInput(ssln1_wts_idx, &is_constant);
  const auto* wts_data_ort = m_weights.GetTensorData<Ort::Float16_t>();

  auto dimensions_wts = m_weights.GetTensorTypeAndShapeInfo().GetShape();

  num_el = std::accumulate(
    dimensions_wts.begin(), dimensions_wts.end(), (size_t)1,
    std::multiplies<int64_t>()
  );
  wts_data_.reserve(num_el);
  for (int i = 0; i < num_el; i++) {
    wts_data_.push_back(wts_data_ort[i].ToFloat());
  }

  wts_ =
    init_rmsnorm_wts(ss_->rms_norm_.get(), epsilon_, wts_data_, allocator_);
  num_el_bo = wts_.size();

  is_constant = 0;
  if (is_ssgmlp_) {
    m2_weights =
      info.GetTensorConstantInput(14, &is_constant);  // ssgmlp gemma pattern
  } else {
    m2_weights =
      info.GetTensorConstantInput(12, &is_constant);  // ssmlp gemma pattern
  }

  const auto* wts2_data_ort = m2_weights.GetTensorData<Ort::Float16_t>();

  auto dimensions_wts2 = m2_weights.GetTensorTypeAndShapeInfo().GetShape();

  num_el2 = m2_weights.GetTensorTypeAndShapeInfo().GetElementCount();
  wts2_data_.reserve(num_el2);
  for (int i = 0; i < num_el2; i++) {
    wts2_data_.push_back(wts2_data_ort[i].ToFloat());
  }

  wts2_ =
    init_rmsnorm_wts(ss_->rms_norm_.get(), epsilon_, wts2_data_, allocator_);
  num_el_bo2 = wts2_.size();

  if (is_ssgmlp_) {
    // ssgmlp gemma ssmlp first additional simplified layer norm
    is_constant = 0;
    m3_weights = info.GetTensorConstantInput(2, &is_constant);
    const auto* wts3_data_ort = m3_weights.GetTensorData<Ort::Float16_t>();

    auto dimensions_wts3 = m3_weights.GetTensorTypeAndShapeInfo().GetShape();

    num_el3 = m3_weights.GetTensorTypeAndShapeInfo().GetElementCount();
    wts3_data_.reserve(num_el3);
    for (int i = 0; i < num_el3; i++) {
      wts3_data_.push_back(wts3_data_ort[i].ToFloat());
    }
    wts3_ =
      init_rmsnorm_wts(ss_->rms_norm_.get(), epsilon_, wts3_data_, allocator_);
    num_el_bo3 = wts3_.size();
    // ssgmlp gemma ssmlp second additional simplified layer norm
    is_constant = 0;
    m4_weights = info.GetTensorConstantInput(13, &is_constant);
    const auto* wts4_data_ort = m4_weights.GetTensorData<Ort::Float16_t>();

    auto dimensions_wts4 = m4_weights.GetTensorTypeAndShapeInfo().GetShape();
    num_el4 = m4_weights.GetTensorTypeAndShapeInfo().GetElementCount();
    wts4_data_.reserve(num_el4);
    for (int i = 0; i < num_el4; i++) {
      wts4_data_.push_back(wts4_data_ort[i].ToFloat());
    }
    wts4_ =
      init_rmsnorm_wts(ss_->rms_norm_.get(), epsilon_, wts4_data_, allocator_);
    num_el_bo4 = wts4_.size();
  }
  // MY_LOG(2) << "initialization for SSLRN done." << std::endl;

  // MLP
  //  Extracting the attribute information
  //  Gate proj
  // MY_LOG(2) << "initialization for MLP begin" << std::endl;
  gp_k = info.GetAttribute<int64_t>("gate_K");
  gp_n = info.GetAttribute<int64_t>("gate_N");
  gp_bits = info.GetAttribute<int64_t>("gate_bits");
  gp_block_size = info.GetAttribute<int64_t>("gate_block_size");
  // Up proj
  up_k = info.GetAttribute<int64_t>("up_K");
  up_n = info.GetAttribute<int64_t>("up_N");
  up_bits = info.GetAttribute<int64_t>("up_bits");
  up_block_size = info.GetAttribute<int64_t>("up_block_size");
  // Down proj
  dp_k = info.GetAttribute<int64_t>("down_K");
  dp_n = info.GetAttribute<int64_t>("down_N");
  dp_bits = info.GetAttribute<int64_t>("down_bits");
  dp_block_size = info.GetAttribute<int64_t>("down_block_size");

  try {
    input_cast_indices_ = info.GetAttributes<int64_t>("hybrid_llm_cast_input");
  } catch (const Ort::Exception&) {
    input_cast_indices_ = {0, 1};
  }

  try {
    output_cast_indices_ =
      info.GetAttributes<int64_t>("hybrid_llm_cast_output");
  } catch (const Ort::Exception&) {
    output_cast_indices_ = {0, 1};
  }

  // mladfmatmulbias operator handles

  auto attrs = getCommonAttrs();
  updateAttrsForSharedWeights(attrs);

  if (ss_->instances__ == 0) {
    auto attr_gate = attrs;
    if (gate_up_fused_) attr_gate["wts_interleaved"] = true;
    // Create qlinear-2 handle
    ss_->gate_proj_ = std::make_shared<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, attr_gate
    );
  }
  auto gp_ptr =
    (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
      ss_->gate_proj_.get();

  if (ss_->instances__ == 0) {
    // Create qlinear-2 handle
    ss_->up_proj_ = std::make_shared<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, attrs
    );
  }
  auto up_ptr =
    (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
      ss_->up_proj_.get();

  if (ss_->instances__ == 0) {
    // Create qlinear-2 handle
    ss_->down_proj_ = std::make_shared<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, attrs
    );
  }
  auto dp_ptr =
    (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
      ss_->down_proj_.get();

  int total_input_num = info.GetInputCount();
  auto expected_packed_input_num = is_ssgmlp_ ? 18 : 16;
  const bool packed_consts = (expected_packed_input_num == total_input_num);
  if (!packed_consts) {
    is_constant = 0;
    int base_idx = 3;
    if (is_ssgmlp_) base_idx = 4;

    auto gp_weights_tensor =
      info.GetTensorConstantInput(base_idx, &is_constant);
    ;
    const int8_t* gp_wts = (gp_weights_tensor.GetTensorData<int8_t>());

    is_constant = 0;
    auto gp_scales_tensor =
      info.GetTensorConstantInput(base_idx + 1, &is_constant);
    const auto* gp_scl = gp_scales_tensor.GetTensorData<Ort::Float16_t>();

    int is_gpz_constant = 0;
    auto gp_zeros = info.GetTensorConstantInput(base_idx + 2, &is_gpz_constant);
    const int8_t* gp_zps = gp_zeros.GetTensorData<int8_t>();

    is_constant = 0;
    auto up_weights_tensor =
      info.GetTensorConstantInput(base_idx + 3, &is_constant);
    const int8_t* up_wts = (up_weights_tensor.GetTensorData<int8_t>());
    is_constant = 0;
    auto up_scales_tensor =
      info.GetTensorConstantInput(base_idx + 4, &is_constant);
    const auto* up_scl = up_scales_tensor.GetTensorData<Ort::Float16_t>();
    int is_upz_constant = 0;
    auto up_zeros = info.GetTensorConstantInput(base_idx + 5, &is_upz_constant);
    const int8_t* up_zps = up_zeros.GetTensorData<int8_t>();

    is_constant = 0;
    auto dp_weights_tensor =
      info.GetTensorConstantInput(base_idx + 6, &is_constant);
    const int8_t* dp_wts = (dp_weights_tensor.GetTensorData<int8_t>());
    is_constant = 0;
    auto dp_scales_tensor =
      info.GetTensorConstantInput(base_idx + 7, &is_constant);
    const auto* dp_scl = dp_scales_tensor.GetTensorData<Ort::Float16_t>();
    int is_dpz_constant = 0;
    auto dp_zeros = info.GetTensorConstantInput(base_idx + 8, &is_dpz_constant);
    const int8_t* dp_zps = dp_zeros.GetTensorData<int8_t>();
    // MY_LOG(2) << "Got attributes for MLP" << std::endl;

    /////////////////////////// Gate /////////////////////////////////
    std::vector<float> gp_bias(gp_n, 0);  // fill with zeros
    std::vector<float> gp_scales(gp_k * gp_n / gp_block_size);
    std::vector<int8_t> gp_weights(gp_k * gp_n, 0);
    // fill this with zeros for Symmetric quantization
    std::vector<int8_t> gp_zpoints(
      gp_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0
    );

    size_t gp_kblks = gp_k / gp_block_size;

    // mladf version 2 only support uint input, so no need to correct data to
    // int
    uint8_t wts_type_off = 8;
    extractAndTransposeWeights(gp_wts, gp_weights, gp_k, gp_n, wts_type_off);

    transposeScales(gp_scl, gp_scales, gp_n, gp_kblks);

    int64_t gzp_shape =
      (gp_n * std::floor((float)((gp_kblks + 1) * gp_bits) / 8.0f));
    // fill this with zeros for Symmetric quantization
    if (is_gpz_constant) {
      int gp_kblks_pad = 2 * gzp_shape / gp_n;
      extractZeroPoints(gp_zps, gp_zpoints, gp_n, gp_kblks_pad, wts_type_off);
    }

    /////////////////////////// Up /////////////////////////////////
    std::vector<float> up_bias(up_n, 0);  // fill with zeros
    std::vector<float> up_scales(up_k * up_n / up_block_size);
    std::vector<int8_t> up_weights(up_k * up_n, 0);
    // fill this with zeros for Symmetric quantization
    std::vector<int8_t> up_zpoints(
      up_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0
    );

    size_t up_kblks = up_k / up_block_size;

    extractAndTransposeWeights(up_wts, up_weights, up_k, up_n, wts_type_off);

    transposeScales(up_scl, up_scales, up_n, up_kblks);

    int64_t uzp_shape =
      (up_n * std::floor((float)((up_kblks + 1) * up_bits) / 8.0f));
    // fill this with zeros for Symmetric quantization
    if (is_upz_constant) {
      int up_kblks_pad = 2 * uzp_shape / up_n;
      extractZeroPoints(up_zps, up_zpoints, up_n, up_kblks_pad, wts_type_off);
    }

    /////////////////////////// Down /////////////////////////////////
    std::vector<float> dp_bias(dp_n, 0);  // fill with zeros
    std::vector<float> dp_scales(dp_k * dp_n / dp_block_size);
    std::vector<int8_t> dp_weights(dp_k * dp_n, 0);
    // fill this with zeros for Symmetric quantization
    std::vector<int8_t> dp_zpoints(
      dp_zeros.GetTensorTypeAndShapeInfo().GetElementCount() * 2, 0
    );

    size_t dp_kblks = dp_k / dp_block_size;
    int64_t zp_shape =
      (dp_n * std::floor((float)((dp_kblks + 1) * dp_bits) / 8.0f));

    extractAndTransposeWeights(dp_wts, dp_weights, dp_k, dp_n, wts_type_off);

    transposeScales(dp_scl, dp_scales, dp_n, dp_kblks);

    // fill this with zeros for Symmetric quantization
    if (is_dpz_constant) {
      int dp_kblks_pad = 2 * zp_shape / dp_n;
      extractZeroPoints(dp_zps, dp_zpoints, dp_n, dp_kblks_pad, wts_type_off);
    }

    initializeProjection(
      gp_ptr, gp_weights, gp_scales, gp_zpoints, gp_bias, gp_k, gp_n,
      gp_block_size, true, true
    );

    initializeProjection(
      up_ptr, up_weights, up_scales, up_zpoints, up_bias, up_k, up_n,
      up_block_size, true, true
    );

    initializeProjection(
      dp_ptr, dp_weights, dp_scales, dp_zpoints, dp_bias, dp_k, dp_n,
      dp_block_size, true, true
    );
  } else {
    // packed_consts = true
    is_constant = 0;
    int tensor_offset = gate_up_fused_ ? -2 : -3;
    Ort::ConstValue gp_packed_consts =
      info.GetTensorConstantInput(total_input_num - 3, &is_constant);
    if (is_constant) {
      if (gp_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount() > 1) {
        const uint8_t* value = gp_packed_consts.GetTensorData<uint8_t>();
        initialize(
          value, gp_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount(),
          gp_block_size, gp_ptr, gp_k, gp_n, 0, true
        );
      } else {
        if (!shared_weights_.ready()) {
          jit_tensor_gate_ =
            getExternalTensorInfo(header.get(), name(), tensor_offset);
          ss_->jit_max_bo_size_gate_ =
            proto::getNpuMaxSize(header.get(), op_type_);
        }
      }
    }

    if (!gate_up_fused_) {
      is_constant = 0;
      Ort::ConstValue up_packed_consts =
        info.GetTensorConstantInput(total_input_num - 2, &is_constant);
      if (is_constant) {
        if (up_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount() >
            1) {
          const uint8_t* value = up_packed_consts.GetTensorData<uint8_t>();
          initialize(
            value,
            up_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount(),
            up_block_size, up_ptr, up_k, up_n, 0, true
          );
        } else {
          if (!shared_weights_.ready()) {
            jit_tensor_up_ = getExternalTensorInfo(header.get(), name(), -2);
            ss_->jit_max_bo_size_up_ =
              proto::getNpuMaxSize(header.get(), op_type_);
          }
        }
      }
    }
    is_constant = 0;
    Ort::ConstValue dp_packed_consts =
      info.GetTensorConstantInput(total_input_num - 1, &is_constant);
    if (is_constant) {
      if (dp_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount() > 1) {
        const uint8_t* value = dp_packed_consts.GetTensorData<uint8_t>();
        initialize(
          value, dp_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount(),
          dp_block_size, dp_ptr, dp_k, dp_n, 0, true
        );
      } else {
        if (!shared_weights_.ready()) {
          jit_tensor_down_ = getExternalTensorInfo(header.get(), name(), -1);

          ss_->jit_max_bo_size_down_ =
            proto::getNpuMaxSize(header.get(), op_type_);

          // load weights after all BO sizes set
          this->loadFirstData();
        }
      }
    }
  }

  if (ss_->instances__ == 0) {
    auto attr_mul = getCommonAttrs();
    attr_mul.insert({{"skip_create_input", 1}, {"skip_create_output", 1}});
    // ElwMul
    ss_->ewmul_ =
      std::make_shared<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_mul
      );
    if (is_ssgmlp_) {
      attr_mul["op_name"] = std::string("gelu");
    }
    // Silu
    if (!gate_up_fused_) {
      // Silu
      ss_->silu_ = std::make_shared<ryzenai::silu<uint16_t, uint16_t>>(
        "bfloat16", true, attr_mul
      );
    } else {
      // Silu

      attr_mul["activation_type"] = std::string("silu");
      ss_->silu_ = std::make_shared<ryzenai::dynamic_dispatch::transformer::
                                      general_activation<uint16_t, uint16_t>>(
        true, attr_mul
      );
      std::map<std::string, std::any> attr;
      std::vector<Tensor> const_params;
      auto silu = (ryzenai::dynamic_dispatch::transformer::general_activation<
                   uint16_t, uint16_t>*)ss_->silu_.get();
      silu->initialize_const_params(const_params, attr);
    }
  }
  UpdateSharedBuffer(
    getNPUKernelGranularity(getInitPromptSize(session_configs))
  );

  initializeLora();

  // Register for LoRA loading via LoraCompute/LoraComputeFromBuffer
  Lora::addLoraOp(this);

  cnt_ = ss_->instances__++;
  // MY_LOG(2) << "SSMLP- Init MLP done";
#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point config_end = Clock::now();
  const Duration config_duration = config_end - config_start;
  measurements_.at(EventID::CONFIG_ID) += config_duration;
#endif
}

template <typename T>
SSMLPBase<T>::~SSMLPBase() {
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
    for (int i = 0; i < this->readAhead(); i++) {
      ss_->bo_data_down_[i] = {};
      ss_->bo_data_gate_[i] = {};
      ss_->bo_data_up_[i] = {};
    }
    // free some mem
    ss_->ewmul_.reset();
    ss_->silu_.reset();
    ss_->gate_proj_.reset();
    ss_->up_proj_.reset();
    ss_->down_proj_.reset();
  }
}

template <typename T>
void SSMLPBase<T>::initialize(
  const uint8_t* preformat_consts, size_t size, int block_size,
  ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* ptr, int k,
  int n, int bo_size, bool skip_input, bool use_shared_buffer
) {
  // Weights shape
  std::vector<size_t> consts_shape_dd = {size};
  // Constant tensors
  Tensor packed_const_tensor = {
    (uint8_t*)preformat_consts, consts_shape_dd, "uint8"
  };
  Tensor weight_tensor = {
    nullptr, {static_cast<size_t>(k), static_cast<size_t>(n)}, "int4"
  };
  std::vector<Tensor> constant_tensors = {weight_tensor, packed_const_tensor};

  // Initialize constant tensors (setting up XRT BOs)
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["max_m"] = maxSeqLength();
  attrs["group_size"] = static_cast<int>(block_size);
  attrs["num_preformat_tensors"] = 1;
  attrs["tensor_size"] = use_shared_buffer ? static_cast<int>(size) : 0;
  attrs["use_host_buffer"] = 1;
  attrs["skip_create_output"] = 1;
  if (skip_input) {
    attrs["skip_create_input"] = 1;
  }

  ptr->initialize_const_params(constant_tensors, attrs);
}

template <typename T>
void SSMLPBase<T>::initializeKernels() {
  /*
    these are 2 skipsimplifiedlayernorm that are at
    beginning and end

    x_0 -> add -> x2 -> rmsnorm -> x3
    x_1           w0

    x2, x3 are global states

    x3 fed to up/gate
    x2 fed to second layer

    x2 -> add -> x4 -> rmsnorm -> x5
    dp           w1

    x4/x5 are ouputs

    For buffer space optimization, translate to

    x_0 -> add -> x0 -> rmsnorm -> x1
    x_1           w0

    x0, x1 are global states

    x1 fed to up/gate
    x0 fed to second layer

    x0 -> add -> x0 -> rmsnorm -> x2
    dp           w1

    x0/x1 are input buffers for add
    w0/w1 are const buffers of rmsnorm
    x2 is output buffer for rmsnorm

    use op attributes to skip creating buffers for DD operators
  */
  if (mladfVersion() != "v1" && mladfVersion() != "v2") {
    std::cerr << "Invalid version: " << mladfVersion() << std::endl;
  }

  if (is_ssgmlp_ && mladfVersion() != "v2") {
    std::cerr << "Only v2 supported for ssgmlp: " << mladfVersion()
              << std::endl;
  }

  if (!ss_->rms_norm_) {
    auto attr_rmsnorm_1 = getCommonAttrs();
    attr_rmsnorm_1.insert(
      {{"skip_create_input_a", 1}, {"skip_create_output", 1}}
    );

    auto attr_add_1 = getCommonAttrs();
    attr_add_1.insert({{"skip_create_output", 1}, {"skip_create_input", 1}});

    // SSLRN1
    ss_->rms_norm_ =
      std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_rmsnorm_1
      );
    ss_->add_ =
      std::make_unique<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_add_1
      );
    ss_->seq_len_ = 0;

    // SSLRN2
    auto attr_rmsnorm_2 = getCommonAttrs();
    attr_rmsnorm_2.insert(
      {{"skip_create_input_a", 1}, {"skip_create_output", 1}}
    );

    ss_->rms_norm2_ =
      std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_rmsnorm_2
      );

    if (is_ssgmlp_) {
      // ssgmlp gemma ssmlp first additional simplified layer norm

      // for Gemma2, need to use eps6 to fix accuracy issue
      std::map<std::string, std::any> attr_rmsnorm_3 = {
        {"op_version", mladfVersion()}
      };
      ss_->rms_norm3_ =
        std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, attr_rmsnorm_3
        );

      std::vector<Tensor> const_Tensor3;
      ss_->rms_norm3_->initialize_const_params(const_Tensor3);

      // ssgmlp gemma ssmlp second additional simplified layer norm
      // for Gemma2, need to use eps6 to fix accuracy issue
      std::map<std::string, std::any> attr_rmsnorm_4 = {
        {"op_version", mladfVersion()}
      };
      ss_->rms_norm4_ =
        std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, attr_rmsnorm_4
        );

      std::vector<Tensor> const_Tensor4;
      ss_->rms_norm4_->initialize_const_params(const_Tensor4);
    }
  }
}

template <typename T>
void SSMLPBase<T>::extractAndTransposeWeights(
  const int8_t* src_weights, std::vector<int8_t>& dst_weights, int64_t k,
  int64_t n, uint8_t type_offset
) {
  // Original weights are in NxK/2 packed as uint8
  // Convert to KxN uint8

  for (int64_t i = 0; i < k; i += 2) {
    for (int64_t j = 0; j < n; j++) {
      auto srcv = src_weights[j * k / 2 + i / 2];
      auto src0 = (srcv & 0xf) - type_offset;
      auto src1 = ((srcv & 0xf0) >> 4) - type_offset;
      dst_weights[i * n + j] = static_cast<int8_t>(src0);
      dst_weights[(i + 1) * n + j] = static_cast<int8_t>(src1);
    }
  }
}

template <typename T>
void SSMLPBase<T>::transposeScales(
  const Ort::Float16_t* src_scales, std::vector<float>& dst_scales, int n,
  int kblks
) {
  // Original Scales are in Nx(K/BlockSize) shape
  // Convert to (K/BLOCK_SIZE)xN shape
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < kblks; j++) {
      dst_scales[j * n + i] = src_scales[i * kblks + j].ToFloat();
    }
  }
}

template <typename T>
void SSMLPBase<T>::extractZeroPoints(
  const int8_t* src_zps, std::vector<int8_t>& dst_zpoints, int n, int kblks_pad,
  uint8_t type_offset
) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < kblks_pad; j = j + 2) {
      auto zpv = src_zps[((i * kblks_pad) / 2) + (j / 2)];
      dst_zpoints[j * n + i] = (zpv & 0xf) - type_offset;
      dst_zpoints[(j + 1) * n + i] = ((zpv & 0xf0) >> 4) - type_offset;
    }
  }
}

template <typename T>
void SSMLPBase<T>::initializeProjection(
  ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>* ptr,
  const std::vector<int8_t>& weights, const std::vector<float>& scales,
  const std::vector<int8_t>& zpoints, const std::vector<float>& bias, int k,
  int n, int block_size, bool skip_input, bool skip_output
) {
  std::vector<size_t> wts_shape_dd = {
    static_cast<size_t>(k), static_cast<size_t>(n)
  };

  Tensor wts_tensor = {
    const_cast<int8_t*>(weights.data()), wts_shape_dd, "int4"
  };
  Tensor scl_tensor = {
    const_cast<float*>(scales.data()), {(size_t)block_size, 1}, "float"
  };
  Tensor zps_tensor = {
    const_cast<int8_t*>(zpoints.data()), wts_shape_dd, "int4"
  };
  Tensor bias_tensor = {
    const_cast<float*>(bias.data()), {(size_t)block_size, 1}, "float"
  };

  std::vector<Tensor> const_tensors = {
    wts_tensor, bias_tensor, scl_tensor, zps_tensor
  };

  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["group_size"] = block_size;
  attrs["max_m"] = maxSeqLength();
  if (skip_input) {
    attrs["skip_create_input"] = 1;
  }
  if (skip_output) {
    attrs["skip_create_output"] = 1;
  }
  updateAttrsForSharedWeights(attrs);
  ptr->initialize_const_params(const_tensors, attrs);
}
template <typename T>
void SSMLPBase<T>::get_fused_size(size_t kernel_size) {
  auto gp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->gate_proj_.get();
  // Down projection
  auto dp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->down_proj_.get();

  auto elwmul =
    (ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>*)ss_->ewmul_.get();
  std::vector<size_t> a_shape_g = {1, kernel_size, static_cast<size_t>(gp_k)};
  std::vector<size_t> b_shape_g = {
    static_cast<size_t>(gp_k), static_cast<size_t>(gp_n * 2)
  };
  std::vector<size_t> c_shape_g = {
    1, kernel_size, static_cast<size_t>(gp_n * 2)
  };

  std::vector<size_t> a_shape_d = {1, kernel_size, static_cast<size_t>(dp_k)};
  std::vector<size_t> b_shape_d = {
    static_cast<size_t>(dp_k), static_cast<size_t>(dp_n)
  };
  std::vector<size_t> c_shape_d = {1, kernel_size, static_cast<size_t>(dp_n)};
  NPUTensor input_tensor_g = {nullptr, a_shape_g, "bfloat16"};
  NPUTensor wts_tensor_g = {nullptr, b_shape_g, "bfloat16"};
  NPUTensor out_tensor_g = {nullptr, c_shape_g, "bfloat16"};

  NPUTensor input_tensor_d = {nullptr, a_shape_d, "bfloat16"};
  NPUTensor wts_tensor_d = {nullptr, b_shape_d, "bfloat16"};
  NPUTensor out_tensor_d = {nullptr, c_shape_d, "bfloat16"};
  NPUTensor placeholder;

  std::vector<NPUTensor> inputs_g = {input_tensor_g, wts_tensor_g,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_g};

  std::vector<NPUTensor> inputs_d = {input_tensor_d, wts_tensor_d,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_d};

  std::vector<NPUTensor> inputs_add = {input_tensor_g, input_tensor_g};
  std::vector<NPUTensor> outputs_add = {input_tensor_g};

  std::vector<NPUTensor> inputs_mul = {input_tensor_d, input_tensor_d};
  std::vector<NPUTensor> outputs_mul = {input_tensor_d};

  std::vector<NPUTensor> outputs;
  std::map<std::string, std::any> attrs;
  std::vector<int> grp_size = {static_cast<int>(gp_block_size)};
  attrs["group_size"] = grp_size;
  attrs["mem_opt"] = 1;

  std::vector<OpArgMap> arg_map_g =
    gp_->get_buffer_reqs(inputs_g, outputs, attrs);
  grp_size = {static_cast<int>(up_block_size)};
  attrs["group_size"] = grp_size;

  std::vector<OpArgMap> arg_map_d =
    dp_->get_buffer_reqs(inputs_d, outputs, attrs);
  auto size_map = get_NPU_tensor_size(arg_map_d, mladfVersion());
  size_t d_bo_in_size = size_map["in0"];
  size_t d_bo_size = size_map["out"];
  size_t scratch0_bo_size = 0;
  buffer_size_ = 0;

  if (mladfVersion() == "v2") {
    scratch0_bo_size = size_map["scratch"];
  }

  size_map = get_NPU_tensor_size(arg_map_g, mladfVersion());
  size_t g_bo_size = size_map["out"];
  if (mladfVersion() == "v2") {
    if (scratch0_bo_size < size_map["scratch"])
      scratch0_bo_size = size_map["scratch"];
  }

  auto size_map_add = get_NPU_tensor_size(
    ss_->add_->get_buffer_reqs(inputs_add, outputs_add), mladfVersion()
  );
  auto size_map_mul = get_NPU_tensor_size(
    elwmul->get_buffer_reqs(inputs_mul, outputs_mul), mladfVersion()
  );

  SharedBuffer::Requirements shared_buffer_reqs{
    {"add", alignTo4096(size_map_add["in0"])},
    {"add1", alignTo4096(size_map_add["in0"])},
    {"gp", alignTo4096(g_bo_size)},
    {"elwmul_out", alignTo4096(std::max(d_bo_in_size, size_map_mul["out"]))},
    {"dp", alignTo4096(d_bo_size)},
  };

  if (mladfVersion() == "v2") {
    shared_buffer_reqs.emplace_back("scratch", alignTo4096(scratch0_bo_size));
  }

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}
template <typename T>
void SSMLPBase<T>::UpdateSharedBuffer(size_t kernel_size) {
  if (gate_up_fused_) {
    get_fused_size(kernel_size);
    return;
  }
  // clang-format off
  // memory layout
  // |[--down output--][--down in/elwmul out--][--up output--][--gate output--][--add in0--][--add in1--]
  // clang-format on
  // TODO pad rmsnorm input
  // Get operator pointers
  auto gp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->gate_proj_.get();
  auto up_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->up_proj_.get();
  auto dp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->down_proj_.get();
  auto elwmul =
    (ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>*)ss_->ewmul_.get();

  // Define tensor shapes
  std::vector<size_t> a_shape_g = {1, kernel_size, static_cast<size_t>(gp_k)};
  std::vector<size_t> b_shape_g = {
    static_cast<size_t>(gp_k), static_cast<size_t>(gp_n)
  };
  std::vector<size_t> c_shape_g = {1, kernel_size, static_cast<size_t>(gp_n)};
  std::vector<size_t> b_shape_u = {
    static_cast<size_t>(up_k), static_cast<size_t>(up_n)
  };
  std::vector<size_t> c_shape_u = {1, kernel_size, static_cast<size_t>(up_n)};
  std::vector<size_t> a_shape_d = {1, kernel_size, static_cast<size_t>(dp_k)};
  std::vector<size_t> b_shape_d = {
    static_cast<size_t>(dp_k), static_cast<size_t>(dp_n)
  };
  std::vector<size_t> c_shape_d = {1, kernel_size, static_cast<size_t>(dp_n)};

  NPUTensor input_tensor_g = {nullptr, a_shape_g, "bfloat16"};
  NPUTensor wts_tensor_g = {nullptr, b_shape_g, "bfloat16"};
  NPUTensor out_tensor_g = {nullptr, c_shape_g, "bfloat16"};
  NPUTensor wts_tensor_u = {nullptr, b_shape_u, "bfloat16"};
  NPUTensor out_tensor_u = {nullptr, c_shape_u, "bfloat16"};
  NPUTensor input_tensor_d = {nullptr, a_shape_d, "bfloat16"};
  NPUTensor wts_tensor_d = {nullptr, b_shape_d, "bfloat16"};
  NPUTensor out_tensor_d = {nullptr, c_shape_d, "bfloat16"};
  NPUTensor placeholder;

  std::vector<NPUTensor> inputs_g = {input_tensor_g, wts_tensor_g,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_g};
  std::vector<NPUTensor> inputs_u = {input_tensor_g, wts_tensor_u,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_u};
  std::vector<NPUTensor> inputs_d = {input_tensor_d, wts_tensor_d,
                                     placeholder,    placeholder,
                                     placeholder,    out_tensor_d};

  // to keep tensor index same with fusion for get_buffer_reqs()
  if (Lora::isEnabled()) {
    std::vector<size_t> lora_shape_gp = {
      (gp_k + gp_n) * LoraBuffer::getMaxRank()
    };
    std::vector<size_t> lora_shape_up = {
      (up_k + up_n) * LoraBuffer::getMaxRank()
    };
    std::vector<size_t> lora_shape_dp = {
      (dp_k + dp_n) * LoraBuffer::getMaxRank()
    };
    NPUTensor lora_tensor_gp = {nullptr, lora_shape_gp, "bfloat16"};
    NPUTensor lora_tensor_up = {nullptr, lora_shape_up, "bfloat16"};
    NPUTensor lora_tensor_dp = {nullptr, lora_shape_dp, "bfloat16"};
    inputs_g.insert(inputs_g.begin() + 1, lora_tensor_gp);
    inputs_u.insert(inputs_u.begin() + 1, lora_tensor_up);
    inputs_d.insert(inputs_d.begin() + 1, lora_tensor_dp);
  }

  std::vector<NPUTensor> inputs_add = {input_tensor_g, input_tensor_g};
  std::vector<NPUTensor> outputs_add = {input_tensor_g};
  std::vector<NPUTensor> inputs_mul = {input_tensor_d, input_tensor_d};
  std::vector<NPUTensor> outputs_mul = {input_tensor_d};

  std::vector<NPUTensor> outputs;
  std::map<std::string, std::any> attrs;

  // Get buffer requirements for each operator
  std::vector<int> grp_size = {static_cast<int>(gp_block_size)};
  attrs["group_size"] = grp_size;
  attrs["mem_opt"] = 1;

  std::vector<OpArgMap> arg_map_g =
    gp_->get_buffer_reqs(inputs_g, outputs, attrs);

  grp_size = {static_cast<int>(up_block_size)};
  attrs["group_size"] = grp_size;
  std::vector<OpArgMap> arg_map_u =
    up_->get_buffer_reqs(inputs_u, outputs, attrs);

  grp_size = {static_cast<int>(dp_block_size)};
  attrs["group_size"] = grp_size;
  std::vector<OpArgMap> arg_map_d =
    dp_->get_buffer_reqs(inputs_d, outputs, attrs);

  // Calculate buffer sizes
  auto size_map = get_NPU_tensor_size(arg_map_d, mladfVersion());
  size_t d_bo_in_size = size_map["in0"];
  size_t d_bo_size = size_map["out"];
  size_t scratch0_bo_size = 0;

  if (mladfVersion() == "v2") {
    scratch0_bo_size = size_map["scratch"];
  }
  size_map = get_NPU_tensor_size(arg_map_u, mladfVersion());
  size_t u_bo_size = size_map["out"];
  if (mladfVersion() == "v2" && scratch0_bo_size < size_map["scratch"]) {
    scratch0_bo_size = size_map["scratch"];
  }
  size_map = get_NPU_tensor_size(arg_map_g, mladfVersion());
  size_t g_bo_size = size_map["out"];
  if (mladfVersion() == "v2" && scratch0_bo_size < size_map["scratch"]) {
    scratch0_bo_size = size_map["scratch"];
  }

  auto size_map_add = get_NPU_tensor_size(
    ss_->add_->get_buffer_reqs(inputs_add, outputs_add), mladfVersion()
  );
  auto size_map_mul = get_NPU_tensor_size(
    elwmul->get_buffer_reqs(inputs_mul, outputs_mul), mladfVersion()
  );

  SharedBuffer::Requirements shared_buffer_reqs{
    {"add", alignTo4096(size_map_add["in0"])},
    {"add1", alignTo4096(size_map_add["in0"])},
    {"gp", alignTo4096(std::max(g_bo_size, size_map_mul["out"]))},
    {"up", alignTo4096(std::max(g_bo_size, size_map_mul["out"]))},
    {"elwmul_out", alignTo4096(std::max(d_bo_in_size, size_map_mul["out"]))},
    {"dp", alignTo4096(std::max(d_bo_size, alignTo4096(size_map_add["in0"])))},
  };

  if (mladfVersion() == "v2") {
    shared_buffer_reqs.emplace_back("scratch", alignTo4096(scratch0_bo_size));
  }

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}

template <typename T>
void SSMLPBase<T>::initializeLora() {
  if (Lora::isEnabled()) {
    lora_buffers_.addBo(gp_k, gp_n, ss_->gate_proj_.get(), allocator_);
    lora_buffers_.addBo(up_k, up_n, ss_->up_proj_.get(), allocator_);
    lora_buffers_.addBo(dp_k, dp_n, ss_->down_proj_.get(), allocator_);
    lora_buffers_.syncLora();
  }
}

template <typename T>
void SSMLPBase<T>::loadOneLoraData(int idx) {
  ExternalTensorInfo jit_tensor_info =
    getExternalTensorInfo(Lora::prefillHeader(), name(), idx);
  if (jit_tensor_info.size > 0) {
    Lora::loadBinData(
      lora_buffers_.data(idx), jit_tensor_info.size, jit_tensor_info.offset
    );
  } else {
    // Need to clean up the previous lora buffers even if no new lora data is
    // loaded
    lora_buffers_.zeroesLoraData(idx);
  }
}

template <typename T>
void SSMLPBase<T>::loadLoraData() {
  loadOneLoraData(0);  // gate proj
  loadOneLoraData(1);  // up proj
  loadOneLoraData(2);  // down proj

  lora_buffers_.syncLora();
}

template <typename T>
void SSMLPBase<T>::LoadLora() {
  const auto& lora_name = Lora::getLoraName();
  if (lora_name != lora_buffers_.getLoraName()) {
    lora_buffers_.setLora(lora_name);
    if (lora_name != "base") {
      loadLoraData();
    } else {
      lora_buffers_.zeroesAllLoraData();
      lora_buffers_.syncLora();
    }
  }
}

template <typename T>
void SSMLPBase<T>::readDataImpl(int idx) {
  if (shared_weights_.ready()) return;
  updateJitBuffer(
    dynamicJitFactor(), ss_->bo_data_gate_[idx], jit_tensor_gate_.size,
    ss_->jit_max_bo_size_gate_
  );
  if (!gate_up_fused_) {
    updateJitBuffer(
      dynamicJitFactor(), ss_->bo_data_up_[idx], jit_tensor_up_.size,
      ss_->jit_max_bo_size_up_
    );
  }
  updateJitBuffer(
    dynamicJitFactor(), ss_->bo_data_down_[idx], jit_tensor_down_.size,
    ss_->jit_max_bo_size_down_
  );
  const auto external_data = externalData().string();
  loadBin(
    ss_->bo_data_gate_[idx].Data(), external_data, jit_tensor_gate_.size,
    jit_tensor_gate_.offset
  );
  if (!gate_up_fused_) {
    loadBin(
      ss_->bo_data_up_[idx].Data(), external_data, jit_tensor_up_.size,
      jit_tensor_up_.offset
    );
  }
  loadBin(
    ss_->bo_data_down_[idx].Data(), external_data, jit_tensor_down_.size,
    jit_tensor_down_.offset
  );
}

template <typename T>
void SSMLPBase<T>::loadDataImpl(int idx) {
  // if we're reading everything, we don't need to use a shared buffer
  const bool use_shared_buffer = this->isJitEnabled();

  auto gate_ptr =
    (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
      ss_->gate_proj_.get();
  initialize(
    (const uint8_t*)ss_->bo_data_gate_[idx].Data(), jit_tensor_gate_.size,
    gp_block_size, gate_ptr, gp_k, gp_n, ss_->bo_data_gate_[idx].Size(), true,
    use_shared_buffer
  );
  if (!gate_up_fused_) {
    auto up_ptr =
      (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
        ss_->up_proj_.get();

    initialize(
      (const uint8_t*)ss_->bo_data_up_[idx].Data(), jit_tensor_up_.size,
      up_block_size, up_ptr, up_k, up_n, ss_->bo_data_up_[idx].Size(), true,
      use_shared_buffer
    );
  }

  auto down_ptr =
    (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
      ss_->down_proj_.get();
  initialize(
    (const uint8_t*)ss_->bo_data_down_[idx].Data(), jit_tensor_down_.size,
    dp_block_size, down_ptr, dp_k, dp_n, ss_->bo_data_down_[idx].Size(), true,
    use_shared_buffer
  );
}

static void rmsnorm_cpu(
  std::uint16_t* in_ptr, std::uint16_t* const_ptr, std::uint16_t* out_ptr,
  int M, int K, float epsilon
) {
  for (int i = 0; i < M; i++) {
    // step 1 calculate root mean square of row

    float avg_sqaure = 0.0;
    const float scale = 1.0f / K;
    for (int j = 0; j < K; j++) {
      std::uint32_t val_int = (in_ptr[i * K + j] << 16);
      float val = *reinterpret_cast<float*>(&val_int);

      avg_sqaure += val * (val * scale);
    }

    // this is from model, to make sure non-zero
    avg_sqaure += epsilon;

    // step 2 calculate norm_scale = 1/sqrt(x)

    __m128 x = _mm_set_ss(avg_sqaure);
    x = _mm_rsqrt_ss(x);

    float avg_square_recip_root = 0;

    _mm_store_ss(&avg_square_recip_root, x);

    // step 3 apply scale of gamma_i * norm_scale
    //        and write to output

    for (int j = 0; j < K; j++) {
      std::uint32_t val_int = (in_ptr[i * K + j] << 16);
      float val = *reinterpret_cast<float*>(&val_int);

      val_int = const_ptr[j] << 16;
      float gamma = *reinterpret_cast<float*>(&val_int);

      val = val * gamma * avg_square_recip_root;

      val_int = *reinterpret_cast<std::uint32_t*>(&val);

      out_ptr[i * K + j] = (val_int >> 16) & 0xFFFF;
    }
  }
}

template <bool accurate>
static void silu_cpu(
  std::uint16_t* in_ptr, std::uint16_t* out_ptr, int M, int K
) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      std::uint32_t val_int = in_ptr[i * K + j] << 16;
      float val = *reinterpret_cast<float*>(&val_int);

      // SiLU(x) = x * Sigmoid(x)
      // Sigmoid(x) = 1/(1 + e^-x) = e^x / (1 + e^x)
      //                           = 1 - 1 / (1 + e^x)
      // for numerical stability use following
      //  x >= 0  : 1/(1 + e^-x)
      //  x < 0   : 1 - 1 / (1 + e^x)

      if constexpr (accurate) {
        bool is_neg = val < 0;
        float val_abs = is_neg ? -val : val;
        __m128 x = _mm_set_ss(1.0f + std::expf(-val_abs));
        x = _mm_rcp_ss(x);

        float recip = 0;

        _mm_store_ss(&recip, x);

        float sigmoid_val = is_neg ? (1 - recip) : (recip);

        val = val * sigmoid_val;
      } else {
        val = val * (1.0f / (1.0f + std::expf(-val)));
      }

      val_int = *reinterpret_cast<std::uint32_t*>(&val);

      out_ptr[i * K + j] = (val_int >> 16) & 0xFFFF;
    }
  }
}

static void gelu_cpu(
  const std::uint16_t* in_ptr, float* out_ptr, int M, int N
) {
  for (size_t i = 0; i < M * N; ++i) {
    constexpr double B = 0.7978845608028654;  // sqrt(2.0 / M_PI)
    constexpr double C = 0.044715;
    auto x = bfloat16_to_float_single(in_ptr[i]);
    auto y = 0.5 * (1 + std::tanh(B * x * (1 + C * x * x))) * x;
    out_ptr[i] = static_cast<float>(y);
  }
}
template <typename T>
void SSMLPBase<T>::activation_execute(
  size_t gp_M, std::vector<xrt::bo> gate_outputs,
  std::vector<xrt::bo> ewmul_outputs, bool wait
) {
  // Silu
  auto silu = (ryzenai::dynamic_dispatch::transformer::general_activation<
               uint16_t, uint16_t>*)ss_->silu_.get();
  std::vector<size_t> a_shape_silu = {
    static_cast<size_t>(gp_M), static_cast<size_t>(gp_n * 2)
  };
  std::vector<size_t> b_shape_silu = {
    static_cast<size_t>(gp_M), static_cast<size_t>(gp_n)
  };
  std::map<std::string, std::any> attr = {};

  NPUTensor input_tensor = {nullptr, a_shape_silu, "bfloat16"};
  NPUTensor output_tensor = {nullptr, b_shape_silu, "bfloat16"};

  std::vector<NPUTensor> inputs_tmp = {input_tensor};
  std::vector<NPUTensor> outputs_tmp = {output_tensor};

  // if constexpr (!silu_cpu_en) {
  tryContinueOnException([&]() {
    RecordDuration(Metric::Tiling, [&]() {
      silu->set_tensor_shape(inputs_tmp, outputs_tmp, attr);
    });
  });
  auto const_bos = silu->get_const();
  std::vector<NPUBufferSpan> inputs = {
    {gate_outputs[0], 0, gate_outputs[0].size()},
    {const_bos[0], 0, const_bos[0].size()}
  };
  std::vector<NPUBufferSpan> outputs = {
    {ewmul_outputs[0], 0, ewmul_outputs[0].size()}
  };
  tryContinueOnException([&]() {
    RecordDuration(Metric::KernelExecution, [&]() {
      auto run = silu->create_run(inputs, outputs);

      run.value().start();

      // if (wait)
      run.value().wait2();
    });
  });
}
template <typename T>
void SSMLPBase<T>::Compute(OrtKernelContext* context) {
  // to dump to file
  // MY_LOG(2) << "\n\n- AMD SSLRN compute start ...\n";
  PROFILING_START(Compute)
  PROFILING_START(setup)
#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif

  if (gate_up_fused_) {
    shared_weights_.setWeightAddr(0);
    shared_weights_.setWeightAddr(1);
  } else {
    shared_weights_.setWeightAddr(0);
    shared_weights_.setWeightAddr(1);
    shared_weights_.setWeightAddr(2);
  }

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

  // passing true to execute causes it to wait
  const bool execute_sync = !continueOnException();
#ifdef _WIN32
  const bool execute_async = false;
#else
  // on Linux, cannot execute in async mode: xrt::run objects go out of scope
  // in DD eager execute. Windows makes a copy unlike Linux
  const bool execute_async = true;
#endif  // _WIN32
  PROFILING_END(setup, false, name().c_str())

  // MY_LOG(2) << "num_inputs " << num_inputs << " "
  //           << "num_outputs " << num_outputs << " ";

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

  std::vector<xrt::bo> rms_norm3_outputs;
  if (is_ssgmlp_) {
    if (ss_->seq_len_ != M) {
      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          ss_->rms_norm3_->set_kernel_shape(a_shape);
        });
      });
    }

    // ssgmlp - gemma ssmlp first additional simplified layer norm
    //  SLN
    static bool sln_cpu = false;  // true;
    rms_norm3_outputs = {ss_->rms_norm3_->get_outputs()[0]};
    if (!sln_cpu) {
      const auto rms_norm_operand_size_in_bytes =
        a_shape[0] * a_shape[1] * sizeof(uint16_t);
      auto rms_norm3_inputs = ss_->rms_norm3_->get_inputs();
      uint16_t* rms_norm3_input_0_map =
        rms_norm3_inputs[0].template map<uint16_t*>();
      MemCpy(
        (void*)rms_norm3_input_0_map, (uint16_t*)skip_ptr,
        rms_norm_operand_size_in_bytes
      );

      const auto rmsnorm3_in_size = rms_norm_operand_size_in_bytes;
      const auto rmsnorm_3_in_offset = 0;

      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm3_inputs[0].sync(
          XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm3_in_size, rmsnorm_3_in_offset
        );
      });

      auto rms_norm3_wt_inputs = ss_->rms_norm3_->get_inputs()[1];
      uint16_t* rms_wt3_map = rms_norm3_wt_inputs.template map<uint16_t*>();
      MemCpy((void*)rms_wt3_map, wts3_.data(), num_el_bo3);

      const auto rmsnorm3_wts_size = num_el_bo3;
      const auto rmsnorm3_wts_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm3_wt_inputs.sync(
          XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm3_wts_size, rmsnorm3_wts_offset
        );
      });
      std::vector<xrt::bo> rms_in3 = {rms_norm3_inputs[0], rms_norm3_wt_inputs};

      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->rms_norm3_->execute(rms_in3, rms_norm3_outputs, execute_sync);
        });
      });
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm3_outputs[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, rmsnorm3_in_size, rmsnorm_3_in_offset
        );
      });
      skip_ptr = rms_norm3_outputs[0].map<uint16_t*>();

    } else {
      uint16_t* out_ptr = rms_norm3_outputs[0].map<uint16_t*>();
      ort_slrn_.execute(
        (Ort::BFloat16_t*)skip_ptr, (Ort::BFloat16_t*)skip_ptr,
        dimensions_input, wts3_data_.data(), {static_cast<int64_t>(K)},
        allocator_, context
      );
      // rms_norm3_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
  }

  // sslrn1 aie kernel bos
  std::vector<xrt::bo> rms_norm1_outputs_;
  std::vector<xrt::bo> add1_outputs_;
  const bool supported_shapes =
    (M > 1) ||
    (std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
     supported_lengths.end());

  RyzenMM::BufferRef input_a, input_b, output_1, output_2;

  // for cpu
  sslrn_cpu_out = false;
  RyzenMM::BufferRef sslrn_out_data_token1;

  if (supported_shapes) {
    std::vector<xrt::bo> add_inputs;
    add_inputs = ss_->add_->get_inputs();

    std::vector<xrt::bo> add0_in;
    if (is_ssgmlp_) {
      add0_in = {add_inputs[0], rms_norm3_outputs[0]};
    }

    // do add in-place
    add1_outputs_ = {add_inputs[0]};

    auto rms_norm_wt_inputs = ss_->rms_norm_->get_inputs()[1];
    rms_norm1_outputs_ = {add_inputs[1]};
    if (ss_->seq_len_ != M) {
      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          ss_->add_->set_kernel_shape(a_shape);
        });
      });
      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          ss_->rms_norm_->set_kernel_shape(a_shape);
        });
      });
      ss_->seq_len_ = M;
    }
    const auto add_operand_size_in_bytes =
      a_shape[0] * a_shape[1] * sizeof(uint16_t);
    {
      // Input BO
      if (need_op0_copy) {
        uint16_t* add_input_0_map = add_inputs[0].map<uint16_t*>();
        MemCpy(
          (void*)add_input_0_map, (uint16_t*)input_ptr,
          add_operand_size_in_bytes
        );

        const auto add0_in_size = add_operand_size_in_bytes;
        const auto add0_in_offset = 0;

        RecordDuration(Metric::XRTBOSync, [&]() {
          add_inputs[0].sync(
            XCL_BO_SYNC_BO_TO_DEVICE, add0_in_size, add0_in_offset
          );
        });
      }

      uint16_t* add_input_1_map = add_inputs[1].map<uint16_t*>();
      MemCpy(
        (void*)add_input_1_map, (uint16_t*)skip_ptr, add_operand_size_in_bytes
      );

      const auto add1_in_size = add_operand_size_in_bytes;
      const auto add1_in_offset = 0;

      RecordDuration(Metric::XRTBOSync, [&]() {
        add_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      });
    }

    constexpr bool rmsnorm_cpu_en = false;

    PROFILING_START(execute_add)
    if (is_ssgmlp_) {
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->add_->execute(
            add0_in, add1_outputs_, rmsnorm_cpu_en || execute_async
          );
        });
      });
    } else {
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->add_->execute(
            add_inputs, add1_outputs_, rmsnorm_cpu_en || execute_async
          );
        });
      });
    }
    PROFILING_END(execute_add, false, name().c_str())

    uint16_t* rms_wt_map = rms_norm_wt_inputs.template map<uint16_t*>();
    MemCpy((void*)rms_wt_map, wts_.data(), num_el_bo);

    const auto rmsnorm_wts_size = num_el_bo;
    const auto rmsnorm_wts_offset = 0;

    RecordDuration(Metric::XRTBOSync, [&]() {
      rms_norm_wt_inputs.sync(
        XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm_wts_size, rmsnorm_wts_offset
      );
    });

    if constexpr (!rmsnorm_cpu_en) {
      std::vector<xrt::bo> rms_in = {add1_outputs_[0], rms_norm_wt_inputs};

      // Execute RMS Norm
      PROFILING_START(execute_rmsnorm)
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->rms_norm_->execute(rms_in, rms_norm1_outputs_, execute_async);
        });
      });
      PROFILING_END(execute_rmsnorm, false, name().c_str())
    } else {
      // std::cout << "rmsnorm1 on cpu M = " << M << ", K = " << K << "\n";

      std::uint16_t* in_ptr = add1_outputs_[0].map<std::uint16_t*>();
      std::uint16_t* const_ptr = getRmsNormConstData(rms_wt_map);
      std::uint16_t* out_ptr = rms_norm1_outputs_[0].map<std::uint16_t*>();

      RecordDuration(Metric::XRTBOSync, [&]() {
        add1_outputs_[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      });

      rmsnorm_cpu(in_ptr, const_ptr, out_ptr, M, K, epsilon_);

      // make sure to sync back to NPU
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm1_outputs_[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
      });
    }

    // MY_LOG(2) << "- SSLRN 1 AIE done ...\n";

  } else {
    // MY_LOG(2) << "- AMD SSLRN CPU ...\n";
    sslrn_cpu_out =
      true;  // set flag to notify MLP and next SSLRN to memcopy and sync
             // outputs of SSLRN 1 with input of MLP and SSLRN2
    sslrn_out_data_token1 = allocator_.AllocateBuffer(M * K * sizeof(uint16_t));

    // Define input shape
    std::vector<int64_t> input_shape = {(int64_t)B, (int64_t)M, (int64_t)K};

    RecordDuration(Metric::Casting, [&]() {
      if (!input_a)
        input_a = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!input_b)
        input_b = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!output_1)
        output_1 = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!output_2)
        output_2 = allocator_.AllocateBuffer(num_elements * sizeof(float));

      bfloat16_buffer_to_float(
        (uint16_t*)input_ptr, num_elements, input_a.Data<float>()

      );  // M x K
      bfloat16_buffer_to_float(
        (uint16_t*)skip_ptr, num_elements, input_b.Data<float>()
      );
    });

    auto dimensions_wts = m_weights.GetTensorTypeAndShapeInfo().GetShape();

    ort_sslrn_.execute(
      output_1.Data<float>(), output_2.Data<float>(), input_a.Data<float>(),
      input_b.Data<float>(), input_shape, wts_data_.data(), dimensions_wts,
      context
    );

    RecordDuration(Metric::Casting, [&]() {
      ryzenai::float_buffer_to_bfloat16(
        output_1.Data<float>(), M * K, sslrn_out_data_token1.Data<uint16_t>()

      );  // M x K
    });
  }

#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point add_rms_0_end = Clock::now();
#endif
  // MY_LOG(2) << "- AMD SSLRN1 compute done ...\n";
  ///////////////////////////////////////////////////////////
  // MY_LOG(2) << "- MLP compute start ...";
  auto input_shape = dimensions_input;

  // Gate projection
  auto gp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->gate_proj_.get();
  // Up projection
  auto up_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->up_proj_.get();
  // Down projection
  auto dp_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               ss_->down_proj_.get();

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

  // ElwMul
  auto ewmul =
    (ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>*)ss_->ewmul_.get();
  size_t gate_n_mul = gate_up_fused_ ? 2 : 1;

  // Ryzen-AI implementation
  int gp_M = input_shape[0] * input_shape[1];
  std::vector<size_t> gp_a_shape = {
    static_cast<size_t>(gp_M), static_cast<size_t>(input_shape[2])
  };

  std::vector<size_t> gp_c_shape = {
    static_cast<size_t>(gp_M), static_cast<size_t>(gp_n * gate_n_mul)
  };

  std::vector<size_t> gp_wts_shape = {
    static_cast<size_t>(gp_k), static_cast<size_t>(gp_n * gate_n_mul)
  };

  std::vector<size_t> up_wts_shape = {
    static_cast<size_t>(up_k), static_cast<size_t>(up_n)
  };

  std::vector<size_t> dp_wts_shape = {
    static_cast<size_t>(dp_k), static_cast<size_t>(dp_n)
  };

  std::vector<size_t> dp_a_shape = {
    static_cast<size_t>(gp_M), static_cast<size_t>(dp_k)
  };

  std::vector<size_t> dp_c_shape = {
    static_cast<size_t>(gp_M), static_cast<size_t>(dp_n)
  };

  // NOT: this assumes buffer layout is
  //[add_op0|add_op1|gate|up|down]

  PROFILING_START(create_bo)
  if (auto rebind =
        shared_buffer_.Validate("gp", gp_->get_outputs(kGemmBOsSelector)[0])) {
    gp_->create_bo(rebind->ptr, rebind->len, 1, kGemmBOsSelector);
  }

  if (!gate_up_fused_) {
    if (auto rebind = shared_buffer_.Validate(
          "up", up_->get_outputs(kGemmBOsSelector)[0]
        )) {
      up_->create_bo(rebind->ptr, rebind->len, 1, kGemmBOsSelector);
    }
  }

  if (auto rebind =
        shared_buffer_.Validate("dp", dp_->get_outputs(kGemmBOsSelector)[0])) {
    dp_->create_bo(rebind->ptr, rebind->len, 1, kGemmBOsSelector);
  }

  if (auto rebind =
        shared_buffer_.Validate("elwmul_out", ewmul->get_outputs()[0])) {
    ewmul->create_bo(rebind->ptr, rebind->len, 2);
  }

  if (mladfVersion() == "v2") {
    if (auto rebind = shared_buffer_.Validate(
          "scratch", ss_->gemm_last_scratch_ptr_, ss_->gemm_last_scratch_len_
        )) {
      gp_->create_bo(rebind->ptr, rebind->len, 2, kGemmBOsSelector);
      if (!gate_up_fused_)
        up_->create_bo(rebind->ptr, rebind->len, 2, kGemmBOsSelector);
      dp_->create_bo(rebind->ptr, rebind->len, 2, kGemmBOsSelector);

      ss_->gemm_last_scratch_ptr_ = rebind->ptr;
      ss_->gemm_last_scratch_len_ = rebind->len;
    }
  }
  PROFILING_END(create_bo, false, name().c_str())

  std::vector<xrt::bo> gate_inputs = {ss_->add_->get_inputs()[1]};
  auto gate_outputs = gp_->get_outputs(kGemmBOsSelector);
  auto dp_outputs = dp_->get_outputs(kGemmBOsSelector);
  gate_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  dp_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  gate_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<xrt::bo> ewmul_outputs = ewmul->get_outputs();
  gp_->set_shape(gp_a_shape, gp_wts_shape, gp_block_size);
  dp_->set_shape(dp_a_shape, dp_wts_shape, dp_block_size);

  std::vector<xrt::bo> gate_const;
  std::vector<xrt::bo> up_const;
  std::vector<xrt::bo> down_const;

  if (gp_M > 1) {
    // MY_LOG(2) << "gp_M>1...  PROMPT PHASE, cnt=" << cnt_ << std::endl;
    std::vector<xrt::bo> gate_in;
    std::vector<xrt::bo> up_in;

    if (sslrn_cpu_out) {  // M>1 but unsupported shapes for sslrn
      // Input BO
      uint16_t* in_map = gate_inputs.at(0).map<uint16_t*>();
      MemCpy(
        (void*)in_map, (void*)sslrn_out_data_token1.Data(),
        gp_M * input_shape[2] * sizeof(uint16_t)
      );

      const auto gate_in_size = gp_M * input_shape[2] * sizeof(std::uint16_t);
      const auto gate_in_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        gate_inputs.at(0).sync(
          XCL_BO_SYNC_BO_TO_DEVICE, gate_in_size, gate_in_offset
        );
      });
      gate_in = {gate_inputs.at(0)};
      up_in = {gate_inputs.at(0)};
    } else {
      gate_in = {rms_norm1_outputs_[0]};
      up_in = {rms_norm1_outputs_[0]};
    }

    bool silu_cpu_en = false;
    bool wait = silu_cpu_en || execute_async;

    PROFILING_START(execute_matmul_gp)
    if (shared_weights_.ready()) {
      std::vector<xrt::bo> inputs = {gate_in[0]};
      if (Lora::isEnabled()) {
        inputs.push_back(xrt::bo());  // dummy BO to fix input schema
        inputs.push_back(lora_buffers_.getBo(0));
      }
      std::vector<uint64_t> input_addrs = {0, shared_weights_.weightAddr(0), 0};
      std::vector<uint64_t> out_addrs;
      RecordDuration(Metric::KernelExecution, [&]() {
        gp_->execute(inputs, input_addrs, gate_outputs, out_addrs, wait);
      });
    } else {
      gate_const = gp_->get_const();
      gate_in.push_back(gate_const[cnt]);
      if (Lora::isEnabled()) gate_in.push_back(lora_buffers_.getBo(0));
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          gp_->execute(gate_in, gate_outputs, wait);
        });
      });
    }
    PROFILING_END(execute_matmul_gp, false, name().c_str())

    // MY_LOG(2) << "exec gp...";
    //  Up proj
    PROFILING_START(execute_matmul_up)
    if (!gate_up_fused_) {
      up_->set_shape(gp_a_shape, up_wts_shape, up_block_size);
      auto up_outputs = up_->get_outputs(gp_M);
      up_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

      if (shared_weights_.ready()) {
        std::vector<xrt::bo> inputs = {up_in[0]};
        if (Lora::isEnabled()) {
          inputs.push_back(xrt::bo());  // dummy BO to fix input schema
          inputs.push_back(lora_buffers_.getBo(1));
        }
        std::vector<uint64_t> input_addrs = {
          0, shared_weights_.weightAddr(1), 0
        };
        std::vector<uint64_t> out_addrs;
        RecordDuration(Metric::KernelExecution, [&]() {
          up_->execute(inputs, input_addrs, up_outputs, out_addrs, wait);
        });
      } else {
        up_const = up_->get_const();
        up_in.push_back(up_const[cnt]);
        if (Lora::isEnabled()) up_in.push_back(lora_buffers_.getBo(1));
        tryContinueOnException([&]() {
          RecordDuration(Metric::KernelExecution, [&]() {
            up_->execute(up_in, up_outputs, wait);
          });
        });
      }
      PROFILING_END(execute_matmul_up, false, name().c_str())

      // Silu
      // MY_LOG(2) << "exec up...";
      // TODO: Fixed M size for Silu for prefill phase
      std::vector<size_t> a_shape_silu = {
        static_cast<size_t>(gp_M), static_cast<size_t>(gp_n)
      };
      auto silu = (ryzenai::silu<uint16_t, uint16_t>*)ss_->silu_.get();

      if (!silu_cpu_en) {
        // if ssgmlp, the silu executes a "GELU" since the "gelu" attr is set
        tryContinueOnException([&]() {
          RecordDuration(Metric::Tiling, [&]() {
            silu->set_kernel_shape(a_shape_silu);
          });
        });
        PROFILING_START(execute_silu)
        tryContinueOnException([&]() {
          RecordDuration(Metric::KernelExecution, [&]() {
            silu->execute(gate_outputs, gate_outputs, execute_async);
          });
        });
        PROFILING_END(execute_silu, false, name().c_str())
      } else {
        // std::cout << "running silu on cpu M = " << gp_M << ", K = " << gp_n
        // <<
        // "\n";
        RecordDuration(Metric::XRTBOSync, [&]() {
          gate_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        });
        std::uint16_t* in_ptr = gate_outputs[0].map<std::uint16_t*>();
        std::uint16_t* out_ptr = gate_outputs[0].map<std::uint16_t*>();

        if (is_ssgmlp_) {
          auto output = allocator_.AllocateBuffer(gp_M * gp_n * sizeof(float));
          gelu_cpu(in_ptr, output.Data<float>(), gp_M, gp_n);
          RecordDuration(Metric::Casting, [&]() {
            ryzenai::float_buffer_to_bfloat16(
              output.Data<float>(), gp_M * gp_n, out_ptr
            );
          });
        } else {
          constexpr bool accurate = true;
          silu_cpu<accurate>(in_ptr, out_ptr, gp_M, gp_n);
        }

        RecordDuration(Metric::XRTBOSync, [&]() {
          gate_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        });
      }

      // MY_LOG(2) << "execute silu...";
      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          ewmul->set_kernel_shape(a_shape_silu);
        });
      });

      std::vector<xrt::bo> ewmul_inputs = {gate_outputs[0], up_outputs[0]};

      PROFILING_START(execute_ewmul)
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ewmul->execute(ewmul_inputs, ewmul_outputs, execute_async);
        });
      });
      PROFILING_END(execute_ewmul, false, name().c_str())
    } else {
      // Silu + mul
      activation_execute(gp_M, gate_outputs, ewmul_outputs, wait);
    }
    // MY_LOG(2) << "exec mul...";
    std::vector<xrt::bo> dp_inputs = {ewmul_outputs[0]};

    // will go next to CPU sslrn, so must wait
    wait = !supported_shapes && !continueOnException();
    // MY_LOG(2) << "exec gp...";
    //  Up proj
    PROFILING_START(execute_matmul_dp)
    int dp_share_buf_index = gate_up_fused_ ? 1 : 2;
    if (shared_weights_.ready()) {
      std::vector<xrt::bo> inputs = {dp_inputs[0]};
      if (Lora::isEnabled()) {
        inputs.push_back(xrt::bo());  // dummy BO to fix input schema
        inputs.push_back(lora_buffers_.getBo(2));
      }
      std::vector<uint64_t> input_addrs = {
        0, shared_weights_.weightAddr(dp_share_buf_index), 0
      };
      std::vector<uint64_t> out_addrs;
      RecordDuration(Metric::KernelExecution, [&]() {
        dp_->execute(inputs, input_addrs, dp_outputs, out_addrs, wait);
      });
    } else {
      down_const = dp_->get_const();
      dp_inputs.push_back(down_const[cnt]);
      if (Lora::isEnabled()) dp_inputs.push_back(lora_buffers_.getBo(2));
      // Don't wait if M=supported shape; i.e. NPU sslrn2
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          dp_->execute(dp_inputs, dp_outputs, wait);
        });
      });
    }
    PROFILING_END(execute_matmul_dp, false, name().c_str())
  } else {
    // MY_LOG(2) << "gp_M==1..."; // token phase
    gate_const = gp_->get_const();
    up_const = up_->get_const();
    down_const = dp_->get_const();
    std::vector<xrt::bo> gate_in;
    std::vector<xrt::bo> up_in;
    auto up_outputs = up_->get_outputs(kGemmBOsSelector);
    if (sslrn_cpu_out) {
      // Input BO
      uint16_t* in_map = gate_inputs.at(0).map<uint16_t*>();
      MemCpy(
        (void*)in_map, (void*)sslrn_out_data_token1.Data(),
        gp_k * sizeof(uint16_t)
      );

      const auto gate_in_size = gp_k * sizeof(std::uint16_t);
      const auto gate_in_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        gate_inputs.at(0).sync(
          XCL_BO_SYNC_BO_TO_DEVICE, gate_in_size, gate_in_offset
        );
      });
      gate_in = {gate_inputs.at(0), gate_const[cnt]};
      up_in = {gate_inputs.at(0)};
    } else {
      gate_in = {rms_norm1_outputs_[0], gate_const[cnt]};
      up_in = {rms_norm1_outputs_[0]};
    }

    if (Lora::isEnabled()) gate_in.push_back(lora_buffers_.getBo(0));

    // Exec

    bool wait = execute_async;
    if (Lora::isEnabled()) {
      wait = true;
    }

    // Gate proj
    PROFILING_START(execute_matmul_gp)
    tryContinueOnException([&]() {
      RecordDuration(Metric::KernelExecution, [&]() {
        gp_->execute(gate_in, gate_outputs, wait);
      });
    });
    PROFILING_END(execute_matmul_gp, false, name().c_str())

    // Up proj
    if (!gate_up_fused_) {
      up_->set_shape(gp_a_shape, up_wts_shape, up_block_size);
      up_const = up_->get_const();
      up_in.push_back(up_const[cnt]);
      PROFILING_START(execute_matmul_up)
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          up_->execute(up_in, up_outputs, wait);
        });
      });
      PROFILING_END(execute_matmul_up, false, name().c_str())

      // Silu
      auto silu = (ryzenai::silu<uint16_t, uint16_t>*)ss_->silu_.get();

      std::vector<size_t> a_shape_silu = {
        static_cast<size_t>(gp_M), static_cast<size_t>(gp_n)
      };
      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          silu->set_kernel_shape(a_shape_silu);
        });
      });

      PROFILING_START(execute_silu)
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          silu->execute(gate_outputs, gate_outputs, execute_async);
        });
      });
      PROFILING_END(execute_silu, false, name().c_str())

      tryContinueOnException([&]() {
        RecordDuration(Metric::Tiling, [&]() {
          ewmul->set_kernel_shape(a_shape_silu);
        });
      });
      std::vector<xrt::bo> ewmul_inputs = {gate_outputs[0], up_outputs[0]};

      PROFILING_START(execute_ewmul)
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ewmul->execute(ewmul_inputs, ewmul_outputs, execute_async);
        });
      });
      PROFILING_END(execute_ewmul, false, name().c_str())
    } else {
      activation_execute(gp_M, gate_outputs, ewmul_outputs, wait);
    }
    std::vector<xrt::bo> dp_inputs = {ewmul_outputs[0], down_const[cnt]};
    if (Lora::isEnabled()) dp_inputs.push_back(lora_buffers_.getBo(2));

    wait = execute_sync;
    if (Lora::isEnabled()) {
      wait = true;
    }

    PROFILING_START(execute_matmul_dp)
    tryContinueOnException([&]() {
      RecordDuration(Metric::KernelExecution, [&]() {
        dp_->execute(dp_inputs, dp_outputs, wait);
      });
    });
    PROFILING_END(execute_matmul_dp, false, name().c_str())
  }

  if (useExternalData()) {
    this->loadData();
  }

#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point mlp_0_end = Clock::now();
#endif
  uint16_t* sln2_output_bo = nullptr;
  std::vector<xrt::bo> rms_norm4_outputs;
  if (is_ssgmlp_) {
    static bool sln2_cpu = false;

    if (!sln2_cpu) {
      // ssgmlp - gemma ssmlp second additional simplified layer norm
      RecordDuration(Metric::Tiling, [&]() {
        ss_->rms_norm4_->set_kernel_shape(a_shape);
      });
      rms_norm4_outputs = {ss_->rms_norm4_->get_outputs()[0]};
      auto rms_norm4_wt_inputs = ss_->rms_norm4_->get_inputs()[1];
      uint16_t* rms_wt4_map = rms_norm4_wt_inputs.template map<uint16_t*>();
      MemCpy((void*)rms_wt4_map, wts4_.data(), num_el_bo4);

      const auto rmsnorm4_wts_size = num_el_bo4;
      const auto rmsnorm4_wts_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm4_wt_inputs.sync(
          XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm4_wts_size, rmsnorm4_wts_offset
        );
      });
      std::vector<xrt::bo> rms_in4 = {dp_outputs[0], rms_norm4_wt_inputs};

      const auto rmsnorm4_out_size =
        dp_c_shape[0] * dp_c_shape[1] * sizeof(std::uint16_t);
      const auto rmsnorm4_out_offset = 0;

      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->rms_norm4_
            ->execute(rms_in4, rms_norm4_outputs, true /* execute_sync*/);
        });
      });
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm4_outputs[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, rmsnorm4_out_size, rmsnorm4_out_offset
        );
      });
      sln2_output_bo = rms_norm4_outputs[0].map<uint16_t*>();  // dp_output
    }
  }

  if (supported_shapes) {
    // MY_LOG(2) << "- AMD SSLRN 2 AIE...";
    std::vector<xrt::bo> add2_outputs = {ss_->add_->get_inputs()[0]};
    auto rms_norm2_wt_inputs = ss_->rms_norm2_->get_inputs()[1];
    std::vector<xrt::bo> rms_norm2_outputs;
    rms_norm2_outputs = {ss_->add_->get_inputs()[1]};

    auto output = ctx.GetOutput(0, dimensions_input);  // Output activation
    auto ssmlp_out_data = output.GetTensorMutableData<uint16_t>();
    auto output_count = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // rms_norm2_->set_kernel_shape(a_shape);
    constexpr bool rmsnorm_cpu_en = false;
    if (num_outputs == 2) {
      // Input BO
      PROFILING_START(execute_add)
      std::vector<xrt::bo> add_in;
      if (is_ssgmlp_) {
        add_in = {add1_outputs_[0], rms_norm4_outputs[0]};
      } else {
        add_in = {add1_outputs_[0], dp_outputs[0]};
      }
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->add_->execute(
            add_in, add2_outputs, rmsnorm_cpu_en || execute_async
          );
        });
      });
      PROFILING_END(execute_add, false, name().c_str())

      auto skip_input_bias_add_output =
        ctx.GetOutput(1, dimensions_input);  // Output activation

      auto skip_out_data =
        skip_input_bias_add_output.GetTensorMutableData<uint16_t>();

      const auto skip_out_count =
        skip_input_bias_add_output.GetTensorTypeAndShapeInfo()
          .GetElementCount();

      uint16_t* rms_wt2_map = rms_norm2_wt_inputs.template map<uint16_t*>();
      MemCpy((void*)rms_wt2_map, wts2_.data(), num_el_bo2);

      const auto rmsnorm2_wts_size = num_el_bo2;
      const auto rmsnorm2_wts_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm2_wt_inputs.sync(
          XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm2_wts_size, rmsnorm2_wts_offset
        );
      });
      std::vector<xrt::bo> rms_in = {add2_outputs[0], rms_norm2_wt_inputs};

      PROFILING_START(execute_rmsnorm)
      if constexpr (!rmsnorm_cpu_en) {
        tryContinueOnException([&]() {
          RecordDuration(Metric::KernelExecution, [&]() {
            ss_->rms_norm_->execute(rms_in, rms_norm2_outputs, execute_sync);
          });
        });
        PROFILING_END(execute_rmsnorm, false, name().c_str())
      } else {
        // std::cout <<"rmsnorm2 on cpu M = " << M << ", K = " << K << "\n";
        std::uint16_t* in_ptr = add2_outputs[0].map<std::uint16_t*>();
        std::uint16_t* out_ptr = rms_norm2_outputs[0].map<std::uint16_t*>();
        std::uint16_t* const_ptr = getRmsNormConstData(rms_wt2_map);

        RecordDuration(Metric::XRTBOSync, [&]() {
          add2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        });

        rmsnorm_cpu(in_ptr, const_ptr, out_ptr, M, K, epsilon_);

        RecordDuration(Metric::XRTBOSync, [&]() {
          rms_norm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        });
      }

      const auto add2_out_size = skip_out_count * sizeof(std::uint16_t);
      const auto add2_out_offset = 0;

      RecordDuration(Metric::XRTBOSync, [&]() {
        add2_outputs[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, add2_out_size, add2_out_offset
        );
      });

      uint16_t* output_data_1 = add2_outputs[0].map<uint16_t*>();
      // Create NPU output tensor

      if (!output_cast_indices_.empty()) {
        RecordDuration(Metric::Casting, [&]() {
          ort_cast_bf16_to_fp16_.execute(
            (Ort::Float16_t*)skip_out_data, (Ort::BFloat16_t*)output_data_1,
            dimensions_input, context
          );
        });
      } else {
        MemCpy(skip_out_data, output_data_1, B * M * K * sizeof(uint16_t));
      }

    } else {
      PROFILING_START(execute_add)
      std::vector<xrt::bo> add_in = {add1_outputs_[0], dp_outputs[0]};
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->add_->execute(
            add_in, add2_outputs, rmsnorm_cpu_en || execute_async
          );
        });
      });
      PROFILING_END(execute_add, false, name().c_str())
      uint16_t* rms_wt2_map = rms_norm2_wt_inputs.template map<uint16_t*>();
      MemCpy((void*)rms_wt2_map, wts2_.data(), num_el_bo2);

      const auto rmsnorm2_wts_size = num_el_bo2;
      const auto rmsnorm2_wts_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm2_wt_inputs.sync(
          XCL_BO_SYNC_BO_TO_DEVICE, rmsnorm2_wts_size, rmsnorm2_wts_offset
        );
      });

      // Execute RMS Norm
      if constexpr (!rmsnorm_cpu_en) {
        PROFILING_START(execute_rmsnorm)
        std::vector<xrt::bo> rms_in = {add2_outputs[0], rms_norm2_wt_inputs};
        tryContinueOnException([&]() {
          RecordDuration(Metric::KernelExecution, [&]() {
            ss_->rms_norm_->execute(rms_in, rms_norm2_outputs, execute_sync);
          });
        });
        PROFILING_END(execute_rmsnorm, false, name().c_str())
      } else {
        // std::cout <<"rmsnorm2 on cpu M = " << M << ", K = " << K << "\n";

        std::uint16_t* in_ptr = add2_outputs[0].map<std::uint16_t*>();
        std::uint16_t* out_ptr = rms_norm2_outputs[0].map<std::uint16_t*>();
        std::uint16_t* const_ptr = getRmsNormConstData(rms_wt2_map);

        RecordDuration(Metric::XRTBOSync, [&]() {
          add2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        });

        rmsnorm_cpu(in_ptr, const_ptr, out_ptr, M, K, epsilon_);

        RecordDuration(Metric::XRTBOSync, [&]() {
          rms_norm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        });
      }
    }

    // copy and sync rms_norm_outputs

    const auto rmsnorm2_out_size = B * M * K * sizeof(std::uint16_t);
    const auto rmsnorm2_out_offset = 0;

    PROFILING_START(output_format)
    RecordDuration(Metric::XRTBOSync, [&]() {
      rms_norm2_outputs[0].sync(
        XCL_BO_SYNC_BO_FROM_DEVICE, rmsnorm2_out_size, rmsnorm2_out_offset
      );
    });
    uint16_t* output_data_ = rms_norm2_outputs[0].map<uint16_t*>();
    if (!output_cast_indices_.empty()) {
      RecordDuration(Metric::Casting, [&]() {
        ort_cast_bf16_to_fp16_.execute(
          (Ort::Float16_t*)ssmlp_out_data, (Ort::BFloat16_t*)output_data_,
          dimensions_input, context
        );
      });
    } else {
      MemCpy(ssmlp_out_data, output_data_, B * M * K * sizeof(uint16_t));
    }
    PROFILING_END(output_format, false, name().c_str())

  } else {
    uint16_t* dp_output_bo = nullptr;
    if (!is_ssgmlp_) {
      // syncing DP OUTPUT from MLP

      const auto dp_output_size = num_elements * sizeof(std::uint16_t);
      const auto dp_output_offset = 0;
      RecordDuration(Metric::XRTBOSync, [&]() {
        dp_outputs[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, dp_output_size, dp_output_offset
        );  // dp_output
      });
      dp_output_bo = dp_outputs[0].map<uint16_t*>();  // dp_output
    }

    {
      if (!input_a)
        input_a = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!input_b)
        input_b = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!output_1)
        output_1 = allocator_.AllocateBuffer(num_elements * sizeof(float));
      if (!output_2)
        output_2 = allocator_.AllocateBuffer(num_elements * sizeof(float));

      RecordDuration(Metric::Casting, [&]() {
        auto* output_data = is_ssgmlp_ ? sln2_output_bo : dp_output_bo;
        ryzenai::bfloat16_buffer_to_float(
          output_data, num_elements, input_b.Data<float>()
        );
      });
    }

    // Define input shape
    std::vector<int64_t> sslrn_input_shape = {
      (int64_t)B, (int64_t)M, (int64_t)K
    };  // Replace with your actual input shape

    auto dimensions_wts2 = m2_weights.GetTensorTypeAndShapeInfo().GetShape();

    ort_sslrn_.execute(
      output_1.Data<float>(), input_a.Data<float>(), output_2.Data<float>(),
      input_b.Data<float>(), sslrn_input_shape, wts2_data_.data(),
      dimensions_wts2, context
    );

    auto output_token_tensor =
      ctx.GetOutput(0, dimensions_input);  // Output activation
    auto sslrn_out_data_token =
      output_token_tensor.GetTensorMutableData<uint16_t>();

    auto sslrn_cnt =
      output_token_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    RecordDuration(Metric::Casting, [&]() {
      if (!output_cast_indices_.empty()) {
        ort_cast_fp32_to_fp16_.execute(
          (Ort::Float16_t*)sslrn_out_data_token, output_1.Data<float>(),
          dimensions_input, context
        );
      } else
        ryzenai::float_buffer_to_bfloat16(
          output_1.Data<float>(), M * K, sslrn_out_data_token
        );  // M x K
    });

    if (num_outputs == 2) {
      auto skip_input_bias_add_output_token =
        ctx.GetOutput(1, dimensions_input);  // Output activation
      auto skip_cnt =
        skip_input_bias_add_output_token.GetTensorTypeAndShapeInfo()
          .GetElementCount();
      auto skip_out_data_token =
        skip_input_bias_add_output_token.GetTensorMutableData<uint16_t>();

      RecordDuration(Metric::Casting, [&]() {
        if (!output_cast_indices_.empty()) {
          ort_cast_fp32_to_fp16_.execute(
            (Ort::Float16_t*)skip_out_data_token, input_a.Data<float>(),
            dimensions_input, context
          );
        } else
          ryzenai::float_buffer_to_bfloat16(
            input_a.Data<float>(), M * K, skip_out_data_token
          );  // M x K
      });
    }

    // MY_LOG(2) << "- AMD SSLRN2 CPU done ...\n";
  }

  if (input_cast[1]) std::vector<Ort::BFloat16_t>().swap(npu_skip_buffer_);

  if (useExternalData()) {
    this->unloadData();
  }

  if (freeAfterPrefill(name())) {
    ss_->rms_norm_.reset();
    ss_->add_.reset();
    ss_->rms_norm2_.reset();
    if (lastNode(name()) && getPrefillBufferRelease(npu_kernel_size)) {
      shared_buffer_.Reset();
    }
  }

#ifdef NPU_SS_MLP_PROFILE_EN
  const Clock::time_point compute_end = Clock::now();

  const Duration setup_duration = input_format_start - compute_start;
  const Duration input_format_duration = input_format_end - input_format_start;
  const Duration add_rms_0_duration = add_rms_0_end - input_format_end;
  const Duration mlp_0_duration = mlp_0_end - add_rms_0_end;
  const Duration add_rms_1_duration = compute_end - mlp_0_end;

  measurements_.at(EventID::SETUP_ID) += setup_duration;
  measurements_.at(EventID::INPUT_FORMAT_ID) += input_format_duration;
  measurements_.at(EventID::LRN0_EXECUTE_ID) += add_rms_0_duration;
  measurements_.at(EventID::MLP_EXECUTE_ID) += mlp_0_duration;
  measurements_.at(EventID::LRN1_EXECUTE_ID) += add_rms_1_duration;
#endif
  // MY_LOG(2) << "- AMD SSMLP compute done ...\n";
  PROFILING_END(Compute, false, name().c_str())
}  // end of compute

}  // namespace ryzenai::onnx_utils
