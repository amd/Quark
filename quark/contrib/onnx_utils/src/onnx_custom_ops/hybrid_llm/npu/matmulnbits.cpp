// Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "matmulnbits.hpp"

#include "jit_node_impl.hpp"
#include "lora.hpp"
#include "npu_utils.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "profiling/profiling.hpp"

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "../gpu/opInterface.h"
#endif  // ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
#include "ort.hpp"

/*
GPU and NPU both defines Tensor class in different namespaces, GPU defines
in ryzenai::onnx_utils and NPU defines in global namespace.In order to avoid
discrepancies we are renaming NPU tensors type.
*/

using NPUTensor = ::Tensor;

namespace ryzenai::onnx_utils {

constexpr size_t kKernelBufferMinM = 128;
constexpr size_t kKernelComputeMinM = 1;

template class JitNode<AMDMatMulNBitsKernel>;

/// @brief Create mladfmatmulbias operator handle
void AMDMatMulNBitsKernel::initializeKernels() {
  if (!ss_->gemm) {
    auto mm_attrs = getCommonAttrs();
    updateAttrsForSharedWeights(mm_attrs);

    // Create operator instance
    ss_->gemm = std::make_unique<
      ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>>(
      "bfloat16", "int4", "bfloat16", true, mm_attrs
    );
    ss_->seq_len = 0;
    ss_->past_n = 0;
    ss_->past_k = 0;
    ss_->past_grp_size = 0;
  }
}

void AMDMatMulNBitsKernel::initialize(
  std::vector<int8_t>& b, std::vector<int8_t>& zeros,
  std::vector<float>& scales, std::vector<float>& bias, std::string shape_l
) {
  initializeKernels();

  // Weights shape
  std::vector<size_t> b_shape_dd = {
    static_cast<size_t>(m_K), static_cast<size_t>(m_N)
  };
  // Constant tensors
  NPUTensor weight_tensor = {b.data(), b_shape_dd, "int4"};
  NPUTensor bias_tensor = {bias.data(), {(size_t)m_block_size, 0}, "float"};
  NPUTensor scales_tensor = {scales.data(), {(size_t)m_block_size, 0}, "float"};
  NPUTensor zeros_tensor = {zeros.data(), b_shape_dd, "int4"};
  std::vector<NPUTensor> constant_tensors = {
    weight_tensor, bias_tensor, scales_tensor, zeros_tensor
  };
  // Initialize constant tensors (setting up XRT BOs)
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["max_m"] = maxSeqLength();
  attrs["group_size"] = static_cast<int>(m_block_size);
  attrs["skip_create_input"] = 1;
  attrs["skip_create_output"] = 1;
  updateAttrsForSharedWeights(attrs);

  ss_->gemm->initialize_const_params(constant_tensors, attrs);
}

void AMDMatMulNBitsKernel::initialize(
  const uint8_t* preformat_consts, size_t size, bool use_shared_buffer
) {
  initializeKernels();
  // Weights shape
  std::vector<size_t> consts_shape_dd = {size};
  // Constant tensors
  NPUTensor packed_const_tensor = {
    (uint8_t*)preformat_consts, consts_shape_dd, "uint8"
  };
  NPUTensor weight_tensor = {
    nullptr, {static_cast<size_t>(npu_k), static_cast<size_t>(npu_n)}, "int4"
  };
  std::vector<NPUTensor> constant_tensors = {
    weight_tensor, packed_const_tensor
  };
  // Initialize constant tensors (setting up XRT BOs)
  std::map<std::string, std::any> attrs;
  attrs["default_shape"] = 1;
  attrs["op_version"] = mladfVersion();
  attrs["max_m"] = maxSeqLength();
  attrs["group_size"] = static_cast<int>(m_block_size);
  attrs["num_preformat_tensors"] = 1;
  attrs["tensor_size"] = use_shared_buffer ? static_cast<int>(size) : 0;
  attrs["use_host_buffer"] = 1;
  attrs["skip_create_input"] = 1;
  attrs["skip_create_output"] = 1;
  updateAttrsForSharedWeights(attrs);
  ss_->gemm->initialize_const_params(constant_tensors, attrs);
}
/*
void AMDMatMulNBitsKernel::execute(const float* input_data, float* out,
                                   std::vector<int64_t> input_shape,
                                   std::vector<int> wts_shape, int grp_size,
                                   int run_cnt) const {
  auto gemm_ = (ryzenai::mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>*)
               gemm__.get();

  // Ryzen-AI implementation
  int M = input_shape[0] * input_shape[1];
  std::vector<size_t> a_shape = {static_cast<size_t>(M),
                                 static_cast<size_t>(input_shape[2])};

  std::vector<size_t> c_shape = {static_cast<size_t>(M),
                                 static_cast<size_t>(wts_shape[1])};
  std::vector<size_t> wts_shape_dd = {static_cast<size_t>(wts_shape[0]),
                                      static_cast<size_t>(wts_shape[1])};
  gemm_->set_shape(a_shape, wts_shape_dd, grp_size);
  Tensor output_tensor = {output_data_, c_shape, "bfloat16"};
  std::vector<Tensor> output_tensors = {output_tensor};

  if (M > 1) {
    // Exec
    uint16_t* a =
      (uint16_t*)_aligned_malloc(M * wts_shape[0] * sizeof(uint16_t), 64);
    // Convert floating point to bfloat16 using avx512
    float_to_bfloat16_avx512_unrolled(input_data, a, M * wts_shape[0]);
    // DD Tensors
    Tensor input_tensor = {(int16_t*)a, a_shape, "bfloat16"};
    std::vector<Tensor> input_tensors = {input_tensor};

    gemm_->execute_internal(input_tensors, output_tensors, run_cnt);

    _aligned_free(a);
  } else {
    // Convert floating point to bfloat16
    float_to_bfloat16_avx512_unrolled(input_data, input_data_mladf_,
                                      M * wts_shape[0]);
    // DD Tensors
    Tensor input_tensor = {(int16_t*)input_data_mladf_, a_shape, "bfloat16"};
    std::vector<Tensor> input_tensors = {input_tensor};
    gemm_->execute_internal(input_tensors, output_tensors, run_cnt);
  }
  if (wts_shape[1] > 30000) {
    bfloat16_to_float_full(output_data_, out, wts_shape[1] * M);
  } else {
    bfloat16_to_float_avx512_unrolled(output_data_, out, wts_shape[1] * M);
  }
}
*/

void AMDMatMulNBitsKernel::execute(
  const uint16_t* input_data, std::vector<int64_t> input_shape,
  uint16_t* output_data, std::vector<int64_t> output_shape,
  std::vector<int> wts_shape, int grp_size, int run_cnt, bool sync_input,
  int dst_row_index
) const {
  // Ryzen-AI implementation
  int M = input_shape[0] * input_shape[1];
  std::vector<size_t> a_shape = {
    static_cast<size_t>(M), static_cast<size_t>(npu_k)
  };

  std::vector<size_t> c_shape = {
    static_cast<size_t>(M), static_cast<size_t>(npu_n)
  };
  std::vector<size_t> wts_shape_dd = {
    static_cast<size_t>(wts_shape[0]), static_cast<size_t>(wts_shape[1])
  };
  if ((ss_->seq_len != M) || (ss_->past_n != wts_shape[1]) ||
      (ss_->past_k != wts_shape[0]) || (ss_->past_grp_size != grp_size)) {
    ss_->gemm->set_shape(a_shape, wts_shape_dd, grp_size);
    ss_->seq_len = M;
    ss_->past_n = wts_shape[1];
    ss_->past_k = wts_shape[0];
    ss_->past_grp_size = grp_size;
  }
  std::vector<xrt::bo> input_bos = {ss_->gemm_input};
  std::vector<xrt::bo> output_bos = {ss_->gemm_output};
  uint16_t* mm_in = input_bos[0].map<uint16_t*>();
  auto output_M = output_shape[1];
  auto offset = (M - output_M) * wts_shape[1];
  // MY_LOG(2) << "copy output." << std::endl;
  {
    /* NPUTensor output_tensor = {output_data, c_shape, "bfloat16"};
    std::vector<NPUTensor> output_tensors = {output_tensor};

    // Exec
    // DD Tensors
    NPUTensor input_tensor = {(int16_t*)input_data, a_shape, "bfloat16"};
    std::vector<NPUTensor> input_tensors = {input_tensor};

    gemm_->execute_internal(input_tensors, output_tensors, run_cnt); */
    if (npu_k != m_K) {
      // pad input if K not aligned
      memset(
        mm_in + dst_row_index * wts_shape[0], 0,
        output_M * wts_shape[0] * sizeof(std::uint16_t)
      );
      for (size_t idx = 0; idx < output_M; idx++) {
        MemCpy(
          mm_in + dst_row_index * wts_shape[0] + idx * wts_shape[0],
          input_data + m_K * idx, m_K * sizeof(std::uint16_t)
        );
      }
    } else {
      MemCpy(
        mm_in + dst_row_index * wts_shape[0], input_data,
        output_M * wts_shape[0] * sizeof(std::uint16_t)
      );
    }

    const size_t in_bo_size = output_M * wts_shape[0] * sizeof(std::uint16_t);
    const size_t in_bo_offset =
      dst_row_index * wts_shape[0] * sizeof(std::uint16_t);

    RecordDuration(Metric::XRTBOSync, [&]() {
      input_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE, in_bo_size, in_bo_offset);
    });

    if (shared_weights_.ready()) {
      std::vector<xrt::bo> inputs = {input_bos[0]};
      if (Lora::isEnabled()) {
        inputs.push_back(xrt::bo());  // dummy BO to fix input schema
        inputs.push_back(lora_buffers_.getBo(0));
      }
      std::vector<uint64_t> input_addrs = {0, shared_weights_.weightAddr(0), 0};
      std::vector<uint64_t> out_addrs;
      RecordDuration(Metric::KernelExecution, [&]() {
        ss_->gemm->execute(inputs, input_addrs, output_bos, out_addrs, true);
      });
    } else {
      auto const_bos = ss_->gemm->get_const();
      std::vector<xrt::bo> inputs = {input_bos[0], const_bos[run_cnt]};
      if (Lora::isEnabled()) inputs.push_back(lora_buffers_.getBo(0));
      RecordDuration(Metric::KernelExecution, [&]() {
        ss_->gemm->execute(inputs, output_bos);
      });
    }

    const size_t out_bo_size = output_M * wts_shape[1] * sizeof(std::uint16_t);
    const size_t out_bo_offset = offset * sizeof(std::uint16_t);

    RecordDuration(Metric::XRTBOSync, [&]() {
      output_bos[0].sync(
        XCL_BO_SYNC_BO_FROM_DEVICE, out_bo_size, out_bo_offset
      );
    });

    uint16_t* mm_out = output_bos[0].map<uint16_t*>();
    // MY_LOG(2) << "copy output." << std::endl;
    if (npu_n != m_N) {
      // output depad if N not aligned
      for (size_t idx = 0; idx < output_M; idx++) {
        MemCpy(
          output_data + idx * m_N, mm_out + offset + idx * wts_shape[1],
          m_N * sizeof(std::uint16_t)
        );
      }

    } else {
      MemCpy(
        output_data, mm_out + offset,
        output_M * wts_shape[1] * sizeof(std::uint16_t)
      );
    }
  }
}

// Ctor
AMDMatMulNBitsKernel::AMDMatMulNBitsKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : JitNode(this, session_configs),
    NpuOp(k_info, session_configs),
    ss_(session_configs) {
#ifdef NPU_MATMULNBITS_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif
  std::string node_name;
  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  auto op_type = "MatMulNBits";

  if (auto input_type =
        info.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
      input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    op_type = "MatMulNBitsBf";
  }

  auto header = initializeNpuOp(op_type, session_configs, info);

  // Get OP attributes
  m_N = info.GetAttribute<int64_t>("N");
  m_K = info.GetAttribute<int64_t>("K");
  npu_k =
    getAttribute<int64_t>(info, "real_K", info.GetAttribute<int64_t>("K"));
  npu_n =
    getAttribute<int64_t>(info, "real_N", info.GetAttribute<int64_t>("N"));
  m_bits = info.GetAttribute<int64_t>("bits");
  m_block_size = info.GetAttribute<int64_t>("block_size");
  setMladfVersion(info);
  try {
    m_acc_level = info.GetAttribute<int64_t>("accuracy_level");
  } catch (const Ort::Exception&) {
    m_acc_level = 0;
    std::stringstream message;
    message << "- Node: " << name() << " Setting accuracy level to default 0";
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
  }

  try {
    input_cast_indices_ = info.GetAttributes<int64_t>("hybrid_llm_cast_input");
  } catch (const Ort::Exception&) {
    // cast input activations by default
    input_cast_indices_ = {0};
  }
  shared_weights_.setWeightsCount(1);
  shared_weights_.setWeightKey(0, info, "wts_hash");

  try {
    output_cast_indices_ =
      info.GetAttributes<int64_t>("hybrid_llm_cast_output");
  } catch (const Ort::Exception&) {
    // cast output activations by default
    output_cast_indices_ = {0};
  }

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if constexpr (gpu_cast_en_) {
    dml_instance_ = DML_Ops::DMLOps::getInstance(session_configs);
    if (!output_cast_indices_.empty()) {
      ss_->vocab_size = m_N;
    }
  }
#endif

  initializeDynamicDpm(session_configs);

  initializeSharedBuffers(ss_->bo_data);

  ort_cast_fp16_to_bf16_.construct(info);
  ort_cast_bf16_to_fp16_.construct(info);

  // get packed consts
  const bool packed_consts = (6 == info.GetInputCount());

  ss_->n_sizes.push_back({static_cast<int>(npu_k), static_cast<int>(npu_n)});
  ss_->grp_sizes.push_back(m_block_size);

  if (!packed_consts) {
    // Get weights
    int is_constant = 0;
    m_weights = info.GetTensorConstantInput(1, &is_constant);
    if (is_constant) {
      const uint8_t* value = m_weights.GetTensorData<uint8_t>();
      std::stringstream message;
      message << "- Node: " << name() << " Weights[0] = " << int(value[0]);
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
    }

    // Get scales
    is_constant = 0;
    m_scales = info.GetTensorConstantInput(2, &is_constant);
    if (is_constant) {
      const float* value = m_scales.GetTensorData<float>();
      std::stringstream message;
      message << "- Node: " << name()
              << " Scales[0] = " << std::to_string(value[0]) << "\n";
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
    }

    // Get zero-points
    is_constant = 0;
    m_zeros = info.GetTensorConstantInput(3, &is_constant);
    if (is_constant) {
      m_asymmetric = true;
      const uint8_t* value = m_zeros.GetTensorData<uint8_t>();
      std::stringstream message;
      message << "- Node: " << name()
              << " Zero-points[0] = " << std::to_string(value[0]) << "\n";
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
    } else {
      m_asymmetric = false;
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "No zero-point");
    }

    // Get bias
    is_constant = 0;
    int bias_index = 4;
    m_biased = false;
    if (bias_index < info.GetInputCount()) {
      m_bias = info.GetTensorConstantInput(bias_index, &is_constant);
      auto bias_shape = m_bias.GetTensorTypeAndShapeInfo().GetShape();
      if (bias_shape[0] == 0) is_constant = 0;
      if (is_constant) {
        m_biased = true;
        const float* value = m_bias.GetTensorData<float>();
        std::stringstream message;
        message << "- Node: " << name()
                << " Bias[0] = " << std::to_string(value[0]) << "\n";

        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
      } else {
        m_biased = false;
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "No bias");
      }
    }

    std::vector<float> scales(m_K * m_N / m_block_size);
    std::vector<int8_t> b(m_K * m_N, 0);
    // fill this with zeros for Symmetric quantization
    std::vector<int8_t> zeros(m_K * m_N / m_block_size, 0);
    std::vector<float> bias(m_N, 0);  // fill with zeros

    // Original weights are in NxK/2 packed as uint8
    // Convert to KXN uint8
    // mladf version 2 only support uint input, so no need to correct data to
    // int
    uint8_t wts_type_off = 8;
    const uint8_t* wts = m_weights.GetTensorData<uint8_t>();
    for (int64_t i = 0; i < m_K; i += 2) {
      for (int64_t j = 0; j < m_N; j++) {
        auto srcv = wts[j * m_K / 2 + i / 2];
        auto src0 = (srcv & 0xf) - wts_type_off;
        auto src1 = ((srcv & 0xf0) >> 4) - wts_type_off;
        b[i * m_N + j] = static_cast<int8_t>(src0);
        b[(i + 1) * m_N + j] = static_cast<int8_t>(src1);
      }
    }

    size_t kblks = m_K / m_block_size;
    // Original Scales are in Nx(K/BlockSize) shape
    // Convert to (K/BLOCK_SIZE)xN shape
    const auto* scl = m_scales.GetTensorData<Ort::Float16_t>();
    for (int i = 0; i < m_N; i++) {
      for (int j = 0; j < kblks; j++) {
        scales[j * m_N + i] = scl[i * kblks + j].ToFloat();
      }
    }

    // fill this with zeros for Symmetric quantization
    if (m_asymmetric) {
      const uint8_t* zero_pt = m_zeros.GetTensorData<uint8_t>();
      for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < kblks; j = j + 2) {
          auto zpv = zero_pt[(i * (kblks / 2)) + (j / 2)];
          zeros[j * m_N + i] = (zpv & 0xf) - wts_type_off;
          zeros[(j + 1) * m_N + i] = ((zpv & 0xf0) >> 4) - wts_type_off;
        }
      }
    }

    // fill this with zeros for MatMul without bias
    if (m_biased) {
      const Ort::Float16_t* m_bias_ptr = m_bias.GetTensorData<Ort::Float16_t>();
      for (int i = 0; i < m_N; i++) bias[i] = m_bias_ptr[i].ToFloat();
    }

    std::string shape_list_file;  // TODO: How do we get this info?
    initialize(b, zeros, scales, bias, shape_list_file);
  } else {
    // if(packed_consts)
    int is_constant = 0;
    m_packed_consts = info.GetTensorConstantInput(5, &is_constant);

    if (is_constant) {
      if (m_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount() > 1) {
        const uint8_t* value = m_packed_consts.GetTensorData<uint8_t>();
        std::stringstream message;
        message << "- Node: " << name()
                << " Packed_Consts[0] = " << int(value[0]);
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, message.str().c_str());
        // std::vector<uint8_t> b(value, value+
        // m_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount());
        initialize(
          value, m_packed_consts.GetTensorTypeAndShapeInfo().GetElementCount()
        );
      } else {
        initializeKernels();

        if (!shared_weights_.isEnabled()) {
          jit_tensor_info_ = getExternalTensorInfo(header.get(), name(), -1);
          ss_->jit_max_bo_size = proto::getNpuMaxSize(header.get(), op_type);
          loadFirstData();
        }
      }
    }
  }

  initializeLora();

  // Register for LoRA loading via LoraCompute/LoraComputeFromBuffer
  Lora::addLoraOp(this);

  cnt = ss_->instances++;
  auto seq_len =
    info.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
  prune_logits_ = seq_len == 1 && lastNode(name());
  if (prune_logits_) {
    UpdateSharedBuffer(kKernelBufferMinM);
  } else {
    UpdateSharedBuffer(
      getNPUKernelGranularity(getInitPromptSize(session_configs))
    );
  }
#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point config_end = Clock::now();
  const Duration config_duration = config_end - config_start;
  measurements_.at(EventID::CONFIG_ID) += config_duration;
#endif
}

void AMDMatMulNBitsKernel::initializeLora() {
  if (Lora::isEnabled()) {
    lora_buffers_.addBo(m_K, m_N, ss_->gemm.get(), allocator_);
    lora_buffers_.syncLora();
  }
}

void AMDMatMulNBitsKernel::loadDataImpl(int idx) {
  auto value = (uint8_t*)ss_->bo_data[idx].Data();

  // if we're reading everything, we don't need to use a shared buffer
  const bool use_shared_buffer = isJitEnabled();

  initialize(value, ss_->bo_data[idx].Size(), use_shared_buffer);
}

void AMDMatMulNBitsKernel::readDataImpl(int idx) {
  if (shared_weights_.isEnabled()) return;
  updateJitBuffer(
    dynamicJitFactor(), ss_->bo_data[idx], jit_tensor_info_.size,
    ss_->jit_max_bo_size
  );
  ss_->run_instances++;
  // std::cout << "readDataImpl idx: " << idx << std::endl;
  loadBin(
    ss_->bo_data[idx].Data(), externalData().string(), jit_tensor_info_.size,
    jit_tensor_info_.offset
  );
}

void AMDMatMulNBitsKernel::UpdateSharedBuffer(size_t kernel_size) {
  std::vector<size_t> a_shape = {1, kernel_size, static_cast<size_t>(npu_k)};
  std::vector<size_t> b_shape = {
    static_cast<size_t>(npu_k), static_cast<size_t>(npu_n)
  };
  std::vector<size_t> c_shape = {1, kernel_size, static_cast<size_t>(npu_n)};

  NPUTensor input_tensor = {nullptr, a_shape, "bfloat16"};
  NPUTensor wts_tensor = {nullptr, b_shape, "bfloat16"};
  NPUTensor out_tensor = {nullptr, c_shape, "bfloat16"};
  NPUTensor placeholder;
  std::vector<NPUTensor> inputs = {input_tensor, wts_tensor,  placeholder,
                                   placeholder,  placeholder, out_tensor};

  // to keep tensor index same with fusion for get_buffer_reqs()
  if (Lora::isEnabled()) {
    std::vector<size_t> lora_shape = {(m_K + m_N) * LoraBuffer::getMaxRank()};
    NPUTensor lora_tensor = {nullptr, lora_shape, "bfloat16"};
    inputs.insert(inputs.begin() + 1, lora_tensor);
  }

  std::vector<NPUTensor> outputs;
  std::map<std::string, std::any> attrs;
  std::vector<int> grp_size = {static_cast<int>(m_block_size)};
  attrs["group_size"] = grp_size;
  attrs["mem_opt"] = 1;
  std::vector<OpArgMap> arg_map =
    ss_->gemm->get_buffer_reqs(inputs, outputs, attrs);
  auto size_map = get_NPU_tensor_size(arg_map, mladfVersion());
  size_t c_bo_size = size_map["out"];
  size_t a_bo_size = size_map["in0"];

  auto a_size = alignTo4096(a_bo_size);
  auto c_size = alignTo4096(c_bo_size);

  SharedBuffer::Requirements shared_buffer_reqs{
    {"in", a_size}, {"out", c_size}
  };

  if (size_map.find("scratch") != size_map.end()) {
    size_t scratch0_bo_size = size_map["scratch"];
    auto scratch0_size = alignTo4096(scratch0_bo_size);
    shared_buffer_reqs.emplace_back("scratch", scratch0_size);
    has_scratch_buffer_ = true;
  }

  shared_buffer_.Update(std::move(shared_buffer_reqs));
}

void AMDMatMulNBitsKernel::loadLoraData() {
  ExternalTensorInfo jit_tensor_info =
    getExternalTensorInfo(Lora::prefillHeader(), name(), -1);
  if (jit_tensor_info.size > 0) {
    Lora::loadBinData(
      lora_buffers_.data(0), jit_tensor_info.size, jit_tensor_info.offset
    );
  } else {
    // Need to clean up the previous lora buffers even if no new lora data is
    // loaded
    lora_buffers_.zeroesLoraData(0);
  }
  lora_buffers_.syncLora();
}

void AMDMatMulNBitsKernel::LoadLora() {
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

// Kernel Compute
void AMDMatMulNBitsKernel::Compute(
  OrtKernelContext* context, bool with_custom_allocator
) {
#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif
  PROFILING_START(Compute)
  PROFILING_START(setup)
  // Get ORT Kernel Context
  Ort::KernelContext ctx(context);

  initializeKernels();
  shared_weights_.setWeightAddr(0);

  manageDynamicDpmState();

#ifdef NPU_DEBUG
  // Get Inputs
  auto input_X = ctx.GetInput(0);  // Input activations
  auto input_W = ctx.GetInput(1);  // Input Weights
  auto input_S = ctx.GetInput(2);  // Input Scales

  auto dimensions_X = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_W = input_W.GetTensorTypeAndShapeInfo().GetShape();
  auto dimensions_S = input_S.GetTensorTypeAndShapeInfo().GetShape();

  // Get Output
  auto dimensions_out = dimensions_X;
  dimensions_out[2] = m_N;
  auto output = ctx.GetOutput(0, dimensions_out);  // Output activation
  std::cout << "- MatMulNBits Node: " << name() << "\n";
  std::cout << "  Input Activations Shape: ";
  for (auto& dim : dimensions_X) std::cout << dim << " ";
  std::cout << "\n";

  std::cout << "  Input Weights Shape: ";
  for (auto& dim : dimensions_W) std::cout << dim << " ";
  std::cout << "\n";

  std::cout << "  Input Scales Shape: ";
  for (auto& dim : dimensions_S) std::cout << dim << " ";
  std::cout << "\n";

  std::cout << "  Attribute K = " << m_K << "\n";
  std::cout << "  Attribute N = " << m_N << "\n";
  std::cout << "  Attribute bits = " << m_bits << "\n";
  std::cout << "  Attribute block_size = " << m_block_size << "\n";
  std::cout << "  Attribute accuracy_level = " << m_acc_level << "\n";

  const uint8_t* weights = m_weights.GetTensorData<uint8_t>();
  std::cout << "  Weights[0] = " << int(weights[0]) << "\n";

  const float* scales = m_scales.GetTensorData<float>();
  std::cout << "  Scales[0] = " << std::to_string(scales[0]) << "\n";

  std::cout << "  Output Activattions Shape: ";
  for (auto& dim : dimensions_out) std::cout << dim << " ";
  std::cout << "\n";
#endif
  // std::cout << "---------------Starting compute---------------\n";
  int cnt_wts = cnt;
  bool wait_for_data = false;
  if (useExternalData()) {
    readData();

    if (isJitEnabled()) {
      cnt_wts = 0;
    }
    wait_for_data = true;
  }

  PROFILING_END(setup, false, name().c_str())
  PROFILING_START(input_format)

  // Extracting the input and output information
  auto input_tensor = ctx.GetInput(0);  // Input activations
  auto input_data = input_tensor.GetTensorData<uint16_t>();  // bfloat16 input
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();

  const auto prompt_size = input_shape[0] * input_shape[1];

  const auto npu_kernel_size = getNPUKernelGranularity(prompt_size);

  auto kernel_buffer_min_M = static_cast<int64_t>(kKernelBufferMinM);
  auto prune_logits_en =
    (prune_logits_ && input_shape[0] == 1) || (1 == prompt_size);

  if (!prune_logits_en) {
    // when disable, the buffer size will be the same as it in initialization
    UpdateSharedBuffer(npu_kernel_size);
  } else {
    UpdateSharedBuffer(kernel_buffer_min_M);
  }

  // FIXME : Allocator still under test.
  Ort::MemoryInfo memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // auto* allocator_ptr = allocator == nullptr ?
  //                         ctx.GetAllocator(*((OrtMemoryInfo*)(&memory_info)))
  //                         : allocator;

  RyzenMM::BufferRef in_data_buffer;
  uint16_t* in_data_ptr = nullptr;

  // Create NPU input tensor
  // auto input_count =
  // input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  // npu_input_buffer_.resize(input_count);
  // auto input_bf16 = Ort::Value::CreateTensor<Ort::BFloat16_t>(
  //   allocator_ptr, input_shape.data(), input_shape.size());
  // std::memcpy(input_bf16.GetTensorMutableRawData(), npu_input_buffer_.data(),
  //             input_count);

#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  if constexpr (gpu_cast_en_) {
    if (ss_->init_cast_op) {
      ss_->init_cast_op = false;
      size_t size = ss_->vocab_size * input_shape[1];
      auto cast_shader = dml_instance_->getCastShader();
      ss_->async_cast_exe =
        std::async(&BF16ToFP16Shader::CreateBuffers, cast_shader, size);
    }
  }
#endif
  // Create NPU output tensor
  auto input_count = input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  // we know that matmulnbits only has one input so any value implies casting
  if (!input_cast_indices_.empty()) {
    RecordDuration(Metric::Casting, [&]() {
      in_data_buffer =
        allocator_.AllocateBuffer(sizeof(Ort::BFloat16_t) * input_count);
      ort_cast_fp16_to_bf16_.execute(
        in_data_buffer.Data<Ort::BFloat16_t>(), input_tensor, context
      );
    });
    in_data_ptr = (uint16_t*)in_data_buffer.Data();
  } else {
    in_data_ptr = (uint16_t*)input_data;
  }

  PROFILING_END(input_format, false, name().c_str())
#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point input_format_start = Clock::now();
#endif

#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point input_format_end = Clock::now();
#endif

  PROFILING_START(execute_matmul)
  PROFILING_START(create_bo)
  if (auto res = shared_buffer_.Validate("in", ss_->gemm_input)) {
    ss_->gemm_input = ss_->gemm->bind_bo(res->ptr, res->len);
  }

  if (auto res = shared_buffer_.Validate("out", ss_->gemm_output)) {
    ss_->gemm_output = ss_->gemm->bind_bo(res->ptr, res->len);
  }

  if (has_scratch_buffer_) {
    if (auto res = shared_buffer_.Validate(
          "scratch", ss_->gemm_last_scratch_ptr, ss_->gemm_last_scratch_len
        )) {
      ss_->gemm->create_bo(res->ptr, res->len, 2, kGemmBOsSelector);
      ss_->gemm_last_scratch_ptr = res->ptr;
      ss_->gemm_last_scratch_len = res->len;
    }
  }
  PROFILING_END(create_bo, false, name().c_str())

  std::vector<int64_t> out_shape;
  for (unsigned i = 0; i < (input_shape.size() - 1); i++)
    out_shape.push_back(input_shape[i]);
  out_shape.push_back(m_N);
  if (prune_logits_) {
    out_shape[1] = 1;
  }
  auto output_tensor = ctx.GetOutput(
    0, {out_shape.begin(), out_shape.end()}
  );  // Output activation
  auto out = output_tensor.GetTensorMutableData<uint16_t>();
  auto output_count =
    output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  uint16_t* out_data_ptr = (uint16_t*)out;

  if (wait_for_data) {
    auto ready = weightsReady();
    if (ready < 0) {
      std::cerr << "Disable NPU JIT with `hybrid_opt_npu_read_ahead='-1'` in "
                   "session options\n";
      throw std::invalid_argument("Weights not loaded in " + name());
    }
  }
  bool sync_input = true;

  if (prune_logits_en) {
    // we only want to copy over last row in ORT buffer to NPU HW buffer
    int src_row_idx = input_shape[1] - 1;
    uint16_t* exec_in_data = in_data_ptr + src_row_idx * input_shape[2];
    auto kernel_compute_min_M = static_cast<int64_t>(kKernelComputeMinM);
    int dst_row_idx = kernel_compute_min_M - 1;
    /* uint16_t* exec_out_data =
      out_data_ptr + (input_shape[1] - 1) * out_shape[2];
    std::vector<int64_t> exec_out_shape{out_shape[0], 1, out_shape[2]}; */

    std::vector<int64_t> exec_in_shape{
      input_shape[0], kernel_compute_min_M, input_shape[2]
    };
    execute(
      exec_in_data, exec_in_shape, out_data_ptr, out_shape, ss_->n_sizes[cnt],
      ss_->grp_sizes[cnt], cnt_wts, sync_input, dst_row_idx
    );
  } else {
    execute(
      in_data_ptr, input_shape, out_data_ptr, out_shape, ss_->n_sizes[cnt],
      ss_->grp_sizes[cnt], cnt_wts, sync_input, 0
    );
  }
  in_data_ptr = nullptr;
  in_data_buffer = {};
  PROFILING_START(JIRLoadWts)
  if (useExternalData()) {
    loadData();
  }
  PROFILING_END(JIRLoadWts, false, name().c_str())
  PROFILING_END(execute_matmul, false, name().c_str())
#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point execute_end = Clock::now();
#endif

  PROFILING_START(output_format)
  // we know that matmulnbits only has one output so any value implies casting
  if (!output_cast_indices_.empty()) {
    auto casted = false;
#ifdef ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
    if (gpu_cast_en_ && RyzenMM::IsKnown(out_data_ptr)) {
      RecordDuration(Metric::Casting, [&]() {
        ss_->async_cast_exe.wait();
        ss_->init_cast_op = true;

        try {
          const auto d3dres =
            RyzenMM::Platform::DX::GetUnderlyingD3D12ResourceUnretained(
              out_data_ptr
            );
          dml_instance_->getCastShader()->ConvertBF16ToFP16(
            d3dres, out_data_ptr, output_count
          );
          casted = true;
        } catch (...) {
          // it might be NPU memory
        }
      });
    }
#endif

    if (!casted) {
      RecordDuration(Metric::Casting, [&]() {
        ort_cast_bf16_to_fp16_.execute(
          (Ort::Float16_t*)out, (Ort::BFloat16_t*)out_data_ptr, out_shape,
          context
        );
      });
    }
  }
  PROFILING_END(output_format, false, name().c_str())

#ifdef NPU_MATMULNBITS_PROFILE_EN
  const Clock::time_point output_format_end = Clock::now();

  const Duration setup_duration = input_format_start - compute_start;
  const Duration input_format_duration = input_format_end - input_format_start;
  const Duration execute_duration = execute_end - input_format_end;
  const Duration output_format_duration = output_format_end - execute_end;

  measurements_.at(EventID::SETUP_ID) += setup_duration;
  measurements_.at(EventID::INPUT_FORMAT_ID) += input_format_duration;
  measurements_.at(EventID::OUTPUT_FORMAT_ID) += output_format_duration;
  measurements_.at(EventID::MATMUL_EXECUTE_ID) += execute_duration;
#endif

  if (useExternalData()) {
    unloadData();
  }

  if (freeAfterPrefill(name())) {
    if (lastNode(name()) && getPrefillBufferRelease(npu_kernel_size)) {
      shared_buffer_.Reset();
    }
  }
  PROFILING_END(Compute, false, name().c_str())
}

AMDMatMulNBitsKernel::~AMDMatMulNBitsKernel() {
  if (nullptr != input_data_mladf_)
#ifdef _WIN32
    _aligned_free(input_data_mladf_);
#else
    free(input_data_mladf_);
#endif
  if (nullptr != output_data_)
#ifdef _WIN32
    _aligned_free(output_data_);
#else
    free(output_data_);
#endif

#ifdef NPU_MATMULNBITS_PROFILE_EN
  std::ostringstream os;

  int event_id = 0;

  for (const auto& measurement : measurements_) {
    os << name() << ",NPUMatmulnbits," << event_id << ","
       << MillisecondsFp{measurement}.count() << "\n";
    event_id++;
  }

  std::cout << os.str() << std::flush;
#endif
  ss_->instances--;
  if (ss_->instances == 0) {
    for (auto i = 0; i < readAhead(); ++i) {
      ss_->bo_data[i] = {};
      shared_buffer_.Reset();
    }
  }
}

}  // namespace ryzenai::onnx_utils
