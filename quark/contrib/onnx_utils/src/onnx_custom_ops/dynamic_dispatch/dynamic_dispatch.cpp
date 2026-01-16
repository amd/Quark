// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "dynamic_dispatch.hpp"

#include "ort.hpp"
#include "profiling/profiling.hpp"

namespace {

template <int kHeads, int kHiddenSize>
void saveKVcache(
  const std::string& filename, const void* data, size_t total_seq_len
) {
  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    throw std::ios_base::failure("Failed to open file");
  }
  const auto batch_size = 1;
  const auto size =
    batch_size * kHeads * total_seq_len * kHiddenSize * 2 * sizeof(uint16_t);
  file.write(static_cast<const char*>(data), size);

  if (!file) {
    throw std::ios_base::failure("Failed to write data to file");
  }
  file.close();
}

template <int kHeads, int kHiddenSize>
void loadKVCache(
  const std::string& filename, void* data, size_t output_total_seq_len,
  bool load_k = true, bool load_v = true
) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::ios_base::failure("Failed to open file: " + filename);
  }

  file.seekg(0, std::ios::end);
  const auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  const auto batch_size = 1;

  const auto cache_const_size = batch_size * kHeads * kHiddenSize;
  const auto input_total_seq_len =
    file_size / cache_const_size / 2 / sizeof(uint16_t);

  // if the saved data's seq_len is the same as the new destination, we can read
  // it straight
  if (output_total_seq_len == input_total_seq_len) {
    file.read(static_cast<char*>(data), file_size);
    return;
  }

  // if not, we need to shuffle the KV cache into the new space
  auto total_seq_len = std::min(input_total_seq_len, output_total_seq_len);
  std::vector<char> read_data(file_size);
  file.read(read_data.data(), file_size);

  // offsets for k_cache and v_cache, respectively
  std::array<size_t, 2> input_offsets = {
    0U, cache_const_size * input_total_seq_len * sizeof(uint16_t)
  };
  std::array<size_t, 2> output_offsets = {
    0U, cache_const_size * output_total_seq_len * sizeof(uint16_t)
  };

  auto offset_min = load_k ? 0 : 1;
  auto offset_max = load_v ? 2 : 1;

  for (int k = offset_min; k < offset_max; ++k) {
    for (int i = 0; i < kHeads; ++i) {
      for (int j = 0; j < total_seq_len; ++j) {
        auto* dst =
          static_cast<char*>(data) + output_offsets[k] +
          ((i * output_total_seq_len * kHiddenSize) + (j * kHiddenSize)) *
            sizeof(uint16_t);
        const auto* src =
          read_data.data() + input_offsets[k] +
          ((i * input_total_seq_len * kHiddenSize) + (j * kHiddenSize)) *
            sizeof(uint16_t);

        std::memcpy(dst, src, kHiddenSize * sizeof(uint16_t));
      }
    }
  }
}

}  // namespace

namespace ryzenai::onnx_utils {

namespace {
bool isLlmModel(ModelType model_type) {
  if (model_type != ModelType::Unknown) {
    return model_type == ModelType::Llm_Token ||
           model_type == ModelType::Llm_Prefill;
  }
  return false;
}

size_t getStackSize(
  ModelType model_type,
  const std::unordered_map<std::string, std::string>& session_configs,
  bool preemption
) {
  const auto& stack_size_str = session_configs.at("fusion_opt_stack_size");
  if (stack_size_str.empty()) {
    // for legacy models, there may not be an value
    if (model_type == ModelType::Llm_Prefill) {
      // prefill fusion needs a larger stack size for large dynamic shapes
      return 30;
    } else if (model_type == ModelType::Sd30_Mmdit ||
               model_type == ModelType::Sd30_VAE) {
      return preemption ? 30 : 60;
    }
    // use DD default value otherwise
    return 0;
  }
  return std::stoull(stack_size_str);
}

}  // namespace

std::string getCacheDirectory(
  const std::unordered_map<std::string, std::string>& session_configs
) {
  if (const auto& dd_cache_config = session_configs.at("dd_cache");
      !dd_cache_config.empty()) {
    // Need to remove ".cache" at the end of the string if it's present as on
    // legacy models
    auto pos = dd_cache_config.find(".cache");
    if (pos != std::string::npos) {
      return dd_cache_config.substr(0, pos - 1);
    }
    return dd_cache_config;
  }

  std::string model_directory;
  if (const auto& external_data_file = session_configs.at("external_data_file");
      !external_data_file.empty()) {
    auto external_data_path = fs::path(external_data_file);
    if (fs::is_directory(external_data_path)) {
      return external_data_file;
    }
    return external_data_path.parent_path().string();
  }

  throw std::runtime_error(
    "At least one of external_data_file and dd_cache session options must be "
    "set"
  );
}

DynamicDispatchKernelBase::DynamicDispatchKernelBase(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : dd_root_(session_configs.at("dd_root")),
    compile_fusion_rt_(session_configs.at("compile_fusion_rt")),
    dd_const_key_(session_configs.at("onnx_custom_ops_const_key")) {
  auto info_ptr = Ort::ConstKernelInfo(info);

  node_name_ = info_ptr.GetNodeName();
  cache_directory_ = getCacheDirectory(session_configs);
  read_attributes(info_ptr);
  Lora::enableLora(session_configs);

  try {
    initialize_fusion_rt(info_ptr, session_configs);
    if (Lora::isEnabled()) {
      external_buffers_.zeroesAllLoraData();
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize fusion runtime for node '" << node_name_
              << "': " << e.what();
    throw;
  }

  Lora::addLoraOp(this);
}

std::pair<std::string, OpsFusion::DDConfig> getConfig(
  ModelType model_type, bool preemption, const std::string& model_hash
) {
  std::string kernel_name;
  OpsFusion::DDConfig cfg = {};

  switch (model_type) {
    case ModelType::Unet:
    case ModelType::Unet_Bfp:
      kernel_name = "DPU";
      cfg.model_name = "DPU";
      break;
    case ModelType::Decoder:
    case ModelType::Decoder_Bfp:
      kernel_name = "VAE";
      cfg.model_name = "VAE";
      break;
    case ModelType::Sd15_Unet:
    case ModelType::Sd15_Decoder:
    case ModelType::Sd30_Mmdit:
    case ModelType::Sd30_VAE:
      kernel_name = "SD";
      cfg.model_name = "SD";
      cfg.en_dynamic_subgraph_lazy_loading = true;
      cfg.optimize_pm_swap = false;
      if (preemption) {
        cfg.use_elf_flow = true;
        cfg.enable_preemption = true;
      } else {
        cfg.use_elf_flow = false;
      }
      if (model_type == ModelType::Sd30_Mmdit ||
          model_type == ModelType::Sd30_VAE) {
        cfg.use_lazy_scratch_bo = false;
      }
      break;
    case ModelType::Llm_Prefill:
      kernel_name = "DPU";
      cfg.model_name = "DPU";
      cfg.use_elf_flow = false;
      cfg.enable_preemption = false;
      cfg.use_lazy_scratch_bo = false;
      cfg.en_dynamic_subgraph_lazy_loading = true;
      cfg.txn_opt = false;
      if (!model_hash.empty()) cfg.constbo_sharing_key = model_hash;
      break;
    case ModelType::Llm_Token:
      kernel_name = "DPU_ELF";
      cfg.model_name = "DPU_ELF";
      cfg.use_elf_flow = true;
      // cfg.eager_mode = true;
      cfg.enable_preemption = false;
      if (!model_hash.empty()) cfg.const_iter_key = model_hash;

      break;
    default:
      throw std::runtime_error(
        "Unknown model type: " + std::to_string(static_cast<int>(model_type))
      );
  }

  return {kernel_name, cfg};
}

const std::string ModelTypeToString(ModelType type) {
  switch (type) {
    case ModelType::Unet:
      return "Unet";
    case ModelType::Decoder:
      return "Decoder";
    case ModelType::Unet_Bfp:
      return "Unet_Bfp";
    case ModelType::Decoder_Bfp:
      return "Decoder_Bfp";
    case ModelType::Sd15_Unet:
      return "SD15_Unet";
    case ModelType::Sd15_Decoder:
      return "Sd15_Decoder";
    case ModelType::Sd30_Mmdit:
      return "Sd30_Mmdit";
    case ModelType::Sd30_VAE:
      return "Sd30_VAE";
    case ModelType::Llm_Prefill:
      return "Llm_Prefill";
    case ModelType::Llm_Token:
      return "Llm_Token";
    case ModelType::Unknown:
      return "Unknown";
    default:
      throw std::runtime_error("Invalid ModelType in ModelTypeToString");
  }
}

void DynamicDispatchKernelBase::initialize_fusion_rt(
  const Ort::ConstKernelInfo& info,
  const std::unordered_map<std::string, std::string>& session_configs
) {
  if (rt_ == nullptr) {
    auto dd_cache = cache_directory_ + "/.cache";
    const auto op_dir = dd_cache + "/" + node_name_;
    const auto meta_json = op_dir + "_meta.json";
    meta_ = OpsFusion::load_meta_json(meta_json);

    bool preemption_default =
      session_configs.at("hybrid_opt_enable_npu_preemption") == "1";
    bool preemption =
      getAttribute<int64_t>(info, "preemption", preemption_default) == 1;
    auto model_hash = getAttribute<std::string>(info, "model_hash", "");

    auto [kernel_name, cfg] =
      model_type_ == ModelType::Unknown
        ? getConfig(model_type_, preemption, model_hash)
        : getConfig(model_type_, preemption, model_hash);
    cfg.cache_dir = cache_directory_;
    if (auto xclbin_path = dd_root_ + xclbin_;
        xclbin_[0] == '/' && fs::exists(fs::path(xclbin_path))) {
      // legacy name and external xclbin file exists
      auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_path);
      rt_ = std::make_unique<OpsFusion::FusionRuntime>(
        xclbin_path, xclbin_content, kernel_name
      );
    } else {
      if (xclbin_[0] == '/') {
        // legacy name but no external xclbin file exists so assume it's in
        // the shared library
        xclbin_ = "stx_" + fs::path(xclbin_).stem().string();
      }
      rt_ = std::make_unique<OpsFusion::FusionRuntime>(xclbin_, kernel_name);
    }

    cfg.instr_xrt_bo_stack_size_mb =
      getStackSize(model_type_, session_configs, preemption);

    // TODO(varunsh): once model_name_ is removed, it can be deleted here
    const auto metastate_filename = "dd_metastate_" +
                                    ModelTypeToString(model_type_) + "_" +
                                    node_name_ + ".state";
    const auto metastate_filepath = cache_directory_ + "/" + metastate_filename;
    bool force_compile =
      !compile_fusion_rt_.empty() && compile_fusion_rt_ == "1";
    if (force_compile || !fs::exists(fs::path(metastate_filepath))) {
      rt_->compile(meta_, "", cfg);
      rt_->save_state(metastate_filename);
    }
    rt_->load_state(
      metastate_filepath, OpsFusion::empty_uniq<OpsFusion::MetaIOAPI>(),
      dd_const_key_, cfg
    );

    rt_->init(meta_, "", cfg);

    if (meta_.dynamic_shape_subgraph) {
      dynamic_dim_maps_ = meta_.dynamic_shape_list;
    }

    auto skip_ext_buf_copy = session_configs.at("fusion_opt_skip_ext_buf_copy");
    bool io_bind_kv_cache =
      session_configs.at("fusion_opt_io_bind_kv_cache") == "1";
    external_buffers_.construct(
      info, meta_, model_type_, skip_ext_buf_copy, legacy_llm_prefill_model_,
      io_bind_kv_cache
    );
  }
}

void DynamicDispatchKernelBase::read_attributes(
  const Ort::ConstKernelInfo& info_ptr
) {
  xclbin_ = info_ptr.GetAttribute<std::string>("xclbin");
  try {
    model_type_ = ModelType(info_ptr.GetAttribute<int64_t>("model_type"));
  } catch (const Ort::Exception&) {
    // if it's not found, continue with an default value
  }

  if (model_type_ == ModelType::Llm_Prefill) {
    try {
      seq_len_index_ = info_ptr.GetAttribute<int64_t>("seq_len");
    } catch (const Ort::Exception&) {
      // all new models should have this attribute but legacy models do not
      seq_len_index_ = info_ptr.GetInputCount() - 1;
      legacy_llm_prefill_model_ = true;
    }
  }

  size_t input_num;
  try {
    input_num = info_ptr.GetAttribute<int64_t>("input_num");
  } catch (const Ort::Exception&) {
    input_num = legacy_llm_prefill_model_ ? info_ptr.GetInputCount() - 1
                                          : info_ptr.GetInputCount();
  }
  if (input_num < 1) {
    throw std::invalid_argument("input_num attribute must be greater than 0");
  }
  input_shapes_.resize(input_num);
  dynamic_input_shapes_.resize(input_num);
  is_inputs_dims_dynamic_.resize(input_num);

  for (int i = 0; i < input_num; ++i) {
    auto key = "input_shape_" + std::to_string(i);
    input_shapes_[i] = info_ptr.GetAttributes<int64_t>(key.c_str());
    if (input_shapes_[i].empty()) {
      parse_dynamic_dims_string(
        info_ptr.GetAttribute<std::string>(key.c_str()),
        dynamic_input_shapes_[i], is_inputs_dims_dynamic_[i]
      );
    }
  }

  auto output_num = info_ptr.GetOutputCount();
  if (output_num < 1) {
    throw std::invalid_argument("output_num attribute must be greater than 0");
  }
  output_shapes_.resize(output_num);
  dynamic_output_shapes_.resize(output_num);
  is_outputs_dims_dynamic_.resize(output_num);
  for (int i = 0; i < output_num; ++i) {
    auto key = "output_shape_" + std::to_string(i);
    output_shapes_[i] = info_ptr.GetAttributes<int64_t>(key.c_str());
    if (output_shapes_[i].size() == 0) {
      parse_dynamic_dims_string(
        info_ptr.GetAttribute<std::string>(key.c_str()),
        dynamic_output_shapes_[i], is_outputs_dims_dynamic_[i]
      );
    }
  }
  try {
    inp_shapes_padding_ = info_ptr.GetAttributes<int64_t>("inp_padding");
  } catch (const Ort::Exception&) {
    // if it's not found, continue with an empty inp_shapes_padding_
  }
  try {
    out_shapes_padding_ = info_ptr.GetAttributes<int64_t>("out_padding");
  } catch (const Ort::Exception&) {
    // if it's not found, continue with an empty out_shapes_padding_
  }
}

void DynamicDispatchKernelBase::parse_dynamic_dims_string(
  const std::string& dynamic_dims_str, std::vector<std::string>& dims,
  std::vector<bool>& is_dynamic
) {
  dims.clear();
  is_dynamic.clear();
  std::istringstream dim_stream(dynamic_dims_str);
  std::string dim;
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
    dims.push_back(dim);
    is_dynamic.push_back(std::any_of(dim.begin(), dim.end(), ::isalpha));
  }
}

int64_t DynamicDispatchKernelBase::find_dynamic_shape_list(
  const OpsFusion::DynamicShapeInfo& partial_shape_info
) {
  for (int64_t i = 0; i < dynamic_dim_maps_.size(); ++i) {
    const auto& candidate = dynamic_dim_maps_[i];
    bool match = true;
    for (const auto& [key, val] : partial_shape_info.dyn_shapes) {
      auto it = candidate.dyn_shapes.find(key);
      if (it == candidate.dyn_shapes.end() || it->second != val) {
        match = false;
        break;
      }
    }
    if (match) {
      return i;
    }
  }
  return -1;  // no full match found
}

void DynamicDispatchKernelBase::loadLoraData() {
  static const std::map<std::string, int> lora_tensor_map = {
    {"q_proj_lora", -1},    {"k_proj_lora", -1},    {"v_proj_lora", -1},
    {"o_proj_lora", -1},    {"gate_proj_lora", -1}, {"up_proj_lora", -1},
    {"down_proj_lora", -1},
  };

  for (const auto& op : meta_.op_list) {
    for (auto input_name : op.in_args) {
      size_t prev = 0, pos = 0;
      pos = input_name.find(".", prev);
      pos = (pos == std::string::npos) ? input_name.length() : pos;
      auto lora_key = input_name.substr(prev, pos - prev);

      if (lora_tensor_map.count(lora_key)) {
        ExternalTensorInfo tensor_info = getExternalTensorInfo(
          Lora::tokenHeader(), input_name, lora_tensor_map.at(lora_key)
        );
        if (tensor_info.size > 0) {
          external_buffers_.loadLoraBin(
            input_name, tensor_info.offset, tensor_info.size, meta_
          );
        }
      }
    }
  }
}

void DynamicDispatchKernelBase::LoadLora() {
  const auto& lora_name = Lora::getLoraName();
  if (lora_name != external_buffers_.getLoraName()) {
    external_buffers_.zeroesAllLoraData();
    if (lora_name != "base") {
      loadLoraData();
    }
  }
}

void DynamicDispatchKernel::Compute(OrtKernelContext* context) {
  PROFILING_START(Compute)
  Ort::KernelContext ctx(context);
  if (external_buffers_.hasExternalBuffers()) {
    if (isLlmModel(model_type_)) {
      uint32_t seq_len;
      if (model_type_ == ModelType::Llm_Prefill) {
        auto input_tensor = ctx.GetInput(0);  // Input activations
        auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
        seq_len = static_cast<uint32_t>(input_shape[0] * input_shape[1]);
      } else {
        auto attention_mask_idx = external_buffers_.getAttentionMaskIndex();
        auto attention_mask = getInputTensor(
          ctx, static_cast<int>(attention_mask_idx),
          input_shapes_[attention_mask_idx]
        );
        seq_len = *((uint32_t*)attention_mask.data) - 1;
      }

      if (past_seq_len_ == 0 || seq_len != past_seq_len_ + 1) {
        // TODO(varunsh): this is assuming the new prompt will not be the same
        // size as the previous number of tokens
        PROFILING_START(external_buffers_initialize)
        RecordDuration(Metric::MemCpy, [&]() {
          external_buffers_.initialize(ctx, input_shapes_, output_shapes_);
        });
        PROFILING_END(external_buffers_initialize, false, "fusionruntime")

        // this should only happen once. DD pushes the tensors internally
        if (past_seq_len_ == 0) {
          if (external_buffers_.bindKvCache()) {
            auto kv_cache_bo = external_buffers_.getKvCacheBO();
            rt_->setup_external_bos(kv_cache_bo);
            const bool include_kv_cache = false;
            auto external_bufs =
              external_buffers_.getExternalBuffers(include_kv_cache);
            if (!external_bufs.empty()) {
              rt_->setup_external_tensors(external_bufs);
            }
          } else {
            const bool include_kv_cache = true;
            auto external_bufs =
              external_buffers_.getExternalBuffers(include_kv_cache);
            rt_->setup_external_tensors(external_bufs);
          }
        }
        PROFILING_START(external_buffers_syncForDevice)
        RecordDuration(Metric::XRTBOSync, [&]() {
          external_buffers_.syncForDevice(rt_.get());
        });
        PROFILING_END(external_buffers_syncForDevice, false, "fusionruntime")
      }

      if (past_seq_len_ == 0 || seq_len != past_seq_len_ + 1) {
        std::vector<uint32_t> state_table = {seq_len};  // prompt length
        if (model_type_ == ModelType::Llm_Token) {
          rt_->initialize_state_table(state_table);
        }
      }
      past_seq_len_ = seq_len;
    }
  }

  if (Lora::isEnabled()) {
    PROFILING_START(LoadLoraData)
    const auto& lora_name = Lora::getLoraName();
    if (lora_name != external_buffers_.getLoraName()) {
      Lora::releasePrefillHeader();
      external_buffers_.setLora(lora_name);
      external_buffers_.syncLora((void*)rt_.get());
      Lora::releaseTokenHeader();
    }
    PROFILING_END(LoadLoraData, false, "fusionruntime")
  }

  auto input_indices = external_buffers_.getInputIndices();
  std::vector<::Tensor> input_Tensor;
  input_Tensor.reserve(input_indices.size());
  std::vector<std::vector<int64_t>> runtime_input_shapes;
  runtime_input_shapes.reserve(input_indices.size());
  OpsFusion::DynamicShapeInfo dynamic_dim_map;
  size_t tensor_idx = 0;
  for (const auto& index : input_indices) {
    ::Tensor input_tensor;
    runtime_input_shapes.push_back(
      ctx.GetInput(index).GetTensorTypeAndShapeInfo().GetShape()
    );
    if (input_shapes_[index].size() != 0) {
      input_tensor = getInputTensor(ctx, index, input_shapes_[index]);
    } else {
      input_tensor =
        getInputTensor(ctx, index, runtime_input_shapes.at(tensor_idx));
      for (auto dim_i = 0; dim_i < is_inputs_dims_dynamic_[index].size();
           dim_i++) {
        if (is_inputs_dims_dynamic_[index][dim_i] == false) {
          continue;
        }
        dynamic_dim_map.dyn_shapes.insert(
          std::make_pair(
            dynamic_input_shapes_[index][dim_i],
            runtime_input_shapes[tensor_idx][dim_i]
          )
        );
      }
    }

    input_Tensor.push_back(input_tensor);
    tensor_idx++;
  }

  int64_t symbolic_dims_indice = find_dynamic_shape_list(dynamic_dim_map);

  auto output_indices = external_buffers_.getOutputIndices();
  std::vector<::Tensor> output_Tensor;
  output_Tensor.reserve(output_indices.size());
  for (const auto& index : output_indices) {
    ::Tensor output_tensor;
    if (output_shapes_[index].size() != 0) {
      output_tensor = getOutputTensor(ctx, index, output_shapes_[index]);
    } else {
      std::vector<int64_t> output_shape;
      for (auto dim_i = 0; dim_i < is_outputs_dims_dynamic_[index].size();
           dim_i++) {
        if (is_outputs_dims_dynamic_[index][dim_i] == false) {
          output_shape.push_back(stoi(dynamic_output_shapes_[index][dim_i]));
        } else {
          output_shape.push_back(
            dynamic_dim_maps_[symbolic_dims_indice].dyn_shapes.at(
              dynamic_output_shapes_[index][dim_i]
            )
          );
        }
      }
      output_tensor = getOutputTensor(ctx, index, output_shape);
    }
    output_Tensor.push_back(output_tensor);
  }

  PROFILING_START(execute)
  RecordDuration(Metric::KernelExecution, [&]() {
    rt_->execute(input_Tensor, output_Tensor);
  });
  PROFILING_END(execute, false, "fusionruntime")

  if (external_buffers_.hasExternalBuffers()) {
    PROFILING_START(syncForHost)
    RecordDuration(Metric::XRTBOSync, [&]() {
      external_buffers_.syncForHost(rt_.get());
    });
    PROFILING_END(syncForHost, false, "fusionruntime")
    if (model_type_ == ModelType::Llm_Prefill) {
      PROFILING_START(copyKvCacheToOrtTensors)
      RecordDuration(Metric::Casting, [&]() {
        external_buffers_.copyKvCacheToOrtTensors(
          ctx, output_shapes_, seq_len_index_, true
        );
      });
      PROFILING_END(copyKvCacheToOrtTensors, false, "fusionruntime")
    } else {
      if (external_buffers_.bindKvCache()) {
        auto external_output_indices =
          external_buffers_.getExternalOutputIndices();
        for (auto index : external_output_indices) {
          getOutputTensor(ctx, static_cast<int>(index), output_shapes_[index]);
        }
      } else {
        PROFILING_START(copyKvCacheToOrtTensors)
        RecordDuration(Metric::Casting, [&]() {
          external_buffers_.copyKvCacheToOrtTensors(
            ctx, output_shapes_, past_seq_len_, false
          );
        });
        PROFILING_END(copyKvCacheToOrtTensors, false, "fusionruntime")
      }
    }
  }

  PROFILING_END(Compute, false, "fusionruntime")
}

}  // namespace ryzenai::onnx_utils
