/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 * Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 * Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "slrn.hpp"

#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "common.hpp"
#include "external_data.hpp"
#include "jit_node_impl.hpp"

namespace ryzenai::onnx_utils {

template class JitNode<AMDSLRNKernel>;
using NPUTensor = ::Tensor;

void AMDSLRNKernel::initializeKernels() {
  if (mladfVersion() != "v1" && mladfVersion() != "v2") {
    std::cerr << "Invalid version: " << mladfVersion() << std::endl;
  }

  if (!ss_->rms_norm_) {
    auto attr_rmsnorm = getCommonAttrs();

    ss_->rms_norm_ =
      std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
        "bfloat16", true, attr_rmsnorm
      );

    ss_->seq_len_ = 0;
  }
}

AMDSLRNKernel::AMDSLRNKernel(
  const OrtKernelInfo* k_info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : JitNode(this, session_configs),
    NpuOp(k_info, session_configs),
    ss_(session_configs) {
#ifdef NPU_SLRN_PROFILE_EN
  measurements_.resize(EventID::MAX_EVENTS);
  const Clock::time_point config_start = Clock::now();
#endif

  // Get constant info for the node
  Ort::ConstKernelInfo info{k_info};

  auto header = initializeNpuOp(op_type_, session_configs, info);

  // Get Logger
  logger_ = info.GetLogger();

  shape_in_ = info.GetAttributes<int64_t>("shape_in");

  ort_cast_bf16_to_fp16_.construct(info);
  ort_cast_fp16_to_bf16_.construct(info);

  // Get attributes
  epsilon_ = info.GetAttribute<float>("epsilon");
  axis_ = info.GetAttribute<int64_t>("axis");
  stash_type_ = info.GetAttribute<int64_t>("stash_type");
  setMladfVersion(info);

  initializeKernels();

  try {
    input_cast_indices_ = info.GetAttributes<int64_t>("hybrid_llm_cast_input");
  } catch (const Ort::Exception&) {
    // cast input activations by default
    input_cast_indices_ = {0, 1};
  }

  try {
    output_cast_indices_ =
      info.GetAttributes<int64_t>("hybrid_llm_cast_output");
  } catch (const Ort::Exception&) {
    // cast output activations by default
    output_cast_indices_ = {0};
  }

  ort_slrn_.construct(info);

  manageDynamicDpmState();

  int is_constant = 0;
  m_weights = info.GetTensorConstantInput(1, &is_constant);

  auto dimensions_wts = m_weights.GetTensorTypeAndShapeInfo().GetShape();

  num_el = std::accumulate(
    dimensions_wts.begin(), dimensions_wts.end(), (size_t)1,
    std::multiplies<int64_t>()
  );

  wts_data_.reserve(num_el);
  if (m_weights.GetTensorTypeAndShapeInfo().GetElementType() ==
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* wts_data_ort = m_weights.GetTensorData<Ort::Float16_t>();
    for (int i = 0; i < num_el; i++) {
      wts_data_.push_back(wts_data_ort[i].ToFloat());
    }
  } else {
    const auto* wts_data_ort = m_weights.GetTensorData<float>();
    for (int i = 0; i < num_el; i++) {
      wts_data_.push_back(wts_data_ort[i]);
    }
  }

  wts_ =
    init_rmsnorm_wts(ss_->rms_norm_.get(), epsilon_, wts_data_, allocator_);
  num_el_bo = wts_.size();

  //  packed_consts=true load weights after all BO sizes set
  // loadFirstData();

  // get_buffer_size(getNPUKernelGranularity(getInitPromptSize(session_configs)));
  // inter_buffer_update();

  // MY_LOG(2) << "SLRN- Init MLP done";
#ifdef NPU_SLRN_PROFILE_EN
  const Clock::time_point config_end = Clock::now();
  const Duration config_duration = config_end - config_start;
  measurements_.at(EventID::CONFIG_ID) += config_duration;
#endif
}

AMDSLRNKernel::~AMDSLRNKernel() {
  shared_buffer_.Reset();

#ifdef NPU_SLRN_PROFILE_EN
  std::ostringstream os;

  int event_id = 0;

  for (const auto& measurement : measurements_) {
    os << name() << ",NPUSLRN," << event_id << ","
       << MillisecondsFp{measurement}.count() << "\n";

    event_id++;
  }

  std::cout << os.str() << std::flush;
#endif
  ss_->rms_norm_.reset();
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

void AMDSLRNKernel::Compute(OrtKernelContext* context) {
#ifdef NPU_SLRN_PROFILE_EN
  const Clock::time_point compute_start = Clock::now();
#endif

  initializeKernels();  // reinitialize kernel??

  manageDynamicDpmState();

  Ort::KernelContext ctx(context);

  auto num_inputs = ctx.GetInputCount();
  auto num_outputs = ctx.GetOutputCount();

  // passing true to execute causes it to wait
  const bool execute_sync = !continueOnException();
  const bool execute_async = false;

  auto input_tensor = ctx.GetInput(0);  // Input
  auto input_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto input_elem_cnt =
    input_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  auto input_data = input_tensor.GetTensorData<uint16_t>();

  auto wts_tensor = ctx.GetInput(1);  // wts
  auto wts_shape = wts_tensor.GetTensorTypeAndShapeInfo().GetShape();
  auto wts_data = wts_tensor.GetTensorData<uint16_t>();

  size_t B = input_shape[0];  // Batch
  size_t K =
    wts_shape[wts_shape.size() - 1];    // Hidden size = Num_heads * Head_size
  size_t M = input_elem_cnt / (K * B);  // Seq len

  auto output_tensor = ctx.GetOutput(0, input_shape);  // Output activation
  auto slrn_out_data_token = output_tensor.GetTensorMutableData<uint16_t>();

  std::vector<bool> input_cast = {false, false};
  bool output_cast = false;

  if (!input_cast_indices_.empty()) {
    for (int i = 0; i < input_cast_indices_.size(); i++) input_cast[i] = true;
  }

  uint16_t* in_data_ptr = nullptr;
  uint16_t* wts_data_ptr = nullptr;

  bool need_op0_copy = true;
  constexpr bool rmsnorm_cpu_en = false;
  std::vector<size_t> a_shape = {M, K};

  if constexpr (!rmsnorm_cpu_en) {
    if (ss_->seq_len_ != M) {
      tryContinueOnException([&]() {
        ss_->rms_norm_->set_kernel_shape(a_shape);
      });
      ss_->seq_len_ = M;
    }
  }
  if (input_cast[0]) {
    auto rms_norm_ip0 = ss_->rms_norm_->get_inputs()[0];
    uint16_t* rms_norm_input_0_map = rms_norm_ip0.map<uint16_t*>();
    RecordDuration(Metric::Casting, [&]() {
      ort_cast_fp16_to_bf16_.execute(
        (Ort::BFloat16_t*)rms_norm_input_0_map, input_tensor, context
      );
    });

    in_data_ptr = rms_norm_input_0_map;
    need_op0_copy = false;

    const size_t in_bo_size = input_elem_cnt * sizeof(std::uint16_t);
    const size_t in_bo_offset = 0;

    RecordDuration(Metric::XRTBOSync, [&]() {
      rms_norm_ip0.sync(XCL_BO_SYNC_BO_TO_DEVICE, in_bo_size, in_bo_offset);
    });

  } else {
    in_data_ptr = (uint16_t*)input_data;
  }

  if ((M > 1) ||
      (std::find(supported_lengths.begin(), supported_lengths.end(), M) !=
       supported_lengths.end())) {
    // auto output = ctx.GetOutput(0, input_shape);  // Output activation
    std::vector<xrt::bo> rms_norm_inputs = ss_->rms_norm_->get_inputs();
    std::vector<xrt::bo> rms_norm_outputs = ss_->rms_norm_->get_outputs();
    const auto add_operand_size_in_bytes =
      a_shape[0] * a_shape[1] * sizeof(uint16_t);

    if (need_op0_copy) {
      uint16_t* rms_norm_input_0_map = rms_norm_inputs[0].map<uint16_t*>();
      MemCpy(
        (void*)rms_norm_input_0_map, (uint16_t*)in_data_ptr,
        add_operand_size_in_bytes
      );
      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm_inputs[0].sync(
          XCL_BO_SYNC_BO_TO_DEVICE, add_operand_size_in_bytes, 0
        );
      });
    }

    auto rms_norm_wt_inputs = ss_->rms_norm_->get_inputs()[1];

    uint16_t* rms_wt_map = rms_norm_wt_inputs.map<uint16_t*>();
    MemCpy((void*)rms_wt_map, wts_.data(), num_el_bo);

    const size_t wts_bo_size = num_el_bo;
    const size_t wts_bo_offset = 0;
    RecordDuration(Metric::XRTBOSync, [&]() {
      rms_norm_wt_inputs.sync(
        XCL_BO_SYNC_BO_TO_DEVICE, wts_bo_size, wts_bo_offset
      );
    });

    // Execute RMS Norm
    std::uint16_t* out_ptr = rms_norm_outputs[0].map<std::uint16_t*>();
    auto slrn_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    if constexpr (!rmsnorm_cpu_en) {
      // if (ss_->seq_len_ != M) {
      //   tryContinueOnException([&]() {
      //     ss_->rms_norm_->set_kernel_shape(a_shape);
      //   });
      //   ss_->seq_len_ = M;
      // }
      std::vector<xrt::bo> rms_in = {rms_norm_inputs[0], rms_norm_wt_inputs};

      // Execute RMS Norm
      tryContinueOnException([&]() {
        RecordDuration(Metric::KernelExecution, [&]() {
          ss_->rms_norm_->execute(rms_in, rms_norm_outputs, true);
        });
      });

      const size_t out_bo_size = slrn_cnt * sizeof(uint16_t);
      const size_t out_bo_offset = 0;

      RecordDuration(Metric::XRTBOSync, [&]() {
        rms_norm_outputs[0].sync(
          XCL_BO_SYNC_BO_FROM_DEVICE, out_bo_size, out_bo_offset
        );
      });
    } else {
      std::uint16_t* in_ptr = rms_norm_inputs[0].map<std::uint16_t*>();
      std::uint16_t* const_ptr = getRmsNormConstData(rms_wt_map);

      rmsnorm_cpu(in_ptr, const_ptr, out_ptr, M, K, epsilon_);
    }

    if (!output_cast_indices_.empty()) {
      RecordDuration(Metric::Casting, [&]() {
        ort_cast_bf16_to_fp16_.execute(
          (Ort::Float16_t*)slrn_out_data_token, (Ort::BFloat16_t*)out_ptr,
          input_shape, context
        );
      });
    } else {
      MemCpy(slrn_out_data_token, out_ptr, slrn_cnt * sizeof(uint16_t));
    }

  } else {
    if (!output_cast_indices_.empty()) {
      ort_slrn_.execute(
        (Ort::Float16_t*)slrn_out_data_token, (Ort::BFloat16_t*)input_data,
        input_shape, (Ort::Float16_t*)wts_data, wts_shape, allocator_, context
      );
    } else {
      ort_slrn_.execute(
        (Ort::BFloat16_t*)slrn_out_data_token, (Ort::BFloat16_t*)input_data,
        input_shape, (Ort::Float16_t*)wts_data, wts_shape, allocator_, context
      );
    }
  }

}  // end of compute

}  // namespace ryzenai::onnx_utils
