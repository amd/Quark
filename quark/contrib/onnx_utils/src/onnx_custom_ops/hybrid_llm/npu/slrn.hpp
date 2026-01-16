/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
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
#pragma once
#include <onnxruntime_cxx_api.h>
#include <ryzenai/ryzen_mm.h>

#include <algorithm>
#include <filesystem>
#include <future>
#include <mutex>

#include "hybrid_llm/ort/cast.hpp"
#include "hybrid_llm/ort/simplified_layer_norm.hpp"
#include "jit_node.hpp"
#include "npu_op.hpp"
#include "npu_utils.hpp"
#include "ops/mladfmatmulbias/mladfmatmulbias.hpp"
#include "ops/mladfrmsnorm/mladfrmsnorm.hpp"
#include "profile.h"

namespace ryzenai::onnx_utils {

#if defined(ONNX_UTILS_ENABLE_CUSTOM_OP_PROFILING) && defined(NPU_SLRN_PROFILE)
#define NPU_SLRN_PROFILE_EN
#endif
namespace fs = std::filesystem;

class AMDSLRNKernel : public JitNode<AMDSLRNKernel>, public NpuOp {
 public:
  AMDSLRNKernel(
    const OrtKernelInfo* info,
    const std::unordered_map<std::string, std::string>& session_configs
  );
  ~AMDSLRNKernel();

  void Compute(OrtKernelContext* context);
  // stage 2
  void readDataImpl(int idx) override {};
  void loadDataImpl(int idx) override {};
  void unloadDataImpl(int idx) override {};

  void UpdateSharedBuffer(size_t kernel_size) override {};

 private:
  void initializeKernels() final;

  Ort::Logger logger_{nullptr};
  float epsilon_;
  int64_t axis_;
  int64_t stash_type_;
  std::vector<uint8_t> wts_;
  size_t num_el;
  size_t num_el_bo;
  Ort::ConstValue m_weights;
  std::vector<float> wts_data_;

  OrtSimplifiedLayerNorm ort_slrn_;
  std::vector<int64_t> shape_in_;

  std::vector<size_t> supported_lengths{32768, 16384, 8192, 4096, 3072, 2048,
                                        1920,  1792,  1664, 1536, 1408, 1280,
                                        1152,  1024,  768,  640,  512,  384,
                                        256,   128,   64,   32};

  std::vector<int64_t> input_cast_indices_;
  std::vector<int64_t> output_cast_indices_;

  OrtCast<Ort::BFloat16_t, Ort::Float16_t> ort_cast_bf16_to_fp16_;
  OrtCast<Ort::Float16_t, Ort::BFloat16_t> ort_cast_fp16_to_bf16_;

  inline static const char* const op_type_ = "SLRN";

#ifdef NPU_SLRN_PROFILE_EN
  std::vector<Duration> measurements_;
#endif

  RyzenMM::NPUAllocator<'SLRN'> allocator_;

  struct State {
    int run_instances__{0};
    int seq_len_{0};
    std::once_flag initFlag_;
    // aie kernels from DD
    std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rms_norm_{
      nullptr
    };
  };

  SessionState<State> ss_;
};

extern template class JitNode<AMDSLRNKernel>;
}  // namespace ryzenai::onnx_utils
