//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT


namespace vai_q {

struct KernelCustomLSTM {
  KernelCustomLSTM(const OrtApi& api,
		    const OrtKernelInfo* info);
  ~KernelCustomLSTM();
 
  void Compute(OrtKernelContext* context);
#if ORT_API_VERSION >= 17
  // This is for adapting to onnxruntime_cxx_api.h in ORT 1.17.0 (and higher)
  OrtStatusPtr ComputeV2(OrtKernelContext* context) { return nullptr; }
#endif

  protected:
    void ComputeBase(OrtKernelContext* context);

  private:
    const OrtApi& api_;
    Ort::KernelInfo info_{nullptr};

    std::string direction_;
    int64_t num_directions_ = 1;
    int64_t hidden_size_ = 0;
    int64_t input_forget_ = 0;
    int64_t layout_ = 0;

    float x_scale_ = 1;
    int64_t x_zero_point_ = 0;
    float w_scale_ = 1;
    int64_t w_zero_point_ = 0;
    float r_scale_ = 1;
    int64_t r_zero_point_ = 0;
    float b_scale_ = 1;
    int64_t b_zero_point_ = 0;
    float y_scale_ = 1;
    int64_t y_zero_point_ = 0;

    int64_t seq_len_ = 0;  // Length of the time sequence
    int64_t input_size_ = 0;  // Size of the input vector
};

}  // namespace vai_q
