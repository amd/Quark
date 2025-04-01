//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT


namespace vai_q {

struct KernelCustomInstanceNormalization {
  KernelCustomInstanceNormalization(const OrtApi& api,
		             const OrtKernelInfo* info);
  ~KernelCustomInstanceNormalization();
 
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

    float epsilon_ = 1e-05;

    std::vector<float> gamma_;
    std::vector<float> beta_;

    std::vector<float> means_;
    std::vector<float> variances_;
};

}  // namespace vai_q
