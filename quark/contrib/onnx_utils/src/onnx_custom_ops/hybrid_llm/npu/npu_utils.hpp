/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#ifndef GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_NPU_UTIL
#define GUARD_ONNX_CUSTOM_OPS_HYBRID_LLM_NPU_UTIL
#include <ryzenai/ryzen_mm.h>

#include <ops/op_interface.hpp>

#include "common.hpp"
#include "ops/mladfrmsnorm/mladfrmsnorm.hpp"
#include "ops/ops_common/dtype_utils.h"

namespace ryzenai::onnx_utils {

constexpr auto kSeqLengthDefault = 128;

inline std::size_t alignTo4096(std::size_t size) {
  constexpr std::size_t alignment = 4096;
  return (size + alignment - 1) & ~(alignment - 1);
}

inline std::map<std::string, size_t> get_NPU_tensor_size(
  const std::vector<OpArgMap>& arg_map, std::string op_version
) {
  std::map<std::string, size_t> size_map;
  size_t pad_bo_size = 0;
  size_t c_bo_size = 0;
  for (auto arg : arg_map) {
    if (arg.arg_type == OpArgMap::OpArgType::INPUT) {
      size_map["in" + std::to_string(arg.onnx_arg_idx)] = arg.size;
    }
    if (op_version == "v1") {
      if (arg.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
        pad_bo_size = arg.size;
      }
    } else {
      if (arg.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
        size_map["scratch"] = arg.size;
      }
    }
    if (arg.arg_type == OpArgMap::OpArgType::OUTPUT) {
      c_bo_size = arg.size;
    }
  }
  c_bo_size = std::max(c_bo_size, pad_bo_size);
  size_map["out"] = c_bo_size;
  return size_map;
}

static inline size_t get_xrt_bo_size(const xrt::bo& bo) {
  // right now, if you query default constructed bo for size
  // it will crash
  // add helper function so we have a safe way to query
  if (!bo) {
    return 0;
  }

  return bo.size();
}

inline std::vector<std::uint8_t> init_rmsnorm_wts(
  ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>* rms_norm, float epsilon,
  const std::vector<float>& wts_data, const RyzenMM::Allocator& allocator
) {
  size_t num_el = wts_data.size();
  uint16_t eps = ryzenai::onnx_utils::float_to_bfloat16(epsilon);
  auto wts_vec = allocator.AllocateBuffer(num_el * sizeof(uint16_t));
  // std::vector<uint16_t> wts_vec(num_el);
  //  Convert floating point to bfloat16 using avx512

  ryzenai::float_buffer_to_bfloat16(
    wts_data.data(), num_el, (uint16_t*)wts_vec.Data()
  );  // K
  std::vector<size_t> wts_shape = {num_el};
  std::vector<size_t> eps_shape = {1};
  ::Tensor wts_T = {wts_vec.Data(), wts_shape, "uint16_t"};
  ::Tensor eps_T = {&eps, eps_shape, "uint16_t"};
  std::vector<::Tensor> const_Tensor;
  const_Tensor.push_back(wts_T);
  const_Tensor.push_back(eps_T);

  auto buffer = rms_norm->export_const_params(const_Tensor);
  wts_vec = {};
  return buffer[0];
}

/**
 * @brief In the new RMSNorm in DD, the first 64 bytes are reserved for epsilon
 * data so the actual const data starts after that
 *
 * @return uint16_t*
 */
inline uint16_t* getRmsNormConstData(uint16_t* base_addr) {
  return base_addr + 64 / sizeof(uint16_t);
}

}  // namespace ryzenai::onnx_utils
#endif
