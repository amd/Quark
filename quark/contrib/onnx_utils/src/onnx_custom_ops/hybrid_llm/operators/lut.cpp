// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "lut.hpp"

#include <any>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <type_traits>

#include "external_data.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"
#include "opUtils.h"
#include "ryzenai/onnx_utils/custom_ops_options.hpp"

namespace fs = std::filesystem;

namespace ryzenai::onnx_utils {

LutKernel::LutKernel(
  const OrtApi& ort_api, const OrtKernelInfo* info,
  const std::unordered_map<std::string, std::string>& session_configs
)
  : HybridKernel(ort_api, info, session_configs) {
  // printLog("Constructing LutKernel custom op...");

  // setup op attributes
  vocab_size_ = getAttribute<int64_t>("vocab_size");
  embedding_dim_ = getAttribute<int64_t>("embedding_dim");

  const auto& external_data_file = session_configs.at("external_data_file");

  auto header = proto::getHeader(session_configs);

  if (header->external_data().embedding()) {
    use_external_data_ = true;
    external_data_path_ = (fs::path(external_data_file).parent_path() /
                           header->external_data().filename())
                            .string();
  }

  if (use_external_data_) {
#ifdef _WIN32
    hFile_ = CreateFile(
      external_data_path_.c_str(),  // File name
      GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
    );

    const auto& hybrid_opt_embedding_mmap_str =
      session_configs.at("hybrid_opt_embedding_mmap");

    bool hybrid_opt_embedding_mmap = false;

    if (!hybrid_opt_embedding_mmap_str.empty()) {
      // mmap embedding table instead of loading into memory
      hybrid_opt_embedding_mmap = hybrid_opt_embedding_mmap_str != "0";
    }

    if (hFile_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("Could not open file");
    }

    // should be read from protobuf
    const auto& tensor = header->operators().at(nodeName()).data().at(0);
    auto tensor_offset = tensor.offset();
    auto tensor_size = tensor.size();

    if (hybrid_opt_embedding_mmap) {
      hMapping_ = CreateFileMapping(
        hFile_, nullptr,
        PAGE_READONLY,  // Read access
        0, 0,
        nullptr
      );  // No name

      if (hMapping_ == nullptr) {
        CloseHandle(hFile_);
        throw std::runtime_error("Could not create file mapping");
      }

      SYSTEM_INFO si;
      GetSystemInfo(&si);

      DWORD gran = si.dwAllocationGranularity;  // usually 65536

      uint64_t viewOffset = (tensor_offset / gran) * gran;
      size_t delta = tensor_offset - viewOffset;

      DWORD offset_hi = viewOffset >> 32;
      DWORD offset_lo = static_cast<std::uint32_t>(viewOffset);
      DWORD size = tensor_size + (delta);

      pBuf_ =
        MapViewOfFile(hMapping_, FILE_MAP_READ, offset_hi, offset_lo, size);

      if (pBuf_ == nullptr) {
        CloseHandle(hMapping_);
        CloseHandle(hFile_);
        throw std::runtime_error("Could not map view of file");
      }

      mapped_lut_ = true;
      external_lut_ptr_ = static_cast<const std::uint8_t*>(pBuf_) + delta;
    } else {
      embedding_ = RyzenMM::CPUAllocator<'LUT'>().AllocateBuffer(tensor_size);
      external_lut_ptr_ = static_cast<const std::uint8_t*>(embedding_.Data());

      DWORD read = 0;
      OVERLAPPED ol{};
      // Set the offset from the start of the file
      ol.Offset = static_cast<std::uint32_t>(tensor_offset);
      ol.OffsetHigh = tensor_offset >> 32;
      ReadFile(hFile_, (void*)embedding_.Data(), tensor_size, &read, &ol);
      CloseHandle(hFile_);
    }
#else
    abort();
#endif
  }

  printLog("Constructed ", nodeName(), " of type LutKernel Custom op\n ");
}
LutKernel::~LutKernel() {
  if (mapped_lut_) {
#ifdef _WIN32
    UnmapViewOfFile(pBuf_);
    CloseHandle(hMapping_);
    CloseHandle(hFile_);
#else
    abort();
#endif
  }
}

struct embedding_payload {
  const std::int64_t* indices_data_ptr;
  const std::uint16_t* lut_data_ptr;
  std::uint16_t* out_data_ptr;
  int64_t embedding_dim;
};

static void embedding_lut(void* data, size_t index) {
  const embedding_payload* payload = (embedding_payload*)data;

  const auto& indices_data_ptr = payload->indices_data_ptr;
  const auto& lut_data_ptr = payload->lut_data_ptr;
  const auto& base_out_data_ptr = payload->out_data_ptr;
  const auto& embedding_dim = payload->embedding_dim;

  auto lut_index = indices_data_ptr[index];
  auto out_data_ptr = &base_out_data_ptr[index * embedding_dim];
  // index can technically be in range [-vocab_size, vocab_size - 1]
  // lut_index = (lut_index + indices_dims.at(1)) % indices_dims.at(1);
  memcpy(
    out_data_ptr, &lut_data_ptr[lut_index * embedding_dim],
    embedding_dim * sizeof(std::uint16_t)
  );
}

void LutKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  const auto input_num = ctx.GetInputCount();

  // LUT, e.g. embeddings in float16/bfloat16/float32
  auto lut = ctx.GetInput(0);
  // typically [vocab_size, embedding_dim]
  // if in external file, will be [0]
  auto lut_dims = lut.GetTensorTypeAndShapeInfo().GetShape();
  bool use_external_lut =
    lut.GetTensorTypeAndShapeInfo().GetElementCount() == 0;

  if (!use_external_data_ && use_external_lut) {
    throw std::runtime_error(
      "external_data_file not specified and LUT not in memory!"
    );
  }

  const std::uint16_t* lut_data_ptr = nullptr;

  if (use_external_lut) {
    lut_data_ptr = (const std::uint16_t*)external_lut_ptr_;
  } else {
    auto lut_data = lut.GetTensorData<std::uint16_t>();
    lut_data_ptr = static_cast<const std::uint16_t*>(lut_data);
  }

  // typically 1D vector of [int64]
  auto indices = ctx.GetInput(1);
  auto indices_dims = indices.GetTensorTypeAndShapeInfo().GetShape();
  auto indices_count = indices.GetTensorTypeAndShapeInfo().GetElementCount();

  auto indices_data = indices.GetTensorData<std::int64_t>();
  const std::int64_t* indices_data_ptr =
    static_cast<const std::int64_t*>(indices_data);

  auto out_dims = indices_dims;

  out_dims.push_back(embedding_dim_);

  auto output = ctx.GetOutput(0, out_dims);  // Output

  // NOTE: expecting embedding to be float16 or bfloat16
  auto out_data = output.GetTensorMutableData<uint16_t>();
  std::uint16_t* out_data_ptr = static_cast<std::uint16_t*>(out_data);

  // serial implementation
  /*
  for (auto index = 0; index < indices_count; index++) {
    auto lut_index = *indices_data_ptr++;
    // index can technically be in range [-vocab_size, vocab_size - 1]
    // lut_index = (lut_index + indices_dims.at(1)) % indices_dims.at(1);
    memcpy(out_data_ptr, &lut_data_ptr[lut_index * embedding_dim_],
           embedding_dim_ * sizeof(std::uint16_t));

    out_data_ptr += embedding_dim_;
  }
  */

  embedding_payload payload = {
    indices_data_ptr, lut_data_ptr, out_data_ptr, embedding_dim_
  };

  ctx.ParallelFor(
    embedding_lut, static_cast<size_t>(indices_count), 0, &payload
  );
}

}  // namespace ryzenai::onnx_utils
